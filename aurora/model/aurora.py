"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import contextlib
import dataclasses
import warnings
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from aurora.batch import Batch
from aurora.model.compat import _adapt_checkpoint_air_pollution, _adapt_checkpoint_pretrained
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.lora import LoRAMode
from aurora.model.swin3d import BasicLayer3D, Swin3DTransformerBackbone

__all__ = [
    "Aurora",
    "AuroraPretrained",
    "AuroraSmallPretrained",
    "AuroraSmall",
    "Aurora12hPretrained",
    "AuroraHighRes",
    "AuroraAirPollution",
]


class Aurora(torch.nn.Module):
    """The Aurora model.

    Defaults to the 1.3 B parameter configuration.
    """

    default_checkpoint_repo = "microsoft/aurora"
    """str: Name of the HuggingFace repository to load the default checkpoint from."""

    default_checkpoint_name = "aurora-0.25-finetuned.ckpt"
    """str: Name of the default checkpoint."""

    def __init__(
        self,
        *,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        window_size: tuple[int, int, int] = (2, 6, 12),
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        max_history_size: int = 2,
        timestep: timedelta = timedelta(hours=6),
        stabilise_level_agg: bool = False,
        use_lora: bool = True,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        surf_stats: Optional[dict[str, tuple[float, float]]] = None,
        autocast: bool = False,
        level_condition: Optional[tuple[int | float, ...]] = None,
        dynamic_vars: bool = False,
        atmos_static_vars: bool = False,
        separate_perceiver: tuple[str, ...] = (),
        modulation_head: bool = False,
        positive_surf_vars: tuple[str, ...] = (),
        positive_atmos_vars: tuple[str, ...] = (),
        simulate_indexing_bug: bool = False,
    ) -> None:
        """Construct an instance of the model.

        Args:
            surf_vars (tuple[str, ...], optional): All surface-level variables supported by the
                model.
            static_vars (tuple[str, ...], optional): All static variables supported by the
                model.
            atmos_vars (tuple[str, ...], optional): All atmospheric variables supported by the
                model.
            window_size (tuple[int, int, int], optional): Vertical height, height, and width of the
                window of the underlying Swin transformer.
            encoder_depths (tuple[int, ...], optional): Number of blocks in each encoder layer.
            encoder_num_heads (tuple[int, ...], optional): Number of attention heads in each encoder
                layer. The dimensionality doubles after every layer. To keep the dimensionality of
                every head constant, you want to double the number of heads after every layer. The
                dimensionality of attention head of the first layer is determined by `embed_dim`
                divided by the value here. For all cases except one, this is equal to `64`.
            decoder_depths (tuple[int, ...], optional): Number of blocks in each decoder layer.
                Generally, you want this to be the reversal of `encoder_depths`.
            decoder_num_heads (tuple[int, ...], optional): Number of attention heads in each decoder
                layer. Generally, you want this to be the reversal of `encoder_num_heads`.
            latent_levels (int, optional): Number of latent pressure levels.
            patch_size (int, optional): Patch size.
            embed_dim (int, optional): Patch embedding dimension.
            num_heads (int, optional): Number of attention heads in the aggregation and
                deaggregation blocks. The dimensionality of these attention heads will be equal to
                `embed_dim` divided by this value.
            mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs.
            drop_rate (float, optional): Drop-out rate.
            drop_path (float, optional): Drop-path rate.
            enc_depth (int, optional): Number of Perceiver blocks in the encoder.
            dec_depth (int, optioanl): Number of Perceiver blocks in the decoder.
            dec_mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs in the
                decoder. The embedding dimensionality here is different, which is why this is a
                separate parameter.
            perceiver_ln_eps (float, optional): Epsilon in the perceiver layer norm. layers. Used
                to stabilise the model.
            max_history_size (int, optional): Maximum number of history steps. You can load
                checkpoints with a smaller `max_history_size`, but you cannot load checkpoints
                with a larger `max_history_size`.
            timestep (timedelta, optional): Timestep of the model. Defaults to 6 hours.
            stabilise_level_agg (bool, optional): Stabilise the level aggregation by inserting an
                additional layer normalisation. Defaults to `False`.
            use_lora (bool, optional): Use LoRA adaptation.
            lora_steps (int, optional): Use different LoRA adaptation for the first so-many roll-out
                steps.
            lora_mode (str, optional): LoRA mode. `"single"` uses the same LoRA for all roll-out
                steps, and `"all"` uses a different LoRA for every roll-out step. Defaults to
                `"single"`.
            surf_stats (dict[str, tuple[float, float]], optional): For these surface-level
                variables, adjust the normalisation to the given tuple consisting of a new location
                and scale.
            autocast (bool, optional): Use `torch.autocast` to reduce memory usage. Defaults to
                `False`.
            level_condition (tuple[int | float, ...], optional): Make the patch embeddings dependent
                on pressure level. If you want to enable this feature, provide a tuple of all
                possible pressure levels.
            dynamic_vars (bool, optional): Use dynamically generated static variables, like time
                of day. Defaults to `False`.
            atmos_static_vars (bool, optional): Also concatenate the static variables to the
                atmospheric variables. Defaults to `False`.
            separate_perceiver (tuple[str, ...], optional): In the decoder, use a separate Perceiver
                for specific atmospheric variables. This can be helpful at fine-tuning time to deal
                with variables that have a significantly different behaviour. If you want to enable
                this features, set this to the collection of variables that should be run on a
                separate Perceiver.
            modulation_head (bool, optional): Enable an additional head, the so-called modulation
                head, that can be used to predict the difference. Defaults to `False`.
            positive_surf_vars (tuple[str, ...], optional): Mark these surface-level variables as
                positive. Clamp them before running them through the encoder, and also clamp them
                when autoregressively rolling out the model. The variables are not clamped for the
                first roll-out step.
            positive_atmos_vars (tuple[str, ...], optional): Mark these atmospheric variables as
                positive. Clamp them before running them through the encoder, and also clamp them
                when autoregressively rolling out the model. The variables are not clamped for the
                first roll-out step.
            simulate_indexing_bug (bool, optional): Simulate an indexing bug that's present for the
                air pollution version of Aurora. This is necessary to obtain numerical equivalence
                to the original implementation. Defaults to `False`.
        """
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.surf_stats = surf_stats or dict()
        self.autocast = autocast
        self.max_history_size = max_history_size
        self.timestep = timestep
        self.use_lora = use_lora
        self.positive_surf_vars = positive_surf_vars
        self.positive_atmos_vars = positive_atmos_vars

        if self.surf_stats:
            warnings.warn(
                f"The normalisation statics for the following surface-level variables are manually "
                f"adjusted: {', '.join(sorted(self.surf_stats.keys()))}. "
                f"Please ensure that this is right!",
                stacklevel=2,
            )

        self.encoder = Perceiver3DEncoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            head_dim=embed_dim // num_heads,
            depth=enc_depth,
            latent_levels=latent_levels,
            max_history_size=max_history_size,
            perceiver_ln_eps=perceiver_ln_eps,
            stabilise_level_agg=stabilise_level_agg,
            level_condition=level_condition,
            dynamic_vars=dynamic_vars,
            atmos_static_vars=atmos_static_vars,
            simulate_indexing_bug=simulate_indexing_bug,
        )

        self.backbone = Swin3DTransformerBackbone(
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
            drop_rate=drop_rate,
            use_lora=use_lora,
            lora_steps=lora_steps,
            lora_mode=lora_mode,
        )

        self.decoder = Perceiver3DDecoder(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            # Concatenation at the backbone end doubles the dim.
            embed_dim=embed_dim * 2,
            head_dim=embed_dim * 2 // num_heads,
            num_heads=num_heads,
            depth=dec_depth,
            # Because of the concatenation, high ratios are expensive.
            # We use a lower ratio here to keep the memory in check.
            mlp_ratio=dec_mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
            level_condition=level_condition,
            separate_perceiver=separate_perceiver,
            modulation_head=modulation_head,
        )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        transformed_batch = batch

        # Clamp positive variables.
        if self.positive_surf_vars:
            transformed_batch = dataclasses.replace(
                transformed_batch,
                surf_vars={
                    k: v.clamp(min=0) if k in self.positive_surf_vars else v
                    for k, v in batch.surf_vars.items()
                },
            )
        if self.positive_atmos_vars:
            transformed_batch = dataclasses.replace(
                transformed_batch,
                atmos_vars={
                    k: v.clamp(min=0) if k in self.positive_atmos_vars else v
                    for k, v in batch.atmos_vars.items()
                },
            )

        transformed_batch = self._pre_encoder_hook(transformed_batch)

        x = self.encoder(
            transformed_batch,
            lead_time=self.timestep,
        )
        with torch.autocast(device_type="cuda") if self.autocast else contextlib.nullcontext():
            x = self.backbone(
                x,
                lead_time=self.timestep,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
        pred = self.decoder(
            x,
            batch,
            lead_time=self.timestep,
            patch_res=patch_res,
        )

        # Remove batch and history dimension from static variables.
        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # Insert history dimension in prediction. The time should already be right.
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        pred = self._post_decoder_hook(batch, pred)

        # Clamp positive variables.
        if self.positive_surf_vars and pred.metadata.rollout_step > 1:
            pred = dataclasses.replace(
                pred,
                surf_vars={
                    k: v.clamp(min=0) if k in self.positive_surf_vars else v
                    for k, v in pred.surf_vars.items()
                },
            )
        if self.positive_atmos_vars and pred.metadata.rollout_step > 1:
            pred = dataclasses.replace(
                pred,
                atmos_vars={
                    k: v.clamp(min=0) if k in self.positive_atmos_vars else v
                    for k, v in pred.atmos_vars.items()
                },
            )

        pred = pred.unnormalise(surf_stats=self.surf_stats)

        return pred

    def _pre_encoder_hook(self, batch: Batch) -> Batch:
        """Transform the batch before it goes through the encoder."""
        return batch

    def _post_decoder_hook(self, batch: Batch, pred: Batch) -> Batch:
        """Transform the prediction right after the decoder."""
        return pred

    def load_checkpoint(
        self,
        repo: Optional[str] = None,
        name: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        """Load a checkpoint from HuggingFace.

        Args:
            repo (str, optional): Name of the repository of the form `user/repo`.
            name (str, optional): Path to the checkpoint relative to the root of the repository,
                e.g. `checkpoint.cpkt`.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        repo = repo or self.default_checkpoint_repo
        name = name or self.default_checkpoint_name
        path = hf_hub_download(repo_id=repo, filename=name)
        self.load_checkpoint_local(path, strict=strict)

    def load_checkpoint_local(self, path: str, strict: bool = True) -> None:
        """Load a checkpoint directly from a file.

        Args:
            path (str): Path to the checkpoint.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        # Assume that all parameters are either on the CPU or on the GPU.
        device = next(self.parameters()).device
        d = torch.load(path, map_location=device, weights_only=True)

        d = self._adapt_checkpoint(d)

        # Check if the history size is compatible and adjust weights if necessary.
        current_history_size = d["encoder.surf_token_embeds.weights.2t"].shape[2]
        if self.max_history_size > current_history_size:
            self.adapt_checkpoint_max_history_size(d)
        elif self.max_history_size < current_history_size:
            raise AssertionError(
                f"Cannot load checkpoint with `max_history_size` {current_history_size} "
                f"into model with `max_history_size` {self.max_history_size}."
            )

        self.load_state_dict(d, strict=strict)

    def _adapt_checkpoint(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Adapt an existing checkpoint to make it compatible with the current version of the model.

        Args:
            d (dict[str, torch.Tensor]): Checkpoint.

        Return:
            dict[str, torch.Tensor]: Adapted checkpoint.
        """
        return _adapt_checkpoint_pretrained(self.patch_size, d)

    def adapt_checkpoint_max_history_size(self, checkpoint: dict[str, torch.Tensor]) -> None:
        """Adapt a checkpoint with smaller `max_history_size` to a model with a larger
        `max_history_size` than the current model.

        If a checkpoint was trained with a larger `max_history_size` than the current model,
        this function will assert fail to prevent loading the checkpoint. This is to
        prevent loading a checkpoint which will likely cause the checkpoint to degrade is
        performance.

        This implementation copies weights from the checkpoint to the model and fills zeros
        for the new history width dimension. It mutates `checkpoint`.
        """
        for name, weight in list(checkpoint.items()):
            # We only need to adapt the patch embedding in the encoder.
            enc_surf_embedding = name.startswith("encoder.surf_token_embeds.weights.")
            enc_atmos_embedding = name.startswith("encoder.atmos_token_embeds.weights.")
            if enc_surf_embedding or enc_atmos_embedding:
                # This shouldn't get called with current logic but leaving here for future proofing
                # and in cases where its called outside current context.
                if not (weight.shape[2] <= self.max_history_size):
                    raise AssertionError(
                        f"Cannot load checkpoint with `max_history_size` {weight.shape[2]} "
                        f"into model with `max_history_size` {self.max_history_size}."
                    )

                # Initialize the new weight tensor.
                new_weight = torch.zeros(
                    (weight.shape[0], 1, self.max_history_size, weight.shape[3], weight.shape[4]),
                    device=weight.device,
                    dtype=weight.dtype,
                )
                # Copy the existing weights to the new tensor by duplicating the histories provided
                # into any new history dimensions. The rest remains at zero.
                new_weight[:, :, : weight.shape[2]] = weight

                checkpoint[name] = new_weight

    def configure_activation_checkpointing(self):
        """Configure activation checkpointing.

        This is required in order to compute gradients without running out of memory.
        """
        apply_activation_checkpointing(self, check_fn=lambda x: isinstance(x, BasicLayer3D))


class AuroraPretrained(Aurora):
    """Pretrained version of Aurora."""

    default_checkpoint_name = "aurora-0.25-pretrained.ckpt"

    def __init__(
        self,
        *,
        use_lora: bool = False,
        **kw_args,
    ) -> None:
        Aurora.__init__(
            self,
            use_lora=use_lora,
            **kw_args,
        )


class AuroraSmallPretrained(Aurora):
    """Small pretrained version of Aurora.

    Should only be used for debugging.
    """

    default_checkpoint_name = "aurora-0.25-small-pretrained.ckpt"

    def __init__(
        self,
        *,
        encoder_depths: tuple[int, ...] = (2, 6, 2),
        encoder_num_heads: tuple[int, ...] = (4, 8, 16),
        decoder_depths: tuple[int, ...] = (2, 6, 2),
        decoder_num_heads: tuple[int, ...] = (16, 8, 4),
        embed_dim: int = 256,
        num_heads: int = 8,
        use_lora: bool = False,
        **kw_args,
    ) -> None:
        Aurora.__init__(
            self,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_lora=use_lora,
            **kw_args,
        )


AuroraSmall = AuroraSmallPretrained  #: Alias for backwards compatibility


class Aurora12hPretrained(Aurora):
    """Pretrained version of Aurora with time step 12 hours."""

    default_checkpoint_name = "aurora-0.25-12h-pretrained.ckpt"

    def __init__(
        self,
        *,
        timestep: timedelta = timedelta(hours=12),
        use_lora: bool = False,
        **kw_args,
    ) -> None:
        Aurora.__init__(
            self,
            timestep=timestep,
            use_lora=use_lora,
            **kw_args,
        )


class AuroraHighRes(Aurora):
    """High-resolution version of Aurora."""

    default_checkpoint_name = "aurora-0.1-finetuned.ckpt"

    def __init(
        self,
        *,
        patch_size: int = 10,
        encoder_depths: tuple[int, ...] = (6, 8, 8),
        decoder_depths: tuple[int, ...] = (8, 8, 6),
        **kw_args,
    ) -> None:
        Aurora.__init__(
            self,
            patch_size=patch_size,
            encoder_depths=encoder_depths,
            decoder_depths=decoder_depths,
            **kw_args,
        )


class AuroraAirPollution(Aurora):
    """Fine-tuned version of Aurora for air pollution."""

    default_checkpoint_name = "aurora-0.4-air-pollution.ckpt"

    _predict_difference_history_dim_lookup = {
        "pm1": 0,
        "pm2p5": 0,
        "pm10": 0,
        "co": 1,
        "tcco": 1,
        "no": 0,
        "tc_no": 0,
        "no2": 0,
        "tcno2": 0,
        "so2": 1,
        "tcso2": 1,
        "go3": 1,
        "gtco3": 1,
    }
    """dict[str, int]: For every variable that we want to predict the difference for, the index
    into the history dimension that should be used when predicting the difference."""

    def __init__(
        self,
        *,
        surf_vars: tuple[str, ...] = (
            ("2t", "10u", "10v", "msl")
            + ("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2")
        ),
        static_vars: tuple[str, ...] = (
            ("lsm", "z", "slt")
            + ("static_ammonia", "static_ammonia_log", "static_co", "static_co_log")
            + ("static_nox", "static_nox_log", "static_so2", "static_so2_log")
        ),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q", "co", "no", "no2", "go3", "so2"),
        patch_size: int = 3,
        timestep: timedelta = timedelta(hours=12),
        level_condition: Optional[tuple[int | float, ...]] = (
            (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
        ),
        dynamic_vars: bool = True,
        atmos_static_vars: bool = True,
        separate_perceiver: tuple[str, ...] = ("co", "no", "no2", "go3", "so2"),
        modulation_head: bool = True,
        positive_surf_vars: tuple[str, ...] = (
            ("pm1", "pm2p5", "pm10", "tcco", "tc_no", "tcno2", "gtco3", "tcso2")
        ),
        positive_atmos_vars: tuple[str, ...] = ("co", "no", "no2", "go3", "so2"),
        simulate_indexing_bug: bool = True,
        **kw_args,
    ) -> None:
        Aurora.__init__(
            self,
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            timestep=timestep,
            level_condition=level_condition,
            dynamic_vars=dynamic_vars,
            atmos_static_vars=atmos_static_vars,
            separate_perceiver=separate_perceiver,
            modulation_head=modulation_head,
            positive_surf_vars=positive_surf_vars,
            positive_atmos_vars=positive_atmos_vars,
            simulate_indexing_bug=simulate_indexing_bug,
            **kw_args,
        )

        self.surf_feature_combiner = torch.nn.ParameterDict(
            {v: nn.Linear(2, 1, bias=True) for v in self.positive_surf_vars}
        )
        self.atmos_feature_combiner = torch.nn.ParameterDict(
            {v: nn.Linear(2, 1, bias=True) for v in self.positive_atmos_vars}
        )
        for p in (*self.surf_feature_combiner.values(), *self.atmos_feature_combiner.values()):
            nn.init.constant_(p.weight, 0.5)
            nn.init.zeros_(p.bias)

    def _pre_encoder_hook(self, batch: Batch) -> Batch:
        # Transform the spikey variables with a specific log-transform before feeding them
        # to the encoder. See the paper for a motivation for the precise form of the transform.

        eps = 1e-4
        divisor = -np.log(eps)

        def _transform(z: torch.Tensor, feature_combiner: nn.Module) -> torch.Tensor:
            return feature_combiner(
                torch.stack(
                    [
                        z.clamp(min=0, max=2.5),
                        (torch.log(z.clamp(min=eps)) - np.log(eps)) / divisor,
                    ],
                    dim=-1,
                )
            )[..., 0]

        return dataclasses.replace(
            batch,
            surf_vars={
                k: _transform(v, self.surf_feature_combiner[k])
                if k in self.surf_feature_combiner
                else v
                for k, v in batch.surf_vars.items()
            },
            atmos_vars={
                k: _transform(v, self.atmos_feature_combiner[k])
                if k in self.atmos_feature_combiner
                else v
                for k, v in batch.atmos_vars.items()
            },
        )

    def _post_decoder_hook(self, batch: Batch, pred: Batch) -> Batch:
        # For this version of the model, we predict the difference. Specifically w.r.t. which
        # previous timestep (12 hours ago or 24 hours ago) is given by
        # `Aurora._predict_difference_history_dim_lookup`.

        dim_lookup = AuroraAirPollution._predict_difference_history_dim_lookup

        def _transform(
            prev: dict[str, torch.Tensor],
            model: dict[str, torch.Tensor],
            name: str,
        ) -> torch.Tensor:
            if name in dim_lookup:
                return model[name] + (1 + model[f"{name}_mod"]) * prev[name][:, dim_lookup[name]]
            else:
                return model[name]

        pred = dataclasses.replace(
            pred,
            surf_vars={k: _transform(batch.surf_vars, pred.surf_vars, k) for k in batch.surf_vars},
            atmos_vars={
                k: _transform(batch.atmos_vars, pred.atmos_vars, k) for k in batch.atmos_vars
            },
        )

        # When using LoRA, the lower-atmospheric levels of SO2 can be problematic and blow up.
        # We attempt to fix that by some very aggressive output clipping.
        if self.use_lora:
            parts: list[torch.Tensor] = []
            for i, level in enumerate(pred.metadata.atmos_levels):
                section = pred.atmos_vars["so2"][..., i, :, :]
                if level >= 850:
                    section = section.clamp(max=1)
                parts.append(section)
            pred.atmos_vars["so2"] = torch.stack(parts, dim=-3)

        return pred

    def _adapt_checkpoint(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        d = Aurora._adapt_checkpoint(self, d)
        d = _adapt_checkpoint_air_pollution(self.patch_size, d)
        return d

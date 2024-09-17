"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import contextlib
import dataclasses
import warnings
from datetime import timedelta
from functools import partial
from typing import Any, Callable, Optional

import torch
from huggingface_hub import hf_hub_download
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.lora import LoRAMode
from aurora.model.swin3d import BasicLayer3D, Swin3DTransformerBackbone

__all__ = ["Aurora", "AuroraSmall", "AuroraHighRes"]


class Aurora(torch.nn.Module):
    """The Aurora model.

    Defaults to to the 1.3 B parameter configuration.
    """

    def __init__(
        self,
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
        use_lora: bool = True,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        surf_stats: Optional[dict[str, tuple[float, float]]] = None,
        autocast: bool = False,
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
            max_history_size (int, optional): Maximum number of history steps.
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
        """
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.surf_stats = surf_stats or dict()
        self.autocast = autocast
        self.max_history_size = max_history_size

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

        x = self.encoder(
            batch,
            lead_time=timedelta(hours=6),
        )
        with torch.autocast(device_type="cuda") if self.autocast else contextlib.nullcontext():
            x = self.backbone(
                x,
                lead_time=timedelta(hours=6),
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
            )
        pred = self.decoder(
            x,
            batch,
            lead_time=timedelta(hours=6),
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

        pred = pred.unnormalise(surf_stats=self.surf_stats)

        return pred

    def load_checkpoint(self, repo: str, name: str, strict: bool = True, adapt_fn: Callable[[Any], Any] = 'default', adapt_history: Callable[[Any], Any] = 'default') -> None:
        """Load a checkpoint from HuggingFace.

        Args:
            repo (str): Name of the repository of the form `user/repo`.
            name (str): Path to the checkpoint relative to the root of the repository, e.g.
                `checkpoint.cpkt`.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
            adapt_fn (Callable[[Any], Any], optional): Function to adapt the checkpoint to the current model.
                Defaults to `self.adapt_checkpoint`. Pass `None` to skip adaptation.
            adapt_history (Callable[[Any], Any], optional): Function to fill the history of the checkpoint when 
                Model is larger than checkpoint. Defaults to `self.adapt_checkpoint_max_history_size`. Pass `None` to skip adaptation.
        """
        if adapt_fn == 'default':
            adapt_fn = self.adapt_checkpoint
        
        if adapt_history == 'default':
            adapt_history = self.adapt_checkpoint_max_history_size

        path = hf_hub_download(repo_id=repo, filename=name)
        self.load_checkpoint_local(path, strict=strict, adapt_fn=adapt_fn, adapt_history=adapt_history)

    def load_checkpoint_local(self, path: str, strict: bool = True, adapt_fn: Callable[[Any], Any] = 'default', adapt_history: Callable[[Any], Any] = 'default') -> None:
        """Load a checkpoint directly from a file.

        Args:
            path (str): Path to the checkpoint.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
            adapt_fn (Callable[[Any], Any], optional): Function to adapt the checkpoint to the current model.
                Defaults to `self.adapt_checkpoint`. Pass `None` to skip adaptation.
            adapt_history (Callable[[Any], Any], optional): Function to fill the history of the checkpoint when 
                Model is larger than checkpoint. Defaults to `self.adapt_checkpoint_max_history_size`. Pass `None` to skip adaptation.
        """
        if adapt_fn == 'default':
            adapt_fn = self.adapt_checkpoint

        if adapt_history == 'default':
            adapt_history = self.adapt_checkpoint_max_history_size
            
        # Assume that all parameters are either on the CPU or on the GPU.
        device = next(self.parameters()).device

        d = torch.load(path, map_location=device, weights_only=True)

        # Adapt the checkpoint using the provided function, if not None
        if adapt_fn is not None:
            d = adapt_fn(d)
        
        # Adapt the checkpoint history size using the provided function, if not None
        if adapt_history is not None:
            d = adapt_history(d)

        self.load_state_dict(d, strict=strict)

    def adapt_checkpoint_max_history_size(self, checkpoint) -> Any:
        """Adapt a checkpoint with smaller max_history_size to a model with a larger max_history_size
        than the current model.
        
        If a checkpoint was trained with a larger max_history_size than the current model, this function
        will assert fail to prevent loading the checkpoint. This is to prevent loading a checkpoint
        which will likely cause the checkpoint to degrade is performance. 
        
        This implementation copies weights from the checkpoint to the model, duplicating the weights. 
        """
        
        #find all weights with prefix "encoder.surf_token_embeds.weights."
        for name, weight in list(checkpoint.items()):
            if name.startswith("encoder.surf_token_embeds.weights.") or name.startswith("encoder.atmos_token_embeds.weights."):
                assert weight.shape[2] >= self.max_history_size, f"Cannot load checkpoint with max_history_size {weight.shape[2]} \
                    into model with max_history_size {self.max_history_size} for weight {name}"   
            
                # Initialize the new weight tensor with zeros
                new_weight = torch.zeros((weight.shape[0], 1, self.max_history_size, weight.shape[3], weight.shape[4]))
                # Copy the existing weights to the new tensor my duplicating the histories provided in to any new history dimensions
                for j in range(new_weight.shape[2]):
                    new_weight[:, :, j, :, :] = weight[:, :, j % weight.shape[2], :, :]
                checkpoint[name] = new_weight
        
        return checkpoint
   
    @staticmethod 
    def _adapt_checkpoint_prefix(checkpoint) -> Any:
        """Adapt a checkpoint with a different prefix to the current model.
        
        If a checkpoint was trained with a different prefix than the current model, this function
        will remove the prefix from the checkpoint so that it can be loaded.  
        """
        # Remove possibly prefix from the keys.
        for k, v in list(checkpoint.items()):
            if k.startswith("net."):
                del checkpoint[k]
                checkpoint[k[4:]] = v
        return checkpoint 
   
    def _adapt_checkpoint_parametrization(self, checkpoint) -> Any:
        # Convert the ID-based parametrization to a name-based parametrization.
        if "encoder.surf_token_embeds.weight" in checkpoint:
            weight = checkpoint["encoder.surf_token_embeds.weight"]
            del checkpoint["encoder.surf_token_embeds.weight"]
        
            assert weight.shape[1] == 4 + 3
            for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
                checkpoint[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]
    
        if "encoder.atmos_token_embeds.weight" in checkpoint:
            weight = checkpoint["encoder.atmos_token_embeds.weight"]
            del checkpoint["encoder.atmos_token_embeds.weight"]
        
            assert weight.shape[1] == 5
            for i, name in enumerate(("z", "u", "v", "t", "q")):
                checkpoint[f"encoder.atmos_token_embeds.weights.{name}"] = weight[:, [i]]

        if "decoder.surf_head.weight" in checkpoint:
            weight = checkpoint["decoder.surf_head.weight"]
            bias = checkpoint["decoder.surf_head.bias"]
            del checkpoint["decoder.surf_head.weight"]
            del checkpoint["decoder.surf_head.bias"]

            assert weight.shape[0] == 4 * self.patch_size**2
            assert bias.shape[0] == 4 * self.patch_size**2
            weight = weight.reshape(self.patch_size**2, 4, -1)
            bias = bias.reshape(self.patch_size**2, 4)

            for i, name in enumerate(("2t", "10u", "10v", "msl")):
                checkpoint[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
                checkpoint[f"decoder.surf_heads.{name}.bias"] = bias[:, i]

        if "decoder.atmos_head.weight" in checkpoint:
            weight = checkpoint["decoder.atmos_head.weight"]
            bias = checkpoint["decoder.atmos_head.bias"]
            del checkpoint["decoder.atmos_head.weight"]
            del checkpoint["decoder.atmos_head.bias"]

            assert weight.shape[0] == 5 * self.patch_size**2
            assert bias.shape[0] == 5 * self.patch_size**2
            weight = weight.reshape(self.patch_size**2, 5, -1)
            bias = bias.reshape(self.patch_size**2, 5)

            for i, name in enumerate(("z", "u", "v", "t", "q")):
                checkpoint[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
                checkpoint[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]
        return checkpoint 
    
    def adapt_checkpoint(self, checkpoint) -> Any:
        """Adapt a checkpoint to the current model.
        
        Current model has a different structure than the model that was used to train the checkpoint
        that is being loaded. This function adapts the checkpoint to the current model so that it can 
        be loaded.  
        """
        # You can safely ignore all cumbersome processing below. We modified the model after we
        # trained it. The code below manually adapts the checkpoints, so the checkpoints are
        # compatible with the new model.

        checkpoint = Aurora._adapt_checkpoint_prefix(checkpoint) 
        checkpoint = self._adapt_checkpoint_parametrization(checkpoint)
        return checkpoint 
    
    def configure_activation_checkpointing(self):
        """Configure activation checkpointing.

        This is required in order to compute gradients without running out of memory.
        """
        apply_activation_checkpointing(self, check_fn=lambda x: isinstance(x, BasicLayer3D))


AuroraSmall = partial(
    Aurora,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
)

AuroraHighRes = partial(
    Aurora,
    patch_size=10,
    encoder_depths=(6, 8, 8),
    decoder_depths=(8, 8, 6),
    # One particular static variable requires a different normalisation.
    surf_stats={"z": (-3.270407e03, 6.540335e04)},
)

"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import timedelta
from functools import partial

from torch import nn

from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.swin3d import Int3Tuple, Swin3DTransformerBackbone

__all__ = ["Aurora"]

VariableList = tuple[str, ...]
"""type: Tuple of variable names."""


class Aurora(nn.Module):
    """The Aurora model.

    Defaults to to the 1.3 B parameter configuration.
    """

    def __init__(
        self,
        surf_vars: VariableList = ("2t", "10u", "10v", "msl"),
        static_vars: VariableList = ("lsm", "z", "slt"),
        atmos_vars: VariableList = ("z", "u", "v", "t", "q"),
        window_size: Int3Tuple = (2, 6, 12),
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
    ) -> None:
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size

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

        Raises:
            ValueError: If no metric is provided.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        batch = batch.normalise().crop(patch_size=self.patch_size)

        H, W = batch.spatial_shape
        patch_res: Int3Tuple = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        x = self.encoder(
            batch,
            lead_time=timedelta(hours=6),
        )
        x = self.backbone(
            x,
            lead_time=timedelta(hours=6),
            patch_res=patch_res,
            rollout_step=0,
        )
        pred = self.decoder(
            x,
            batch,
            lead_time=timedelta(hours=6),
            patch_res=patch_res,
        )

        # TODO: Ensure time dim is present and time is right.

        pred = pred.unnormalise()

        return pred


AuroraSmall = partial(
    Aurora,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
)

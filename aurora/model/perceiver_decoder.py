"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import timedelta

import torch
from einops import rearrange
from torch import nn

from aurora.batch import Batch, Metadata
from aurora.model.fourier import levels_expansion
from aurora.model.helpers import (
    Int3Tuple,
    check_lat_lon_dtype,
    create_var_map,
    get_ids_for_var_map,
    init_weights,
    unpatchify,
)
from aurora.model.perceiver import PerceiverResampler


class Perceiver3DDecoder(nn.Module):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver IO architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        perceiver_ln_eps: float = 1e-5,
    ) -> None:
        """Initialize the MultiScaleDecoder.

        Args:
            surf_vars (tuple[str, ...]): Names of surface-level variables.
            atmos_vars (tuple[str, ...]): Names of atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to 4.
            embed_dim (int, optional): Embedding dim. Defaults to 1024.
            level_embed_dim (int): Embedding dim for the pressure levels. Defaults to 1024.
            depth (int, optional): Number of Perceiver cross-attention + feedforward blocks.
                Defaults to 1.
            head_dim (int, optional): Dimension of the attention heads used in the aggregation
                blocks. Defaults to 64.
            num_heads (int, optional): Number of attention heads used in the aggregation blocks.
                Defaults to 8.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimensionality.
                Defaults to 4.0.
            drop_rate (float, optional): Drop-out rate for input patches. Defaults to 0.0.
            perceiver_ln_eps (float, optional): Layer norm epsilon for the Perceiver blocks.
        """
        super().__init__()

        self.patch_size = patch_size
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.surf_var_map = create_var_map(surf_vars)
        self.atmos_var_map = create_var_map(atmos_vars)
        self.embed_dim = embed_dim

        self.level_decoder = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            residual_latent=True,
            ln_eps=perceiver_ln_eps,
        )

        self.surf_head = nn.Linear(embed_dim, len(surf_vars) * patch_size**2)
        self.atmos_head = nn.Linear(embed_dim, len(atmos_vars) * patch_size**2)

        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        self.apply(init_weights)

    def deaggregate_levels(self, level_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """De-aggregate pressure level information.

        Args:
            level_embed (torch.Tensor): `(B, L, C, D)`
            x (torch.Tensor): `(B, L, C', D)`

        Returns:
            torch.Tensor: `(B, L, C, D)`
        """
        B, L, C, D = level_embed.shape
        level_embed = level_embed.flatten(0, 1)  # (BxL, C, D)
        x = x.flatten(0, 1)  # (BxL, C', D)
        _msg = f"Batch size mismatch. Found {level_embed.size(0)} and {x.size(0)}."
        assert level_embed.size(0) == x.size(0), _msg
        assert len(level_embed.shape) == 3, f"Expected 3 dims, found {level_embed.dims()}."
        assert x.dim() == 3, f"Expected 3 dims, found {x.dim()}."

        x = self.level_decoder(level_embed, x)  # (BxL, C, D)
        x = x.reshape(B, L, C, D)
        return x

    def forward(
        self,
        x: torch.Tensor,
        batch: Batch,
        pres: Int3Tuple,
        lead_time: timedelta,
    ) -> Batch:
        """Forward pass of MultiScaleEncoder.

        Args:
            x (torch.Tensor): `(B, L, D)`.
            metadata (Metadata): Metadata information.

        Returns:
            torch.Tensor: Predictions for the surface-level variables of shape `(B, V_S, H, W)`.
            torch.Tensor: Predictions for the atmospheric variables of shape `(V, V_A, C_A, H, W)`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        # Compress the latent dimension from the U-net skip concatenation.
        B, L, D = x.shape

        # Extract the lat, lon and convert to float32.
        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        H, W = lat.shape[0], lon.shape[-1]

        # Unwrap the latent level dimension.
        x = rearrange(x, "B (C H W) D -> B (H W) C D", C=pres[0], H=pres[1], W=pres[2])

        # Decode surface vars.
        x_surf = self.surf_head(x[..., :1, :])  # (B, L, 1, V_S*p*p)
        surf_var_ids = get_ids_for_var_map(surf_vars, self.surf_var_map, x_surf.device)
        surf_preds = unpatchify(x_surf, len(self.surf_vars), H, W, self.patch_size)[:, surf_var_ids]
        surf_preds = surf_preds.squeeze(2)  # (B, V_S, H, W)

        # Embed the atmospheric levels.
        atmos_levels_encode = levels_expansion(
            torch.tensor(atmos_levels, device=x.device), self.embed_dim
        ).to(dtype=x.dtype)
        levels_embed = self.atmos_levels_embed(atmos_levels_encode)  # (C_A, D)

        # De-aggregate the hidden levels into the physical levels.
        levels_embed = levels_embed.expand(B, x.size(1), -1, -1)
        x_atmos = self.deaggregate_levels(levels_embed, x[..., 1:, :])  # (B, L, C_A, D)

        # Decode the atmospheric vars.
        x_atmos = self.atmos_head(x_atmos)  # (B, L, C_A, V_A*p*p)
        atmos_var_ids = get_ids_for_var_map(atmos_vars, self.atmos_var_map, x.device)
        atmos_preds = unpatchify(x_atmos, len(self.atmos_vars), H, W, self.patch_size)
        atmos_preds = atmos_preds[:, atmos_var_ids]

        return Batch(
            {v: surf_preds[:, i] for i, v in enumerate(surf_vars)},
            batch.static_vars,
            {v: atmos_preds[:, i] for i, v in enumerate(atmos_vars)},
            Metadata(
                lat,
                lon,
                tuple(t + lead_time for t in batch.metadata.time),
                atmos_levels,
            ),
        )

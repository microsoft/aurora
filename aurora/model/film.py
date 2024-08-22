"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

`AdaptiveLayerNorm` was inspired by the following file:

    https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101
"""

import torch
from torch import nn

__all__ = ["AdaptiveLayerNorm"]


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalisation with scale and shift modulation."""

    def __init__(self, dim: int, context_dim: int, scale_bias: float = 0) -> None:
        """Initialise.

        Args:
            dim (int): Input dimension.
            context_dim (int): Dimension of the conditioning signal.
            scale_bias (float, optional): Scale bias to add to the scaling factor. Defaults to `0`.
        """
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, dim * 2))
        self.scale_bias = scale_bias

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the weights."""
        nn.init.zeros_(self.ln_modulation[-1].weight)
        nn.init.zeros_(self.ln_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, L, D)`.
            c (torch.Tensor): Conditioning tensor of shape `(B, D)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, L, D)`.
        """
        shift, scale = self.ln_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        return self.ln(x) * (self.scale_bias + scale) + shift

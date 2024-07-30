"""
Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Basic blocks for the Perceiver architecture.

The code borrows elements from the following repositories:
https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, dim, hidden_features: int, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PerceiverAttention(nn.Module):
    """Cross attention module from the Perceiver architecture."""

    def __init__(
        self, latent_dim: int, context_dim: int, head_dim: int = 64, num_heads: int = 8
    ) -> None:
        super().__init__()
        self.inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(latent_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, latent_dim, bias=False)

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Run the cross-attention module.

        Args:
            latents (:class:`torch.Tensor`): Latent features of shape `(B, L1, Latent_D)`
                where typically `L1 < L2` and `Latent_D <= Context_D`. `Latent_D` is equal to
                `self.latent_dim`.
            x (:class:`torch.Tensor`): Context features of shape `(B, L2, Context_D)`.

        Returns:
            :class:`torch.Tensor`: Latent values of shape `(B, L1, Latent_D)`.
        """
        h = self.num_heads

        q = self.to_q(latents)  # (B, L1, D2) -> (B, L1, D)
        k, v = self.to_kv(x).chunk(2, dim=-1)  # (B, L2, D1) -> (B, L2, D) x 2
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L1 D -> B L1 (H D)")  # (B, L1, D)
        return self.to_out(out)  # (B, L1, Latent_D)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler module from the Flamingo paper."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop=0.0,
        residual_latent: bool = True,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.residual_latent = residual_latent
        self.layers = nn.ModuleList([])
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            latent_dim=latent_dim,
                            context_dim=context_dim,
                            head_dim=head_dim,
                            num_heads=num_heads,
                        ),
                        MLP(dim=latent_dim, hidden_features=mlp_hidden_dim, dropout=drop),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                    ]
                )
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Run the module.

        Args:
            latents (:class:`torch.Tensor`): Latent features of shape `(B, L1, D1)`.
            x (:class:`torch.Tensor`): Context features of shape `(B, L2, D1)`.

        Returns:
            torch.Tensor: Latent features of shape `(B, L1, D1)`.
        """
        for attn, ff, ln1, ln2 in self.layers:
            # We use post-res-norm like in Swin v2 and most Transformer architectures these days.
            # This empirically works better than the pre-norm used in the original Perceiver.
            attn_out = ln1(attn(latents, x))
            # HuggingFace suggests using non-residual attention in Perceiver might
            # work better when the semantics of the query and the output are different.
            # https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/perceiver/modeling_perceiver.py#L398
            latents = attn_out + latents if self.residual_latent else attn_out
            latents = ln2(ff(latents)) + latents
        return latents

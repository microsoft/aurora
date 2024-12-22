"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Basic blocks for the Perceiver architecture.

The code borrows elements from the following files:

    https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
    https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py

These files are licenced under respectively the following two licences:

    MIT License

    Copyright (c) 2021 Phil Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    MIT License

    Copyright (c) 2023 Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf
    Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev,
    Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, Ludwig Schmidt.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["MLP", "PerceiverResampler"]


class MLP(nn.Module):
    """A simple one-hidden-layer MLP."""

    def __init__(self, dim: int, hidden_features: int, dropout: float = 0.0) -> None:
        """Initialise.

        Args:
            dim (int): Input dimensionality.
            hidden_features (int): Width of the hidden layer.
            dropout (float, optional): Drop-out rate. Defaults to no drop-out.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""
        return self.net(x)


class PerceiverAttention(nn.Module):
    """Cross attention module from the Perceiver architecture."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
        ln_k_q: bool = False,
    ) -> None:
        """Initialise.

        Args:
            latent_dim (int): Dimensionality of the latent features given as input.
            context_dim (int): Dimensionality of the context features also given as input.
            head_dim (int): Attention head dimensionality.
            num_heads (int): Number of heads.
            ln_k_q (bool): Apply an extra layer norm. to the keys and queries.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.to_q = nn.Linear(latent_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, latent_dim, bias=False)

        if ln_k_q:
            self.ln_k = nn.LayerNorm(num_heads * head_dim)
            self.ln_q = nn.LayerNorm(num_heads * head_dim)
        else:
            self.ln_k = lambda x: x
            self.ln_q = lambda x: x

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

        q = self.to_q(latents)  # (B, L1, D2) to (B, L1, D)
        k, v = self.to_kv(x).chunk(2, dim=-1)  # (B, L2, D1) to twice (B, L2, D)

        # Apply LN before (!) splitting the heads.
        k = self.ln_k(k)
        q = self.ln_q(q)

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
        drop: float = 0.0,
        residual_latent: bool = True,
        ln_eps: float = 1e-5,
        ln_k_q: bool = False,
    ) -> None:
        """Initialise.

        Args:
            latent_dim (int): Dimensionality of the latent features given as input.
            context_dim (int): Dimensionality of the context features also given as input.
            depth (int, optional): Number of attention layers.
            head_dim (int, optional): Attention head dimensionality. Defaults to `64`.
            num_heads (int, optional): Number of heads. Defaults to `16`
            mlp_ratio (float, optional): Rimensionality of the hidden layer divided by that of the
                input for all MLPs. Defaults to `4.0`.
            drop (float, optional): Drop-out rate. Defaults to no drop-out.
            residual_latent (bool, optional): Use residual attention w.r.t. the latent features.
                Defaults to `True`.
            ln_eps (float, optional): Epsilon in the layer normalisation layers. Defaults to
                `1e-5`.
            ln_k_q (bool, optional): Apply an extra layer norm. to the keys and queries of the first
                resampling layer. Defaults to `False`.
        """
        super().__init__()

        self.residual_latent = residual_latent
        self.layers = nn.ModuleList([])
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            latent_dim=latent_dim,
                            context_dim=context_dim,
                            head_dim=head_dim,
                            num_heads=num_heads,
                            ln_k_q=ln_k_q if i == 0 else False,
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
            # HuggingFace suggests using non-residual attention in Perceiver might work better when
            # the semantics of the query and the output are different:
            #
            #   https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/perceiver/modeling_perceiver.py#L398
            #
            latents = attn_out + latents if self.residual_latent else attn_out
            latents = ln2(ff(latents)) + latents
        return latents

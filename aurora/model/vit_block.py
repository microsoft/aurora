"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Extends TIMM transformer block with compute and  memory efficient FlashAttention

References:
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
flash attention: https://github.com/HazyResearch/flash-attention
"""

import torch
from timm.models.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale
from torch import nn


class Attention(nn.Module):
    """Self-attention module with the flash_attention switch.

    The implementation is the same as that in timm.models.vision_transformer.Attention.

    Example:

        attn = Attention(128, num_heads=2)
        x = torch.zeros((16, 16, 128))
        x2 = attn(x)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        if not use_flash_attn:
            self.attn_drop_layer = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_layer = nn.Dropout(proj_drop)

    def forward(self, x: torch.FloatTensor):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.use_flash_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop, is_causal=False
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop_layer(attn)
            # (B, num_heads, N, C//num_heads)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop_layer(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """A Transformer Encoder layer, which is the basic block of Transformer.

    Args:
        dim: the number of expected features in the inputs.
        mlp_ratio: the ratio between dim and the dim of hidden features in Mlp.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_flash_attn = use_flash_attn

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash_attn=use_flash_attn,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TransformerEncoderBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int = 16,
        mlp_ratio: float = 48 / 11,
        drop_path: float = 0.1,
        drop_rate: float = 0.1,
        use_flash_attn: bool = False,
    ):
        super().__init__()

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                    use_flash_attn=use_flash_attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # apply TransformerEncoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

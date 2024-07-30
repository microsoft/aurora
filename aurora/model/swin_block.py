"""
Copyright (c) Microsoft Corporation. Licensed under the MIT license.

--------------------------------------------------------
Swin Transformer V2
Copyright (c) 2022 Microsoft
Licensed under The MIT License [see LICENSE for details]
Written by Ze Liu

Code adapted from

  https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py

--------------------------------------------------------
"""

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple

from aurora.model.helpers import init_weights, maybe_adjust_windows
from aurora.model.lora import LoraMode, LoRARollout


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple[int, int]): window size

    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], C)
    """
    B, H, W, C = x.shape
    assert H % window_size[0] == 0, f"H ({H}) % window_size ({window_size[0]}) must be 0"
    assert W % window_size[1] == 0, f"W ({W}) % window_size ({window_size[1]}) must be 0"

    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: tuple[int, int], H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int, int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Defaults to
            `True`.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_steps=40,
        lora_mode: LoraMode = "single",
        use_lora: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.head_dim = dim // num_heads

        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_lora:
            self.lora_proj = LoRARollout(
                dim, dim, lora_r, lora_alpha, lora_dropout, lora_steps, lora_mode
            )
            self.lora_qkv = LoRARollout(
                dim, dim * 3, lora_r, lora_alpha, lora_dropout, lora_steps, lora_mode
            )
        else:
            self.lora_proj = lambda *args, **kwargs: 0  # type: ignore
            self.lora_qkv = lambda *args, **kwargs: 0  # type: ignore

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, rollout_step: int = 0
    ) -> torch.Tensor:
        """
        Runs the forward pass of the window-based multi-head self attention layer.

        Args:
            x (torch.Tensor): Input features with shape of `(nW*B, N, C)`.
            mask (torch.Tensor, optional): Attention mask of floating-points in the range
                `[-inf, 0)` with shape of `(nW, ws, ws)`, where `nW` is the number of windows,
                and `ws` is the window size (i.e. total tokens inside the window).

        Returns:
            torch.Tensor: Output of shape `(nW*B, N, C)`.
        """
        qkv = self.qkv(x) + self.lora_qkv(x, rollout_step)
        qkv = rearrange(qkv, "B N (qkv H D) -> qkv B H N D", H=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_dropout = self.attn_drop if self.training else 0.0

        if mask is not None:
            nW = mask.shape[0]
            q, k, v = map(lambda t: rearrange(t, "(B nW) H N D -> B nW H N D", nW=nW), (q, k, v))
            mask = mask.unsqueeze(1).unsqueeze(0)  # (1, nW, 1, ws, ws)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=attn_dropout)
            x = rearrange(x, "B nW H N D -> (B nW) H N D")
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout)

        x = rearrange(x, "B H N D -> B N (H D)")
        x = self.proj(x) + self.lora_proj(x, rollout_step)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


def get_two_sidded_padding(H_padding: int, W_padding: int) -> tuple[int, int, int, int]:
    """Returns the padding for the left, right, top, and bottom sides."""
    assert H_padding >= 0, f"H_padding ({H_padding}) must be >= 0"
    assert W_padding >= 0, f"W_padding ({W_padding}) must be >= 0"

    if H_padding:
        padding_top = H_padding // 2
        padding_bottom = H_padding - padding_top
    else:
        padding_top = padding_bottom = 0

    if W_padding:
        padding_left = W_padding // 2
        padding_right = W_padding - padding_left
    else:
        padding_left = padding_right = 0

    return padding_left, padding_right, padding_top, padding_bottom


def pad_2d(x: torch.Tensor, pad_size: tuple[int, int], value: float = 0.0) -> torch.Tensor:
    """Pads the input with value to the specified size."""
    # Padding is done from the last dimension. We use zero padding for the last dimension.
    return F.pad(x, (0, 0, *get_two_sidded_padding(*pad_size)), value=value)


def crop_2d(x: torch.Tensor, pad_size: tuple[int, int]) -> torch.Tensor:
    """Undoes the `pad_2d` function by cropping the padded values."""
    B, H, W, C = x.shape
    H_padding, W_padding = pad_size

    padding_left, padding_right, padding_top, padding_bottom = get_two_sidded_padding(
        H_padding, W_padding
    )
    x = x[:, padding_top : H - padding_bottom, padding_left : W - padding_right, :]
    return x


@lru_cache
def compute_shifted_window_mask(
    H: int,
    W: int,
    window_size: tuple[int, int],
    shift_size: tuple[int, int],
    device: torch.device,
    warped: bool = True,
) -> tuple[torch.tensor, torch.tensor]:
    """Computes the mask of each window for the shifted window attention.

    The algorithm splits the cyclically shifted image into blocks as depicted in the
    middle diagram of Figure 4 of the paper: https://arxiv.org/abs/2103.14030.
    These blocks are numbered as follows:

    -------------------
    | .. 0 .. | 1 | 2 |
    | .. 0 .. | 1 | 2 |
    | .. 0 .. | 1 | 2 |
    -------------------
    | .. 3 .. | 4 | 5 |
    -------------------
    | .. 6 .. | 7 | 8 |
    -------------------

    Two patches in the same window are allowed to communicate if they are in the same block.

    When the image is warped (i.e. the left and right sides are connected), we merge blocks
    1 and 2, 4 and 5, and 7 and 8. This results in the following blocks:

    -------------------
    | .. 0 .. | 2 | 2 |
    | .. 0 .. | 2 | 2 |
    | .. 0 .. | 2 | 2 |
    -------------------
    | .. 3 .. | 5 | 5 |
    -------------------
    | .. 6 .. | 8 | 8 |
    -------------------

    When the resolution is not a multiple of the window size, we pad the image to the
    nearest multiple of the window size. We assign the padded patches to a separate group with
    index nine to avoid any communication between padded and non-padded patches.
    This results in the following blocks:

    -----------------------
    | .. 0 .. | 2 | 2 | 9 |
    | .. 0 .. | 2 | 2 | 9 |
    | .. 0 .. | 2 | 2 | 9 |
    -----------------------
    | .. 3 .. | 5 | 5 | 9 |
    -----------------------
    | .. 6 .. | 8 | 8 | 9 |
    -----------------------
    | .. 9 .. | 9 | 9 | 9 |
    -----------------------

    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        window_size (tuple[int, int]): Window size.
        shift_size (tuple[int, int]): Shift size.
        warped (bool): If warped, we assume the left and right sides of the image are connected.

    Returns:
        attn_mask (torch.tensor): Attention mask for each window. Masked entries are -100 and
            non-masked entries are 0. This matrix is added to the attention matrix before softmax.
        img_mask (torch.tensor): Image mask splitting the input patches into groups.
            Used for debugging purposes.
    """
    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[1], None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )

    # Assign all patches in the same group the same cnt value.
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    if warped:
        img_mask = img_mask.masked_fill(img_mask == 1, 2)  # Merge groups 1 and 2.
        img_mask = img_mask.masked_fill(img_mask == 4, 5)  # Merge groups 4 and 5.
        img_mask = img_mask.masked_fill(img_mask == 7, 8)  # Merge groups 7 and 8.

    # Pad to multiple of window size and assign padded patches to a separate group (cnt).
    pad_size = (
        window_size[0] - H % window_size[0],
        window_size[1] - W % window_size[1],
    )
    pad_size = (pad_size[0] % window_size[0], pad_size[1] % window_size[1])
    img_mask = pad_2d(img_mask, pad_size, value=cnt)

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])
    # Two patches communicate if they are in the same group (i.e. the difference below is 0).
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    return attn_mask, img_mask


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int, int]): Window size.
        shift_size (tuple[int, int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size: tuple[int, int] = (7, 7),
        shift_size: tuple[int, int] = (0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, res: tuple[int, int], warped=True):
        H, W = res
        B, L, C = x.shape
        assert L == H * W, f"Wrong feature size: {L} vs {H}x{W}={H*W}"

        window_size, shift_size = maybe_adjust_windows(self.window_size, self.shift_size, res)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Perform cyclic shift.
        if not all(s == 0 for s in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask, _ = compute_shifted_window_mask(
                H, W, window_size, shift_size, x.device, warped=warped
            )
        else:
            shifted_x = x
            attn_mask = None

        # Pad the input to multiple of window size.
        pad_size = (
            window_size[0] - H % window_size[0],
            window_size[1] - W % window_size[1],
        )
        pad_size = (pad_size[0] % window_size[0], pad_size[1] % window_size[1])
        shifted_x = pad_2d(shifted_x, pad_size)

        # Partition the patches/tokens into windows.
        x_windows = window_partition(shifted_x, window_size)  # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)  # nW*B, ws*ws, C

        # W-MSA/SW-MSA.
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, ws*ws, C

        # Merge the windows into the original input (patch) resolution.
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], C)
        _, pad_H, pad_W, _ = shifted_x.shape
        shifted_x = window_reverse(attn_windows, window_size, pad_H, pad_W)  # B H' W' C

        # Reverse the padding after the attention computations are done.
        shifted_x = crop_2d(shifted_x, pad_size)

        # Reverse the cyclic shift.
        if not all(s == 0 for s in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.reshape(B, H * W, C)  # Cropping requires reshape() instead of view().
        x = shortcut + self.drop_path(x)

        # FFN.
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Goes from (B, H*W, C) --> (B, H/2*W/2, 2*C)

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def _merge(self, x: torch.Tensor, res: tuple[int, int]) -> torch.Tensor:
        H, W = res
        B, L, C = x.shape
        assert L == H * W, f"Wrong feature size: {L} vs {H}x{W}={H*W}"
        assert H > 1, f"Height ({H}) must be larger than 1"
        assert W > 1, f"Width ({W}) must be larger than 1"

        x = x.view(B, H, W, C)
        x = pad_2d(x, (H % 2, W % 2))  # Pad to multiple of 2.
        new_H, new_W = x.shape[1], x.shape[2]
        assert new_H % 2 == 0, f"({new_H}) % 2 != 0"
        assert new_W % 2 == 0, f"({new_W}) % 2 != 0"

        x = x.reshape(B, new_H // 2, 2, new_W // 2, 2, C)
        return rearrange(x, "B H h W w C -> B (H W) (h w C)")

    def forward(self, x: torch.Tensor, input_resolution: tuple[int, int]) -> torch.Tensor:
        """
        x: B, H*W, C
        """
        x = self._merge(x, input_resolution)
        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class PatchSplitting(nn.Module):
    r"""Patch splitting layer.

    Quadruples the number of patches by doubling in the horizontal and vertical directions.

    Changes the shape of the inputs from `(B, H*W, C)` to `(B, 2H*2W, C/2)`.

     Args:
        input_resolution (tuple[int, int]): Resolution of input features.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, f"dim ({dim}) should be divisible by 4."
        self.expansion = nn.Linear(dim // 4, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)

    def _split(self, x: torch.Tensor, res: tuple[int, int], crop: tuple[int, int]) -> torch.Tensor:
        H, W = res
        B, L, C = x.shape
        assert (
            L == H * W
        ), "Dimensionality of the input features do not line up with the given resolution."
        assert C % 4 == 0, f"Number of channels of the input ({C}) is not a multiple of 4."

        x = x.view(B, H, W, 2, 2, C // 4)
        x = rearrange(x, "B H W h w C -> B (H h) (W w) C")  # B 2H, 2W C/4
        x = crop_2d(x, crop)  # Undo padding from PatchMerging (if any).

        return x.reshape(B, -1, C // 4)  # B 2H*2W C/4

    def forward(
        self,
        x: torch.Tensor,
        input_resolution: tuple[int, int],
        crop: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        x = self._split(x, input_resolution, crop)
        x = self.expansion(x)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int, int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Defaults
            to `None`.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size: tuple[int, int],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        upsample=None,
        use_checkpoint=False,
    ):
        super().__init__()

        if downsample is not None and upsample is not None:
            raise ValueError("Cannot set both downsample and upsample")

        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        (0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2)
                    ),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch downsample layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # patch uplsample layer
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(
        self,
        x,
        input_resolution: tuple[int, int],
        crop: tuple[bool, bool] = (False, False),
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for blk in self.blocks:
            x = blk(x, input_resolution)
        if self.downsample is not None:
            x_scaled = self.downsample(x, input_resolution)
            return x_scaled, x
        if self.upsample is not None:
            x_scaled = self.upsample(x, input_resolution, crop)
            return x_scaled, x
        return x, None

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformerBackbone(nn.Module):
    """Swin Transformer

    A PyTorch implementation of "Swin Transformer: Hierarchical Vision Transformer using Shifted
    Windows":

        https://arxiv.org/pdf/2103.14030

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int| tuple(int)): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(
        self,
        embed_dim=96,
        encoder_depths=(2, 2, 6, 2),
        encoder_num_heads=(3, 6, 12, 24),
        decoder_depths=(2, 6, 2, 2),
        decoder_num_heads=(24, 12, 6, 3),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.window_size = to_2tuple(window_size)
        self.num_encoder_layers = len(encoder_depths)
        self.num_decoder_layers = len(decoder_depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        assert sum(encoder_depths) == sum(decoder_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_depths))]

        # build encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_encoder_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=encoder_depths[i_layer],
                num_heads=encoder_num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(encoder_depths[:i_layer]) : sum(encoder_depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=(PatchMerging if (i_layer < self.num_encoder_layers - 1) else None),
            )
            self.encoder_layers.append(layer)

        # build decoder layers
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_decoder_layers):
            exponent = self.num_decoder_layers - i_layer - 1
            layer = BasicLayer(
                dim=int(embed_dim * 2**exponent),
                depth=decoder_depths[i_layer],
                num_heads=decoder_num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(decoder_depths[:i_layer]) : sum(decoder_depths[: i_layer + 1])],
                norm_layer=norm_layer,
                upsample=(PatchSplitting if (i_layer < self.num_decoder_layers - 1) else None),
            )
            self.decoder_layers.append(layer)

        self.apply(init_weights)
        for bly in self.encoder_layers:
            bly._init_respostnorm()
        for bly in self.decoder_layers:
            bly._init_respostnorm()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", "relative_position_bias_table"}

    def get_encoder_specs(
        self, patch_res: tuple[int, int]
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Gets the input resolution and output padding of each encoder layer."""
        all_res = [patch_res]
        padded_outs = []
        for _ in range(1, self.num_encoder_layers):
            H, W = all_res[-1]
            pad_H = H % 2
            pad_W = W % 2
            next_H = (H + 1) // 2 if pad_H else H // 2
            next_W = (W + 1) // 2 if pad_W else W // 2
            padded_outs.append((pad_H, pad_W))
            all_res.append((next_H, next_W))

        padded_outs.append((0, 0))
        return all_res, padded_outs

    def forward(self, x, patch_res: tuple[int, int]) -> torch.Tensor:
        assert x.shape[1] == patch_res[0] * patch_res[1], "Input shape does not match patch size"
        all_enc_res, padded_outs = self.get_encoder_specs(patch_res)

        skip = None
        for i, layer in enumerate(self.encoder_layers):
            x, x_unscaled = layer(x, all_enc_res[i])
            if i == 0:
                skip = x_unscaled
        for i, layer in enumerate(self.decoder_layers[:-1]):
            index = self.num_decoder_layers - i - 1
            x, _ = layer(x, all_enc_res[index], padded_outs[index - 1])

        x, _ = self.decoder_layers[-1](x, all_enc_res[0])
        x = torch.cat([x, skip], dim=-1)
        return x

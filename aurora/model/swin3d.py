"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Code adapted from

    https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py

"""

import itertools
from datetime import timedelta
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_3tuple

from aurora.model.film import AdaptiveLayerNorm
from aurora.model.fourier import lead_time_expansion
from aurora.model.lora import LoRAMode, LoRARollout
from aurora.model.util import init_weights, maybe_adjust_windows

__all__ = ["Swin3DTransformerBackbone"]


class MLP(nn.Module):
    """A one-hidden-layer MLP with dropout after the hidden layer and at the end."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialise.

        Args:
            in_features (int): Input dimensionality.
            hidden_features (int, optional): Hidden layer dimensionality. Defaults to the input
                dimensionality.
            out_features (int, optional): Output dimensionality. Defaults to the input
                dimensionality.
            act_layer (type, optional): Activation function to use. Will be instantiated as
                `act_layer()`. Defaults to `torch.nn.GELU`.
            drop (float, optional): Drop-out rate. Defaults to no drop-out.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA).

    It supports both shifted and non-shifted windows.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lora_r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        use_lora: bool = False,
    ) -> None:
        """Initialise.

        Args:
            dim (int): Number of input channels.
            window_size (tuple[int, int, int]): The size of the windows.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): If `True`, add a learnable bias to the query, key, dn value.
                Defaults to `True`.
            qk_scale (float, optional): If set, overrides the default query-key scale of
                `1/sqrt(head_dim)`.
            attn_drop (float, optional): Drop-out rate of attention weights. Default to `0.0`.
            proj_drop (float, optional): Drop-out rate of the output. Default to `0.0`.
            lora_r (int, optional): LoRA rank. Defaults to `8`.
            lora_alpha (int, optional): LoRA alpha. Defaults to `8`.
            lora_dropout (float, optional): LoRA drop-out rate. Defaults to `0.0`.
            lora_steps (int, optional): Maximum number of LoRA roll-out steps. Defaults to `40`.
            lora_mode (str, optional): Mode. `"single"` uses the same LoRA for all roll-out steps,
                and `"all"` uses a different LoRA for every roll-out step. Defaults to `"single"`.
            use_lora (bool, optional): Enable LoRA. By default, LoRA is disabled.
        """
        super().__init__()

        self.dim = dim
        self.window_size = window_size  # (Wc, Wh, Ww)
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim ({dim}) should be divisible by num_heads ({num_heads})."
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
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rollout_step: int = 0,
    ) -> torch.Tensor:
        """Run the forward pass of the window-based multi-head self-attention layer.

        Args:
            x (torch.Tensor): Input features with shape of `(nW*B, N, C)`.
            mask (torch.Tensor, optional): Attention mask of floating points in the range
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
    """Returns the padding for the left, right, top, and bottom sides, in exactly that order."""
    assert H_padding >= 0, f"H_padding ({H_padding}) must be >= 0."
    assert W_padding >= 0, f"W_padding ({W_padding}) must be >= 0."

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


def window_partition_3d(x: torch.Tensor, ws: tuple[int, int, int]) -> torch.Tensor:
    """Partition into windows.

    Args:
        x: (torch.Tensor): Input tensor of shape `(B, C, H, W, D)`.
        ws: (tuple[int, int, int]): A 3D window size `(Wc, Wh, Ww)`.

    Returns:
        torch.Tensor: Partitioning of shape `(num_windows*B, Wc, Wh, Ww, D)`.
    """
    B, C, H, W, D = x.shape
    assert C % ws[0] == 0, f"C ({C}) % window_size ({ws[0]}) must be 0."
    assert H % ws[1] == 0, f"H ({H}) % window_size ({ws[1]}) must be 0."
    assert W % ws[2] == 0, f"W ({W}) % window_size ({ws[2]}) must be 0."

    x = x.view(B, C // ws[0], ws[0], H // ws[1], ws[1], W // ws[2], ws[2], D)
    windows = rearrange(x, "B C1 Wc H1 Wh W1 Ww D -> (B C1 H1 W1) Wc Wh Ww D")
    return windows


def window_reverse_3d(windows: torch.Tensor, ws: tuple[int, int, int], C: int, H: int, W: int):
    """Unpartition a partitioning.

    Args:
        windows (torch.Tensor): Partitioning of shape `(num_windows*B, Wc, Wh, Ww, D)`.
        ws (tuple[int, int, int]): The 3D window size.
        C (int): Number of levels.
        H (int): Height of image.
        W (int): Width of image.

    Returns:
        torch.Tensor: Unpartitioned input of shape `(B, C, H, W, D)`.
    """
    assert C % ws[0] == 0, f"D ({C}) % window_size ({ws[0]}) must be 0."
    assert H % ws[1] == 0, f"H ({H}) % window_size ({ws[1]}) must be 0."
    assert W % ws[2] == 0, f"W ({W}) % window_size ({ws[2]}) must be 0."

    C1, H1, W1 = C // ws[0], H // ws[1], W // ws[2]
    B = int(windows.shape[0] / (C1 * H1 * W1))
    x = rearrange(
        windows,
        "(B C1 H1 W1) Wc Wh Ww D -> B (C1 Wc) (H1 Wh) (W1 Ww) D",
        B=B,
        C1=C1,
        H1=H1,
        W1=W1,
        Wc=ws[0],
        Wh=ws[1],
        Ww=ws[2],  # fmt: skip
    )
    return x


def get_three_sidded_padding(
    C_padding: int,
    H_padding: int,
    W_padding: int,
) -> tuple[int, int, int, int, int, int]:
    """Returns the padding for the left, right, top, bottom, front, and back sides, in exactly that
    order."""
    assert C_padding >= 0, f"C_padding ({C_padding}) must be >= 0"

    if C_padding:
        pad_front = C_padding // 2
        pad_back = C_padding - pad_front
    else:
        pad_front = pad_back = 0

    return (
        *get_two_sidded_padding(H_padding, W_padding),
        pad_front,
        pad_back,
    )


def pad_3d(x: torch.Tensor, pad_size: tuple[int, int, int], value: float = 0.0) -> torch.Tensor:
    """Pads the input with value to the specified size."""
    # Padding is done from the last dimension. We use zero padding for the last dimension.
    return F.pad(x, (0, 0, *get_three_sidded_padding(*pad_size)), value=value)


def crop_3d(x: torch.Tensor, pad_size: tuple[int, int, int]) -> torch.Tensor:
    """Undoes the `pad_3d` function by cropping the padded values."""
    B, C, H, W, D = x.shape
    Cp, Hp, Wp = pad_size

    pleft, pright, ptop, pbottom, pfront, pback = get_three_sidded_padding(Cp, Hp, Wp)
    x = x[:, pfront : C - pback, ptop : H - pbottom, pleft : W - pright, :]
    return x


def get_3d_merge_groups() -> list[tuple[int, int]]:
    """Returns the groups to be merged for the 3D case to obtain left-right connectivity."""
    merge_groups_2d = [(1, 2), (4, 5), (7, 8)]
    merge_groups_3d = []
    for i_c_slice in range(3):
        for grp1_2d, grp2_2d in merge_groups_2d:
            # The 2D merge groups show up in each of the `c_slices` with an offset of 9. 9
            # correspond to the total number of 2D merge groups. See
            # :func:`compute_3d_shifted_window_mask`.
            offset = i_c_slice * 9
            grp1_3d, grp2_3d = grp1_2d + offset, grp2_2d + offset
            merge_groups_3d.append((grp1_3d, grp2_3d))
    return merge_groups_3d


@lru_cache
def compute_3d_shifted_window_mask(
    C: int,
    H: int,
    W: int,
    ws: tuple[int, int, int],
    ss: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    warped: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the mask of each window for the shifted-window attention.

    Args:
        C (int): Number of levels.
        H (int): Height of the image.
        W (int): Width of the image.
        ws (tuple[int, int, int]): Window sizes of the form `(Wc, Wh, Ww)`.
        ss (tuple[int, int, int]): Shift sizes of the form `(Sc, Sh, Sw)`
        dtype (torch.dtype, optional): Data type of the mask. Defaults to `torch.bfloat16`.
        warped (bool): If `True`,assume that the left and right sides of the image are connected.
            Defaults to `True`.

    Returns:
        torch.Tensor: Attention mask for each window. Masked entries are -100 and non-masked
            entries are 0. This matrix is added to the attention matrix before softmax.
        torch.Tensor: Image mask splitting the input patches into groups. Used for debugging
            purposes.
    """
    img_mask = torch.zeros((1, C, H, W, 1), device=device, dtype=dtype)
    c_slices = (slice(0, -ws[0]), slice(-ws[0], -ss[0]), slice(-ss[0], None))
    h_slices = (slice(0, -ws[1]), slice(-ws[1], -ss[1]), slice(-ss[1], None))
    w_slices = (slice(0, -ws[2]), slice(-ws[2], -ss[2]), slice(-ss[2], None))

    # Assign each patch to a communication group. The iteration order here must be consistent with
    # the indices that :func:`get_3d_merge_groups` computes.
    cnt = 0
    for c, h, w in itertools.product(c_slices, h_slices, w_slices):
        img_mask[:, c, h, w, :] = cnt
        cnt += 1

    if warped:
        for grp1, grp2 in get_3d_merge_groups():
            img_mask = img_mask.masked_fill(img_mask == grp1, grp2)

    # Pad to multiple of window size and assign padded patches to a separate group (`cnt` is still
    # unused).
    pad_size = (ws[0] - C % ws[0], ws[1] - H % ws[1], ws[2] - W % ws[2])
    pad_size = (pad_size[0] % ws[0], pad_size[1] % ws[1], pad_size[2] % ws[2])
    img_mask = pad_3d(img_mask, pad_size, value=cnt)

    mask_windows = window_partition_3d(img_mask, ws)  # (nW*B, ws[0], ws[1], ws[2], 1)
    mask_windows = mask_windows.view(-1, ws[0] * ws[1] * ws[2])  # (nW*B, ws[0] * ws[1] * ws[2])
    # Two patches communicate if they are in the same group (i.e. the difference below is 0).
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    return attn_mask, img_mask


class Swin3DTransformerBlock(nn.Module):
    """3D Swin Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_dim: int,
        window_size: tuple[int, int, int] = (2, 7, 7),
        shift_size: tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        scale_bias: float = 0.0,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        use_lora: bool = False,
    ) -> None:
        """Initialise.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int, int]): Input resolution.
            num_heads (int): Number of attention heads.
            time_dim (int): Dimension of the lead time embedding.
            window_size (tuple[int, int, int]): Window size. Defaults to `(2, 7, 7)`.
            shift_size (tuple[int, int, int]): Shift size for SW-MSA. Defaults to `(0, 0, 0)`.
            mlp_ratio (float): Hidden layer dimensionality divided by that of the input for all
                MLPs. Defaults to `4.0`.
            qkv_bias (bool, optional): If `True,` add a learnable bias to each query, key, and
                value. Defaults to `True`.
            drop (float, optional): Drop-out rate. Defaults to `0.0`.
            attn_drop (float, optional): Attention drop-out rate. Defaults to `0.0`.
            drop_path (float, optional): Stochastic depth rate. Defaults to `0.0`
            act_layer (type, optional): Activation function to use. Will be instantiated as
                `act_layer()`. Defaults to `torch.nn.GELU`.
            scale_bias (float, optional): Scale bias for
                :class:`aurora.model.film.AdaptiveLayerNorm`. Defaults to `0`.
            lora_steps (int, optional): Maximum number of LoRA roll-out steps. Defaults to `40`.
            lora_mode (str, optional): Mode. `"single"` uses the same LoRA for all roll-out steps,
                and `"all"` uses a different LoRA for every roll-out step. Defaults to `"single"`.
            use_lora (bool): Enable LoRA. By default, LoRA is disabled.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = AdaptiveLayerNorm(dim, time_dim, scale_bias=scale_bias)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            lora_steps=lora_steps,
            use_lora=use_lora,
            lora_mode=lora_mode,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = AdaptiveLayerNorm(dim, time_dim, scale_bias=scale_bias)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        res: tuple[int, int, int],
        rollout_step: int,
        warped: bool = True,
    ) -> torch.Tensor:
        """Run the block.

        Args:
            x (torch.Tensor): Input tokens of shape `(B, L, D)`.
            c (torch.Tensor): Conditioning context of shape `(B, D)`.
            res (tuple[int, int, int]): Resolution of the input `x`.
            rollout_step (int): Roll-out step.
            warped (bool, optional): Connect the left and right sides. Defaults to `True`.

        Returns:
            torch.Tensor: Output tokens.
        """
        C, H, W = res
        B, L, D = x.shape
        assert L == C * H * W, f"Wrong feature size: {L} vs {C}x{H}x{W}={C*H*W}"

        # If the window size is larger than the input resolution, we do not partition windows.
        ws, ss = maybe_adjust_windows(self.window_size, self.shift_size, res)

        shortcut = x
        x = x.view(B, C, H, W, D)

        # Perform cyclic shift.
        if not all(s == 0 for s in ss):
            shifted_x = torch.roll(x, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
            attn_mask, _ = compute_3d_shifted_window_mask(
                C, H, W, ws, ss, x.device, x.dtype, warped=warped
            )
        else:
            shifted_x = x
            attn_mask = None

        # Pad the input to multiple of window size.
        pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
        shifted_x = pad_3d(shifted_x, pad_size)

        # Partition the patches/tokens into windows.
        x_windows = window_partition_3d(shifted_x, ws)  # (nW*B, ws, ws, D)
        x_windows = x_windows.view(-1, ws[0] * ws[1] * ws[2], D)  # (nW*B, ws*ws, D)

        # W-MSA/SW-MSA. Has shape (nW*B, ws*ws, D).
        attn_windows = self.attn(x_windows, mask=attn_mask, rollout_step=rollout_step)

        # Merge the windows into the original input (patch) resolution.
        attn_windows = attn_windows.view(-1, ws[0], ws[1], ws[2], D)  # (nW*B, Wc, Wh, Ww, D)
        _, pad_C, pad_H, pad_W, _ = shifted_x.shape
        shifted_x = window_reverse_3d(attn_windows, ws, pad_C, pad_H, pad_W)  # (B C' H' W' D)

        # Reverse the padding after the attention computations are done.
        shifted_x = crop_3d(shifted_x, pad_size)

        # Reverse the cyclic shift.
        if not all(s == 0 for s in ss):
            x = torch.roll(shifted_x, shifts=(ss[0], ss[1], ss[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        x = x.reshape(B, C * H * W, D)

        x = shortcut + self.drop_path(self.norm1(x, c))
        x = x + self.drop_path(self.norm2(self.mlp(x), c))
        return x


class PatchMerging3D(nn.Module):
    """Patch merging layer."""

    def __init__(self, dim: int) -> None:
        """Initialise.

        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def _merge(self, x: torch.Tensor, res: tuple[int, int, int]) -> torch.Tensor:
        C, H, W = res
        B, L, D = x.shape
        assert L == C * H * W, f"Wrong feature size: {L} vs {C}*{H}*{W}={C*H*W}."
        assert H > 1, f"Height ({H}) must be larger than 1."
        assert W > 1, f"Width ({W}) must be larger than 1."

        x = x.view(B, C, H, W, D)
        x = pad_3d(x, (0, H % 2, W % 2))  # Pad to multiple of 2.
        new_H, new_W = x.shape[2], x.shape[3]
        assert x.shape[2] % 2 == 0, f"({new_H}) % 2 != 0."
        assert x.shape[3] % 2 == 0, f"({new_W}) % 2 != 0."

        x = x.reshape(B, C, new_H // 2, 2, new_W // 2, 2, D)
        return rearrange(x, "B C H h W w D -> B (C H W) (h w D)")

    def forward(self, x: torch.Tensor, input_resolution: tuple[int, int, int]) -> torch.Tensor:
        """Perform the path merging operation.

        Args:
            x (torch.Tensor): Input tokens of shape `(B, C*H*W, D)`.
            input_resolution (tuple[int, int, int]): Resolution of `x` of the form `(C, H, W)`.

        Returns:
            torch.Tensor: Merged tokens of shape `(B, C*H/2*W/2, 2*D)`.
        """
        x = self._merge(x, input_resolution)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchSplitting3D(nn.Module):
    """Patch splitting layer."""

    def __init__(self, dim: int) -> None:
        """Initialise.

        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, f"dim ({dim}) should be divisible by 2."
        self.lin1 = nn.Linear(dim, dim * 2, bias=False)
        self.lin2 = nn.Linear(dim // 2, dim // 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def _split(
        self,
        x: torch.Tensor,
        res: tuple[int, int, int],
        crop: tuple[int, int, int],
    ) -> torch.Tensor:
        C, H, W = res
        B, L, D = x.shape
        assert L == C * H * W, f"Wrong number of tokens: {L} != {C}*{H}*{W}={C*H*W}."
        assert D % 4 == 0, f"Number of input features ({D}) is not a multiple of 4."

        x = x.view(B, C, H, W, 2, 2, D // 4)
        x = rearrange(x, "B C H W h w D -> B C (H h) (W w) D")  # (B, C, 2*H, 2*W, D/4)
        x = crop_3d(x, crop)  # Undo padding from `PatchMerging` (if any).
        return x.reshape(B, -1, D // 4)  # (B, C*2H*2W, D/4)

    def forward(
        self,
        x: torch.Tensor,
        input_resolution: tuple[int, int, int],
        crop: tuple[int, int, int] = (0, 0, 0),
    ) -> torch.Tensor:
        """Perform the patch splitting.

        Quadruples the number of patches by doubling in the `H` and `W` dimensions.

        Args:
            x (torch.Tensor): Input tokens of shape `(B, C*H*W, D)`.
            input_resolution (tuple[int, int, int]): Resolution of `x` of the form `(C, H, W)`.
            crop (tuple[int, int, int], optional): Cropping for every dimension. Defaults to
                no cropping.

        Returns:
            torch.Tensor: Splitted tokens of shape `(B, C*(2*H)*(2*W), D/2)`.
        """
        x = self.lin1(x)  # (B, C*H*W, D*2)
        x = self._split(x, input_resolution, crop)
        x = self.norm(x)
        x = self.lin2(x)  # (B, C*(2*H)*(2*W), D/2)
        return x


class BasicLayer3D(nn.Module):
    """A basic 3D Swin Transformer layer for one stage."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        ws: tuple[int, int, int],
        time_dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        downsample: type[PatchMerging3D] | None = None,
        upsample: type[PatchSplitting3D] | None = None,
        scale_bias: float = 0.0,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        use_lora: bool = False,
    ) -> None:
        """Initialise.

        Args:
            dim (int): Number of input channels.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            ws (tuple[int, int, int]): Window size.
            time_dim (int): Dimension of the lead time embedding.
            mlp_ratio (float): Hidden layer dimensionality divided by that of the input for all
                MLPs. Defaults to `4.0`.
            qkv_bias (bool): If `True`, add a learnable bias to the query, key, and value. Defaults
                to `True`.
            drop (float): Drop-out rate. Defaults to `0.0`.
            attn_drop (float): Attention drop-out rate. Defaults to `0.0`.
            drop_path (float): Stochastic depth rate. Defaults to `0.0`.
            downsample (PatchMerging3D, optional): Downsampling layer. Defaults to no downsampling.
            upsample (PatchSplitting3D, optional): Upsampling layer. Defaults to no upsampling.
            scale_bias (float, optional): Scale bias for
                :class:`aurora.model.film.AdaptiveLayerNorm`. Default: 0
            lora_steps (int, optional): Maximum number of LoRA roll-out steps. Defaults to `40`.
            lora_mode (str, optional): Mode. `"single"` uses the same LoRA for all roll-out steps,
                and `"all"` uses a different LoRA for every roll-out step. Defaults to `"single"`.
            use_lora (bool): Enable LoRA. By default, LoRA is disabled.
        """
        super().__init__()

        if downsample is not None and upsample is not None:
            raise ValueError("Cannot set both `downsample` and `upsample`.")

        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                Swin3DTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=ws,
                    shift_size=(
                        (0, 0, 0) if (i % 2 == 0) else (ws[0] // 2, ws[1] // 2, ws[2] // 2)
                    ),
                    time_dim=time_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    scale_bias=scale_bias,
                    use_lora=use_lora,
                    lora_steps=lora_steps,
                    lora_mode=lora_mode,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample: PatchMerging3D | None = downsample(dim=dim)
        else:
            self.downsample = None

        if upsample is not None:
            self.upsample: PatchSplitting3D | None = upsample(dim=dim)
        else:
            self.upsample = None

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        res: tuple[int, int, int],
        crop: tuple[int, int, int] = (0, 0, 0),
        rollout_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the basic layer.

        Args:
            x (torch.Tensor): Input tokens of shape `(B, L, D)`.
            c (torch.Tensor): Conditioning context of shape `(B, D)`.
            res (tuple[int, int, int]): Resolution of the input `x`.
            crop (tuple[int, int, int]): Cropping for every dimension.
            rollout_step (int): Roll-out step.

        Returns:
            torch.Tensor: Output tokens.
        """
        for blk in self.blocks:
            x = blk(x, c, res, rollout_step)
        if self.downsample is not None:
            x_scaled = self.downsample(x, res)
            return x_scaled, x
        if self.upsample is not None:
            x_scaled = self.upsample(x, res, crop)
            return x_scaled, x
        return x, None

    def init_respostnorm(self):
        """Initialise the post-normalisation layers in the residual connection of the windowed
        attention mechanism."""
        for blk in self.blocks:
            blk.norm1.init_weights()
            blk.norm2.init_weights()


class Basic3DEncoderLayer(BasicLayer3D):
    """A basic 3D Swin Transformer encoder layer. Used for FSDP, which requires a subclass."""


class Basic3DDecoderLayer(BasicLayer3D):
    """A basic 3D Swin Transformer decoder layer. Used for FSDP, which requires a subclass."""


class Swin3DTransformerBackbone(nn.Module):
    """Swin 3D Transformer backbone."""

    def __init__(
        self,
        embed_dim: int = 96,
        encoder_depths: tuple[int, ...] = (2, 2, 6, 2),
        encoder_num_heads: tuple[int, ...] = (3, 6, 12, 24),
        decoder_depths: tuple[int, ...] = (2, 6, 2, 2),
        decoder_num_heads: tuple[int, ...] = (24, 12, 6, 3),
        window_size: int | tuple[int, int, int] = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        use_lora: bool = False,
    ) -> None:
        """
        Args:
            embed_dim (int): Patch embedding dimension. Default to `96`.
            encoder_depths (tuple[int, ...]): Number of blocks in each encoder layer. Defaults to
                `(2, 2, 6, 2)`.
            encoder_num_heads (tuple[int, ...]): Number of attention heads in each encoder layer.
                Default to `(3, 6, 12, 24)`.
            decoder_depths (tuple[int, ...]): Number of blocks in each decoder layer. Defaults to
                `(2, 6, 2, 2)`.
            decoder_num_heads (tuple[int, ...]): Number of attention heads in each decoder layer.
                Defaults to `(24, 12, 6, 3)`.
            window_size (int | tuple[int, int, int]): Window size. Defaults to `7`.
            mlp_ratio (float): Hidden layer dimensionality divided by that of the input for all
                MLPs. Defaults to `4.0`.
            qkv_bias (bool): If `True`, add a learnable bias to the query, key, and value. Defaults
                to `True`.
            drop_rate (float): Drop-out rate. Defaults to `0.0`.
            attn_drop_rate (float): Attention drop-out rate. Defaults to `0.1`.
            drop_path_rate (float): Stochastic depth rate. Defaults to `0.1`.
            lora_steps (int, optional): Maximum number of LoRA roll-out steps. Defaults to `40`.
            lora_mode (str, optional): Mode. `"single"` uses the same LoRA for all roll-out steps,
                and `"all"` uses a different LoRA for every roll-out step. Defaults to `"single"`.
            use_lora (bool): Enable LoRA. By default, LoRA is disabled.
        """
        super().__init__()

        self.window_size = to_3tuple(window_size)
        self.num_encoder_layers = len(encoder_depths)
        self.num_decoder_layers = len(decoder_depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        assert sum(encoder_depths) == sum(decoder_depths)
        dpr: list[float] = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_depths))
        ]

        # Build encoder layers.
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_encoder_layers):
            layer = Basic3DEncoderLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=encoder_depths[i_layer],
                num_heads=encoder_num_heads[i_layer],
                ws=self.window_size,
                mlp_ratio=self.mlp_ratio,
                time_dim=embed_dim,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(encoder_depths[:i_layer]) : sum(encoder_depths[: i_layer + 1])],
                downsample=(PatchMerging3D if (i_layer < self.num_encoder_layers - 1) else None),
                use_lora=use_lora,
                lora_steps=lora_steps,
                lora_mode=lora_mode,
            )
            self.encoder_layers.append(layer)

        # Build decoder layers.
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_decoder_layers):
            exponent = self.num_decoder_layers - i_layer - 1
            layer = Basic3DDecoderLayer(
                dim=int(embed_dim * 2**exponent),
                depth=decoder_depths[i_layer],
                num_heads=decoder_num_heads[i_layer],
                ws=self.window_size,
                mlp_ratio=self.mlp_ratio,
                time_dim=embed_dim,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(decoder_depths[:i_layer]) : sum(decoder_depths[: i_layer + 1])],
                upsample=(PatchSplitting3D if (i_layer < self.num_decoder_layers - 1) else None),
                use_lora=use_lora,
                lora_steps=lora_steps,
                lora_mode=lora_mode,
            )
            self.decoder_layers.append(layer)

        self.apply(init_weights)

        # This must overwrite the initialisation of `AdaptiveLayerNorm` by
        # `self.apply(init_weights)` above, so should be called afterwards.
        for bly in self.encoder_layers:
            bly.init_respostnorm()
        for bly in self.decoder_layers:
            bly.init_respostnorm()

    def get_encoder_specs(
        self, patch_res: tuple[int, int, int]
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        """Gets the input resolution and output padding of each encoder layer."""
        all_res = [patch_res]
        padded_outs = []
        for _ in range(1, self.num_encoder_layers):
            C, H, W = all_res[-1]
            pad_H, pad_W = H % 2, W % 2
            # The C dimension is never halved because it's tiny compared to H and W.
            padded_outs.append((0, pad_H, pad_W))
            all_res.append((C, (H + pad_H) // 2, (W + pad_W) // 2))

        padded_outs.append((0, 0, 0))
        return all_res, padded_outs

    def forward(
        self,
        x: torch.Tensor,
        lead_time: timedelta,
        rollout_step: int,
        patch_res: tuple[int, int, int],
    ) -> torch.Tensor:
        """Run the backbone.

        Args:
            x (torch.Tensor): Input tokens of shape `(B, L, D)`.
            lead_time (datetime.timedelta): Lead time.
            rollout_step (int): Roll-out step.
            patch_res (tuple[int, int, int]): Patch resolution of the form `(C, H, W)`.

        Returns:
            torch.Tensor: Output tokens of shape `(B, L, D)`.
        """
        _msg = "Input shape does not match patch size."
        assert x.shape[1] == patch_res[0] * patch_res[1] * patch_res[2], _msg

        # It's costly to pad across the level dimension, so we should not even though our model
        # supports it.
        _msg = f"Patch height ({patch_res[0]}) must be divisible by ws[0] ({self.window_size[0]})"
        assert patch_res[0] % self.window_size[0] == 0, _msg

        all_enc_res, padded_outs = self.get_encoder_specs(patch_res)

        lead_hours = lead_time / timedelta(hours=1)
        lead_times = lead_hours * torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
        c = self.time_mlp(lead_time_expansion(lead_times, self.embed_dim).to(dtype=x.dtype))

        skips = []
        for i, layer in enumerate(self.encoder_layers):
            x, x_unscaled = layer(x, c, all_enc_res[i], rollout_step=rollout_step)
            skips.append(x_unscaled)
        for i, layer in enumerate(self.decoder_layers):
            index = self.num_decoder_layers - i - 1
            x, _ = layer(
                x,
                c,
                all_enc_res[index],
                padded_outs[index - 1],
                rollout_step=rollout_step,
            )

            if 0 < i < self.num_decoder_layers - 1:
                # For the intermediate stages, we use additive skip connections.
                x = x + skips[index - 1]
            elif i == self.num_decoder_layers - 1:
                # For the last stage, we perform concatentation like in Pangu.
                x = torch.cat([x, skips[0]], dim=-1)
        return x

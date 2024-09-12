"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import TypeVar

import torch
from einops import rearrange
from timm.models.vision_transformer import trunc_normal_
from torch import nn

__all__ = [
    "unpatchify",
    "check_lat_lon_dtype",
    "maybe_adjust_windows",
    "init_weights",
]


def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
    """Unpatchify hidden representation.

    Args:
        x (torch.Tensor): Patchified input of shape `(B, L, C, V * P^2)` where `P` is the
            patch size.
        V (int): Number of variables.
        H (int): Number of latitudes.
        W (int): Number of longitudes.

    Returns:
        torch.Tensor: Unpatchified representation of shape `(B, V, C, H, W)`.
    """
    assert x.dim() == 4, f"Expected 4D tensor, but got {x.dim()}D."
    B, C = x.size(0), x.size(2)
    H = H // P
    W = W // P
    assert x.size(1) == H * W
    assert x.size(-1) == V * P**2

    x = x.reshape(shape=(B, H, W, C, P, P, V))
    x = rearrange(x, "B H W C P1 P2 V -> B V C H P1 W P2")
    x = x.reshape(shape=(B, V, C, H * P, W * P))
    return x


def check_lat_lon_dtype(lat: torch.Tensor, lon: torch.Tensor) -> None:
    """Assert that `lat` and `lon` are at least `float32`s."""
    assert lat.dtype in [torch.float32, torch.float64], f"Latitude num. unstable: {lat.dtype}."
    assert lon.dtype in [torch.float32, torch.float64], f"Longitude num. unstable: {lon.dtype}."


T = TypeVar("T", tuple[int, int], tuple[int, int, int])


def maybe_adjust_windows(window_size: T, shift_size: T, res: T) -> tuple[T, T]:
    """Adjust the window size and shift size if the input resolution is smaller than the window
    size."""
    err_msg = f"Expected same length, found {len(window_size)}, {len(shift_size)} and {len(res)}."
    assert len(window_size) == len(shift_size) == len(res), err_msg

    mut_shift_size, mut_window_size = list(shift_size), list(window_size)
    for i in range(len(res)):
        if res[i] <= window_size[i]:
            mut_shift_size[i] = 0
            mut_window_size[i] = res[i]

    new_window_size: T = tuple(mut_window_size)  # type: ignore[assignment]
    new_shift_size: T = tuple(mut_shift_size)  # type: ignore[assignment]

    assert min(new_window_size) > 0, f"Window size must be positive. Found {new_window_size}."
    assert min(new_shift_size) >= 0, f"Shift size must be non-negative. Found {new_shift_size}."

    return new_window_size, new_shift_size


def init_weights(m: nn.Module):
    """Initialise weights of a module with a truncated normal distribution.

    `nn.LayerNorm` is initialised with a `weight` of 1 and a `bias` of 0.

    Args:
        m (torch.nn.Module): Module.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)

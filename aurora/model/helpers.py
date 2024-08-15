"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import TypeVar

import torch
from einops import rearrange
from timm.models.vision_transformer import trunc_normal_
from torch import nn


def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
    """Unpatchify hidden representation.

    Args:
        x (torch.Tensor): Patchified input of shape `(B, L, C, D = V * P**2)` where `P` is the
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


def create_var_map(variables: tuple[str, ...]) -> dict[str, int]:
    """Create dictionary where the keys are variable names and values are unique IDs.

    Args:
        variables (tuple[str, ...]): Variable strings.

    Returns:
        dict[str, int]: Variable map dictionary.
    """
    return {v: i for i, v in enumerate(variables)}


def get_ids_for_var_map(
    variables: tuple, var_maps: dict, device: torch.cuda.device
) -> torch.Tensor:
    """Construct a tensor of variable IDs after retrieving those from a variable map created with
    :func:`.create_var_map`.

    Args:
        variables (tuples[str, ...]): Variables to retrieve the IDs for.
        var_maps (dict[str, int]): Variable map constructed with :func:`.create_var_map`.
        device (torch.cuda.device): Device.

    Returns:
        torch.Tensor: Tensor of variable IDs found in `var_map`.
    """
    return torch.tensor([var_maps[v] for v in variables], device=device)


def check_lat_lon_dtype(lat: torch.Tensor, lon: torch.Tensor) -> None:
    """Assert that `lat` and `lon` are at least `float32`s."""
    assert lat.dtype in [
        torch.float32,
        torch.float64,
    ], f"Latitude numerically unstable. Found type: {lat.dtype}."
    assert lon.dtype in [
        torch.float32,
        torch.float64,
    ], f"Longitude numerically unstable. Found type: {lon.dtype}."


T = TypeVar("T", tuple[int, int], tuple[int, int, int])


def maybe_adjust_windows(window_size: T, shift_size: T, res: T) -> tuple[T, T]:
    """Adjust the window size and shift size if the input res is smaller than the window size."""
    err_msg = f"Expected same length, found {len(window_size)}, {len(shift_size)} and {len(res)}."
    assert len(window_size) == len(shift_size) == len(res), err_msg

    new_shift_size, new_window_size = list(shift_size), list(window_size)
    for i in range(len(res)):
        if res[i] <= window_size[i]:
            new_shift_size[i] = 0
            new_window_size[i] = res[i]

    ws: T = tuple(new_window_size)  # type: ignore[assignment]
    ss: T = tuple(new_shift_size)  # type: ignore[assignment]

    assert min(ws) > 0, f"Window size must be positive. Found {ws}."
    assert min(ss) >= 0, f"Shift size must be non-negative. Found {ss}."

    return ws, ss


def init_weights(m: nn.Module):
    """Initialize weights of a module with a truncated normal distribution.
    LayerNorm is initialised with a weight of 1 and a bias of 0.

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

"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Parts of this code are inspired by

    https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/utils/pos_embed.py
"""

import torch
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple

from aurora.model.fourier import FourierExpansion

__all__ = ["pos_scale_enc"]


def patch_root_area(
    lat_min: torch.Tensor,
    lon_min: torch.Tensor,
    lat_max: torch.Tensor,
    lon_max: torch.Tensor,
) -> torch.Tensor:
    """For a rectangular patch on a sphere, compute the square root of the area of the patch in
    units km^2. The root is taken to return units of km, and thus stay scalable between different
    resolutions.

    Args:
        lat_min (torch.Tensor): Minimum latitutes of patches.
        lon_min (torch.Tensor): Minimum longitudes of patches.
        lat_max (torch.Tensor): Maximum latitudes of patches.
        lon_max (torch.Tensor): Maximum longitudes of patches.

    Returns:
        torch.Tensor: Square root of the area.
    """
    # Calculate area of latitude-longitude grid using the following formula. Phis are latitudes
    # and thetas are longitudes.
    #
    #   area = R**2 * pi * (sin(phi_1) - sin(phi_2)) * (theta_1 - theta_2)
    #
    # Taken from
    #
    #   https://www.johndcook.com/blog/2023/02/21/sphere-grid-area/
    #
    assert (lat_max > lat_min).all(), f"lat_max - lat_min: {torch.min(lat_max - lat_min)}."
    assert (lon_max > lon_min).all(), f"lon_max - lon_min: {torch.min(lon_max - lon_min)}."
    assert (abs(lat_max) <= 90.0).all() and (abs(lat_min) <= 90.0).all()
    assert (lon_max <= 360.0).all() and (lon_min <= 360.0).all()
    assert (lon_max >= 0.0).all() and (lon_min >= 0.0).all()
    area = (
        6371**2
        * torch.pi
        * (torch.sin(torch.deg2rad(lat_max)) - torch.sin(torch.deg2rad(lat_min)))
        * (torch.deg2rad(lon_max) - torch.deg2rad(lon_min))
    )

    assert (area > 0.0).all()
    return torch.sqrt(area)


def pos_scale_enc_grid(
    encode_dim: int,
    grid: torch.Tensor,
    patch_dims: tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the position and scale encoding for a latitude-longitude grid.

    Requires batch dimensions in the input and returns a batch dimension.

    Args:
        encode_dim (int): Output encoding dimension `D`. Must be a multiple of four: splits
            across latitudes and longitudes and across sines and cosines.
        grid (torch.Tensor): Latitude-longitude grid of dimensions `(B, 2, H, W)`. `grid[:, 0]`
            should be the latitudes of `grid[:, 1]` should be the longitudes.
        patch_dims (tuple): Patch dimensions. Different x-values and y-values are supported.
        pos_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            latitudes and longitudes.
        scale_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            patch areas.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Positional encoding and scale encoding of shape
            `(B, H/patch[0] * W/patch[1], D)`.
    """
    assert encode_dim % 4 == 0
    assert grid.dim() == 4

    # Take the 2D pooled values of the mesh. This is the same as subsequent 1D pooling over the
    # x-axis and then ove the y-axis.
    grid_h = F.avg_pool2d(grid[:, 0], patch_dims)
    grid_w = F.avg_pool2d(grid[:, 1], patch_dims)

    # Compute the square root of the area of the patches specified by the latitude-longitude
    # grid.
    grid_lat_max = F.max_pool2d(grid[:, 0], patch_dims)
    grid_lat_min = -F.max_pool2d(-grid[:, 0], patch_dims)
    grid_lon_max = F.max_pool2d(grid[:, 1], patch_dims)
    grid_lon_min = -F.max_pool2d(-grid[:, 1], patch_dims)
    root_area = patch_root_area(grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max)

    # Use half of dimensions for the latitudes of the midpoints of the patches and the other
    # half for the longitudes. Before computing the encodings, flatten over the spatial dimensions.
    B = grid_h.shape[0]
    encode_h = pos_expansion(grid_h.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    encode_w = pos_expansion(grid_w.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    pos_encode = torch.cat((encode_h, encode_w), axis=-1)  # (B, L, D)

    # No need to split things up for the scale encoding.
    scale_encode = scale_expansion(root_area.reshape(B, -1), encode_dim)  # (B, L, D)

    return pos_encode, scale_encode


def lat_lon_meshgrid(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Construct a meshgrid of latitude and longitude coordinates.

    `torch.meshgrid(*tensors, indexing="xy")` gives the same behavior as calling
    `numpy.meshgrid(*arrays, indexing="ij")`::

        lat = torch.tensor([1, 2, 3])
        lon = torch.tensor([4, 5, 6])
        grid_x, grid_y = torch.meshgrid(lat, lon, indexing='xy')
        grid_x = tensor([[1, 2, 3], [1, 2, ,3], [1, 2, 3]])
        grid_y = tensor([[4, 4, 4], [5, 5, ,5], [6, 6, 6]])

    Args:
        lat (torch.Tensor): Vector of latitudes.
        lon (torch.Tensor): Vector of longitudes.

    Returns:
        torch.Tensor: Meshgrid of shape `(2, len(lat), len(lon))`.
    """
    assert lat.dim() == 1
    assert lon.dim() == 1

    grid = torch.meshgrid(lat, lon, indexing="xy")
    grid = torch.stack(grid, axis=0)
    grid = grid.permute(0, 2, 1)

    return grid


def pos_scale_enc(
    encode_dim: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    patch_dims: int | list | tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> torch.Tensor:
    """Positional encoding of latitude-longitude data.

    Does not support batch dimensions in the input and does not return batch dimensions either.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        lat (torch.Tensor): Latitudes, `H`. Can be either a vector or a matrix.
        lon (torch.Tensor): Longitudes, `W`. Can be either a vector or a matrix.
        patch_dims (Union[list, tuple]): Patch dimensions. Different x-values and y-values are
            supported.
        pos_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            latitudes and longitudes.
        scale_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            patch areas.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Positional encoding and scale encoding of shape
            `(H/patch[0] * W/patch[1], D)`.
    """
    if lat.dim() == lon.dim() == 1:
        grid = lat_lon_meshgrid(lat, lon)
    elif lat.dim() == lon.dim() == 2:
        grid = torch.stack((lat, lon), dim=0)
    else:
        raise ValueError(
            f"Latitudes and longitudes must either both be vectors or both be matrices, "
            f"but have dimensionalities {lat.dim()} and {lon.dim()} respectively."
        )

    grid = grid[None]  # Add batch dimension.

    pos_encoding, scale_encoding = pos_scale_enc_grid(
        encode_dim,
        grid,
        to_2tuple(patch_dims),
        pos_expansion=pos_expansion,
        scale_expansion=scale_expansion,
    )

    return pos_encoding.squeeze(0), scale_encoding.squeeze(0)  # Return without batch dimension.

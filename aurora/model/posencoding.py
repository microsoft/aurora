"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Parts of this code are inspired by

    https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/utils/pos_embed.py
"""

import torch
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple

from aurora.model.fourier import FourierExpansion


def get_root_area_on_sphere(
    lat_min: torch.Tensor, lon_min: torch.Tensor, lat_max: torch.Tensor, lon_max: torch.Tensor
) -> torch.Tensor:
    """Calculate the root area of rectangular grid. Latitude and longitude values are used as
    inputs. The root is taken to return units of km, and thus stay scalable between different
    resolutions.

    Args:
        lat_min (torch.Tensor): Latitude of first point.
        lon_min (torch.Tensor): Longitude of first point.
        lat_max (torch.Tensor): Latitude of second point.
        lon_max (torch.Tensor): Longitude of second point.

    Returns:
        torch.Tensor: Tensor of root area on grid.
    """
    # Calculate area of latitude (phi) - longitude (theta) grid using the formula:
    #   R**2 * pi * (sin(phi_1) - sin(phi_2)) *(theta_1 - theta_2)
    # https://www.johndcook.com/blog/2023/02/21/sphere-grid-area/
    assert (lat_max > lat_min).all(), f"lat_max - lat_min: {torch.min(lat_max - lat_min)}"
    assert (lon_max > lon_min).all(), f"lon_max - lon_min: {torch.min(lon_max - lon_min)}"
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


def get_2d_patched_lat_lon_from_grid(
    encode_dim: int,
    grid: torch.Tensor,
    patch_dims: tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates 2D patched position encoding from grid. For each patch the mean latitute and
    longitude values are calculated.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        grid (torch.Tensor): Latitude-longitude grid of dimensions `(B, 2, H, W)`
        patch_dims (tuple): Patch dimensions. Different x- and y-values are supported.
        pos_expansion (:class:`.FourierExpansion`): Fourier expansion for the latitudes and
            longitudes.
        scale_expansion (:class:`.FourierExpansion`): Fourier expansion for the patch areas.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Returns positional encoding tensor and scale tensor of
            shape `(B, H/patch[0] * W/patch[1], D)`.
    """
    # encode_dim has to be % 4 (lat-lon split, sin-cosine split)
    assert encode_dim % 4 == 0
    assert grid.dim() == 4

    # Take the 2D pooled values of the mesh - this is the same as subsequent 1D pooling over x and
    # y axis.
    grid_h = F.avg_pool2d(grid[:, 0], patch_dims)
    grid_w = F.avg_pool2d(grid[:, 1], patch_dims)

    # get min and max values for x and y coordinates to calculate the diagonal of each patch
    grid_lat_max = F.max_pool2d(grid[:, 0], patch_dims)
    grid_lat_min = -F.max_pool2d(-grid[:, 0], patch_dims)
    grid_lon_max = F.max_pool2d(grid[:, 1], patch_dims)
    grid_lon_min = -F.max_pool2d(-grid[:, 1], patch_dims)
    root_area_on_sphere = get_root_area_on_sphere(
        grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max
    )

    # use half of dimensions to encode grid_h
    # (B, H, W) -> (B, H*W)
    encode_h = pos_expansion(
        grid_h.reshape(grid_h.shape[0], -1), encode_dim // 2
    )  # (B, H*W/patch**2, D/2)
    # (B, H, W) -> (B, H*W)
    encode_w = pos_expansion(
        grid_w.reshape(grid_w.shape[0], -1), encode_dim // 2
    )  # (B, H*W/patch**2, D/2)

    # use all dimensions to encode scale
    # (B, H, W) -> (B, H*W)
    scale_encode = scale_expansion(
        root_area_on_sphere.reshape(root_area_on_sphere.shape[0], -1), encode_dim
    )  # (B, H*W/patch**2, D)

    pos_encode = torch.cat((encode_h, encode_w), axis=-1)  # (B, H*W/patch**2, D)
    return pos_encode, scale_encode


def get_lat_lon_grid(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Return meshgrid of latitude and longitude coordinates.

    `torch.meshgrid(*tensors, indexing='xy')` gives the same behavior as calling
    `numpy.meshgrid(*arrays, indexing='ij')`::

        lat = torch.tensor([1, 2, 3])
        lon = torch.tensor([4, 5, 6])
        grid_x, grid_y = torch.meshgrid(lat, lon, indexing='xy')
        grid_x = tensor([[1, 2, 3], [1, 2, ,3], [1, 2, 3]])
        grid_y = tensor([[4, 4, 4], [5, 5, ,5], [6, 6, 6]])

    Args:
        lat (torch.Tensor): 1D tensor of latitude values
        lon (torch.Tensor): 1D tensor of longitude values

    Returns:
        torch.Tensor: Meshgrid of shape `(2, lat.shape, lon.shape)`
    """
    assert lat.dim() == 1
    assert lon.dim() == 1
    grid = torch.meshgrid(lat, lon, indexing="xy")
    grid = torch.stack(grid, axis=0)
    grid = grid.permute(0, 2, 1)

    return grid


def get_2d_patched_lat_lon_encode(
    encode_dim: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    patch_dims: int | list | tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> torch.Tensor:
    """Positional encoding of latitude-longitude data.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        lat (torch.Tensor): Tensor of latitude values `H`.
        lon (torch.Tensor): Tensor of longitude values `W`.
        patch_dims (Union[list, tuple]): Patch dimensions. Different x- and y-values are supported.
        pos_expansion (:class:`.FourierExpansion`): Fourier expansion for the latitudes and
            longitudes.
        scale_expansion (:class:`.FourierExpansion`): Fourier expansion for the patch areas.

    Returns:
        torch.Tensor: Returns positional encoding tensor of shape `(H/patch[0] * W/patch[1], D)`.
    """
    if lat.dim() == lon.dim() == 1:
        grid = get_lat_lon_grid(lat, lon)
    elif lat.dim() == lon.dim() == 2:
        grid = torch.stack((lat, lon), dim=0)
    else:
        raise ValueError(
            f"Latitudes and longitudes must either both be vectors or both be matrices, "
            f"but have dimensionalities {lat.dim()} and {lon.dim()} respectively."
        )

    grid = grid[None]  # Add batch dimension.

    pos_encode, scale_encode = get_2d_patched_lat_lon_from_grid(
        encode_dim,
        grid,
        to_2tuple(patch_dims),
        pos_expansion=pos_expansion,
        scale_expansion=scale_expansion,
    )

    return pos_encode.squeeze(0), scale_encode.squeeze(0)  # Return without batch dimension.

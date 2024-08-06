"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from datetime import datetime

import torch

from aurora.normalisation import (
    normalise_atmos_var,
    normalise_surf_var,
    unnormalise_atmos_var,
    unnormalise_surf_var,
)

__all__ = ["Metadata", "Batch"]


@dataclasses.dataclass
class Metadata:
    """Metadata in a batch.

    Args:
        lat (:class:`torch.Tensor`): Latitudes.
        lon (:class:`torch.Tensor`): Longitudes.
        time (tuple[datetime, ...]): For every batch element, the time.
        atmos_levels (tuple[int |float, ...): Pressure levels for the atmospheric variables in hPa.
    """

    lat: torch.Tensor
    lon: torch.Tensor
    time: tuple[datetime, ...]
    atmos_levels: tuple[int | float, ...]


@dataclasses.dataclass
class Batch:
    """A batch of data.

    Args:
        surf_vars (dict[str, :class:`torch.Tensor`]): Surface-level variables with shape
            `(B, T, H, W)`.
        static_vars (dict[str, :class:`torch.Tensor`]): Static variables with shape
            `(B, T, H, W)`.
        atmos_vars (dict[str, :class:`torch.Tensor`]): Atmospheric variables with shape
            `(B, T, C, H, W)`.
        metadata (:class:`Metadata`): Metadata associated to this batch.
    """

    surf_vars: dict[str, torch.Tensor]
    static_vars: dict[str, torch.Tensor]
    atmos_vars: dict[str, torch.Tensor]
    metadata: Metadata

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Get the spatial shape from an arbitrary surface-level variable."""
        return next(iter(self.surf_vars.values())).shape[-2:]

    def normalise(self) -> "Batch":
        """Normalise all variables in the batch."""
        return Batch(
            surf_vars={k: normalise_surf_var(v, k) for k, v in self.surf_vars.items()},
            static_vars={k: normalise_surf_var(v, k) for k, v in self.static_vars.items()},
            atmos_vars={
                k: normalise_atmos_var(v, k, self.metadata.atmos_levels)
                for k, v in self.atmos_vars.items()
            },
            metadata=self.metadata,
        )

    def unnormalise(self) -> "Batch":
        """Unnormalise all variables in the batch."""
        return Batch(
            surf_vars={k: unnormalise_surf_var(v, k) for k, v in self.surf_vars.items()},
            static_vars={k: unnormalise_surf_var(v, k) for k, v in self.static_vars.items()},
            atmos_vars={
                k: unnormalise_atmos_var(v, k, self.metadata.atmos_levels)
                for k, v in self.atmos_vars.items()
            },
            metadata=self.metadata,
        )

    def crop(self, patch_size: int) -> "Batch":
        """Crop the variables in the batch to patch size `patch_size`."""
        h, w = self.spatial_shape
        assert h % patch_size == 0
        assert w % patch_size == 0
        return self

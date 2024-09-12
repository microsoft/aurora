"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from datetime import datetime
from functools import partial
from typing import Callable

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator as RGI

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
        atmos_levels (tuple[int | float, ...]): Pressure levels for the atmospheric variables in
            hPa.
        rollout_step (int, optional): How many roll-out steps were used to produce this prediction.
            If equal to `0`, which is the default, then this means that this is not a prediction,
            but actual data. This field is automatically populated by the model and used to use a
            separate LoRA for every roll-out step. Generally, you are safe to ignore this field.
    """

    lat: torch.Tensor
    lon: torch.Tensor
    time: tuple[datetime, ...]
    atmos_levels: tuple[int | float, ...]
    rollout_step: int = 0

    def __post_init__(self):
        if not torch.all(self.lat[1:] - self.lat[:-1] < 0):
            raise ValueError("Latitudes must be strictly decreasing.")
        if not (torch.all(self.lat <= 90) and torch.all(self.lat >= -90)):
            raise ValueError("Latitudes must be in the range [-90, 90].")
        if not torch.all(self.lon[1:] - self.lon[:-1] > 0):
            raise ValueError("Longitudes must be strictly increasing.")
        if not (torch.all(self.lon >= 0) and torch.all(self.lon < 360)):
            raise ValueError("Longitudes must be in the range [0, 360).")


@dataclasses.dataclass
class Batch:
    """A batch of data.

    Args:
        surf_vars (dict[str, :class:`torch.Tensor`]): Surface-level variables with shape
            `(b, t, h, w)`.
        static_vars (dict[str, :class:`torch.Tensor`]): Static variables with shape `(h, w)`.
        atmos_vars (dict[str, :class:`torch.Tensor`]): Atmospheric variables with shape
            `(b, t, c, h, w)`.
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

        if w % patch_size != 0:
            raise ValueError("Width of the data must be a multiple of the patch size.")

        if h % patch_size == 0:
            return self
        elif h % patch_size == 1:
            return Batch(
                surf_vars={k: v[..., :-1, :] for k, v in self.surf_vars.items()},
                static_vars={k: v[..., :-1, :] for k, v in self.static_vars.items()},
                atmos_vars={k: v[..., :-1, :] for k, v in self.atmos_vars.items()},
                metadata=Metadata(
                    lat=self.metadata.lat[:-1],
                    lon=self.metadata.lon,
                    atmos_levels=self.metadata.atmos_levels,
                    time=self.metadata.time,
                    rollout_step=self.metadata.rollout_step,
                ),
            )
        else:
            raise ValueError(
                f"There can at most be one latitude too many, "
                f"but there are {h % patch_size} too many."
            )

    def _fmap(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "Batch":
        return Batch(
            surf_vars={k: f(v) for k, v in self.surf_vars.items()},
            static_vars={k: f(v) for k, v in self.static_vars.items()},
            atmos_vars={k: f(v) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=f(self.metadata.lat),
                lon=f(self.metadata.lon),
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def to(self, device: str | torch.device) -> "Batch":
        """Move the batch to another device."""
        return self._fmap(lambda x: x.to(device))

    def type(self, t: type) -> "Batch":
        """Convert everything to type `t`."""
        return self._fmap(lambda x: x.type(t))

    def regrid(self, res: float) -> "Batch":
        """Regrid the batch to a `res` degrees resolution.

        This results in `float32` data on the CPU.

        This function is not optimised for either speed or accuracy. Use at your own risk.
        """

        shape = (round(180 / res) + 1, round(360 / res))
        lat_new = torch.from_numpy(np.linspace(90, -90, shape[0]))
        lon_new = torch.from_numpy(np.linspace(0, 360, shape[1], endpoint=False))
        interpolate_res = partial(
            interpolate,
            lat=self.metadata.lat,
            lon=self.metadata.lon,
            lat_new=lat_new,
            lon_new=lon_new,
        )

        return Batch(
            surf_vars={k: interpolate_res(v) for k, v in self.surf_vars.items()},
            static_vars={k: interpolate_res(v) for k, v in self.static_vars.items()},
            atmos_vars={k: interpolate_res(v) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=lat_new,
                lon=lon_new,
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )


def interpolate(
    v: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    lat_new: torch.Tensor,
    lon_new: torch.Tensor,
) -> torch.Tensor:
    """Interpolate a variable `v` with latitudes `lat` and longitudes `lon` to new latitudes
    `lat_new` and new longitudes `lon_new`."""
    # Perform the interpolation in double precision.
    return torch.from_numpy(
        interpolate_numpy(
            v.double().numpy(),
            lat.double().numpy(),
            lon.double().numpy(),
            lat_new.double().numpy(),
            lon_new.double().numpy(),
        )
    ).float()


def interpolate_numpy(
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_new: np.ndarray,
    lon_new: np.ndarray,
) -> np.ndarray:
    """Like :func:`.interpolate`, but for NumPy tensors."""

    # Implement periodic longitudes in `lon`.
    assert (np.diff(lon) > 0).all()
    lon = np.concatenate((lon[-1:] - 360, lon, lon[:1] + 360))

    # Merge all batch dimensions into one.
    batch_shape = v.shape[:-2]
    v = v.reshape(-1, *v.shape[-2:])

    # Loop over all batch elements.
    vs_regridded = []
    for vi in v:
        # Implement periodic longitudes in `vi`.
        vi = np.concatenate((vi[:, -1:], vi, vi[:, :1]), axis=1)

        rgi = RGI(
            (lat, lon),
            vi,
            method="linear",
            bounds_error=False,  # Allow out of bounds, for the latitudes.
            fill_value=None,  # Extrapolate latitudes if they are out of bounds.
        )
        lat_new_grid, lon_new_grid = np.meshgrid(
            lat_new,
            lon_new,
            indexing="ij",
            sparse=True,
        )
        vs_regridded.append(rgi((lat_new_grid, lon_new_grid)))

    # Recreate the batch dimensions.
    v_regridded = np.stack(vs_regridded, axis=0)
    v_regridded = v_regridded.reshape(*batch_shape, lat_new.shape[0], lon_new.shape[0])

    return v_regridded

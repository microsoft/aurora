"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fsspec
import numpy as np
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

from aurora import Batch, Metadata


def load_batch(day: datetime = datetime(2022, 5, 11), cache_path: str = "~/downloads") -> Batch:
    """Download and load an HRES T0 batch for UTC 12 on `day`.

    Automatically installs any required dependencies.

    Caches the data at `cache_path`.

    Requires no authentication.

    Args:
        day (datetime, optional): Day to download and load a batch for. Defaults to 5 Nov 2022.
        cache_path (str, optional): Path to cache the downloads at.

    Return:
        :class:`aurora.batch.Batch`: Batch.
    """
    return _load_batch(day.strftime("%Y-%m-%d"), Path(cache_path))


def _load_batch(day: str, cache_path: Path) -> Batch:
    # Install any required packages and hide the output. This can be done in a nicer way.
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "fsspec", "gcsfs", "zarr", "matplotlib"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    cache_path = cache_path.expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)

    # We will download from Google Cloud.
    url = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
    ds: xr.Dataset | None = None

    # Download the surface-level variables.
    if not (cache_path / f"{day}-surface-level.nc").exists():
        ds = ds or xr.open_zarr(fsspec.get_mapper(url), chunks=None)
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(str(cache_path / f"{day}-surface-level.nc"))

    # Download the static variables.
    if not (cache_path / "static.nc").exists():
        path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-static.pickle")
        with open(path, "rb") as f:
            static_vars = pickle.load(f)
            ds_static = xr.Dataset(
                data_vars={k: (["lattitude", "longitude"], v) for k, v in static_vars.items()},
                coords={
                    "latitude": ("latitude", np.linspace(90, -90, 721)),
                    "longitude": ("longitude", np.linspace(0, 360, 1440, endpoint=False)),
                },
            )
            ds_static.to_netcdf(str(cache_path / "static.nc"))

    # Download the atmospheric variables.
    if not (cache_path / f"{day}-atmospheric.nc").exists():
        ds = ds or xr.open_zarr(fsspec.get_mapper(url), chunks=None)
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        ds_atmos.to_netcdf(str(cache_path / f"{day}-atmospheric.nc"))

    static_vars_ds = xr.open_dataset(cache_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(cache_path / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(cache_path / f"{day}-atmospheric.nc", engine="netcdf4")

    i = 2  # Select this time index in the downloaded data.

    def _prepare(x: np.ndarray) -> torch.Tensor:
        """Prepare a variable.

        This does the following things:
        * Select time indices `i` and `i - 1`.
        * Insert an empty batch dimension with `[None]`.
        * Flip along the latitude axis to ensure that the latitudes are decreasing.
        * Copy the data, because the data must be contiguous when converting to PyTorch.
        * Convert to PyTorch.
        """
        return torch.from_numpy(x[[i - 1, i]][None][..., ::-1, :].copy())

    return Batch(
        surf_vars={
            "2t": _prepare(surf_vars_ds["2m_temperature"].values),
            "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
            "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
            "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time. They
            # don't need to be flipped along the latitude dimension, because they are from
            # ERA5.
            "z": torch.from_numpy(static_vars_ds["z"].values),
            "slt": torch.from_numpy(static_vars_ds["slt"].values),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values),
        },
        atmos_vars={
            "t": _prepare(atmos_vars_ds["temperature"].values),
            "u": _prepare(atmos_vars_ds["u_component_of_wind"].values),
            "v": _prepare(atmos_vars_ds["v_component_of_wind"].values),
            "q": _prepare(atmos_vars_ds["specific_humidity"].values),
            "z": _prepare(atmos_vars_ds["geopotential"].values),
        },
        metadata=Metadata(
            # Flip the latitudes! We need to copy because converting to PyTorch, because the
            # data must be contiguous.
            lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
        ),
    )

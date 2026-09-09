"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import SavedBatch

from aurora import Batch, Metadata


def test_interpolation(test_input_output: tuple[Batch, SavedBatch]) -> None:
    batch, _ = test_input_output

    # Regridding to the same resolution shouldn't change the data.
    batch_regridded = batch.regrid(0.45)
    batch_regridded = batch_regridded.crop(4)  # Regridding added the south pole. Remove it again.

    for k in batch.surf_vars:
        np.testing.assert_allclose(
            batch.surf_vars[k],
            batch_regridded.surf_vars[k],
            rtol=5e-6,
        )
    for k in batch.static_vars:
        np.testing.assert_allclose(
            batch.static_vars[k],
            batch_regridded.static_vars[k],
            atol=1e-7,
        )
    for k in batch.atmos_vars:
        np.testing.assert_allclose(
            batch.atmos_vars[k],
            batch_regridded.atmos_vars[k],
            rtol=5e-6,
        )

    np.testing.assert_allclose(batch.metadata.lat, batch_regridded.metadata.lat, atol=1e-10)
    np.testing.assert_allclose(batch.metadata.lon, batch_regridded.metadata.lon, atol=1e-10)


def test_save_load(test_input_output: tuple[Batch, SavedBatch], tmp_path: Path) -> None:
    batch, _ = test_input_output

    batch.to_netcdf(tmp_path / "batch.nc")
    batch_loaded = Batch.from_netcdf(tmp_path / "batch.nc")

    for k in batch.surf_vars:
        np.testing.assert_allclose(batch.surf_vars[k], batch_loaded.surf_vars[k])
    for k in batch.static_vars:
        np.testing.assert_allclose(batch.static_vars[k], batch_loaded.static_vars[k])
    for k in batch.atmos_vars:
        np.testing.assert_allclose(batch.atmos_vars[k], batch_loaded.atmos_vars[k])

    np.testing.assert_allclose(batch.metadata.lat, batch_loaded.metadata.lat)
    np.testing.assert_allclose(batch.metadata.lon, batch_loaded.metadata.lon)
    assert batch.metadata.time == batch_loaded.metadata.time
    assert batch.metadata.atmos_levels == batch_loaded.metadata.atmos_levels
    assert batch.metadata.rollout_step == batch_loaded.metadata.rollout_step


_LEVELS = (100, 250, 500, 850)


def _metadata(n_lat: int = 17, n_lon: int = 32, n_levels: int = 4) -> Metadata:
    return Metadata(
        lat=torch.linspace(90, -90, n_lat),
        lon=torch.linspace(0, 360, n_lon + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=_LEVELS[:n_levels],
    )


def _batch(metadata: Metadata | None = None) -> Batch:
    return Batch(
        surf_vars={"2t": torch.randn(1, 2, 17, 32)},
        static_vars={"lsm": torch.randn(17, 32)},
        atmos_vars={"z": torch.randn(1, 2, 4, 17, 32)},
        metadata=metadata if metadata is not None else _metadata(),
    )


@pytest.mark.parametrize("n_lat, n_lon", [(9, 32), (17, 16), (18, 32), (17, 33)])
def test_mismatching_lat_lon_and_spatial_shape(n_lat: int, n_lon: int) -> None:
    # The encoder only asserts this. That assertion carries no message and is removed under
    # `python -O`, so the mismatch reaches the caller as a bare `AssertionError` at best.
    with pytest.raises(ValueError, match="describe a grid of shape"):
        _batch(_metadata(n_lat=n_lat, n_lon=n_lon))


def test_matrix_lat_lon_is_accepted() -> None:
    metadata = _metadata()
    metadata.lat = metadata.lat[:, None].expand(17, 32)
    metadata.lon = metadata.lon[None, :].expand(17, 32)
    assert _batch(metadata).spatial_shape == (17, 32)


def test_mismatching_matrix_lat_lon() -> None:
    metadata = _metadata(n_lat=18)
    metadata.lat = metadata.lat[:, None].expand(18, 32)
    metadata.lon = metadata.lon[None, :].expand(18, 32)
    with pytest.raises(ValueError, match="latitude and longitude matrices"):
        _batch(metadata)


def test_inconsistent_spatial_shape_between_variables() -> None:
    with pytest.raises(ValueError, match="same spatial shape"):
        dataclasses.replace(_batch(), static_vars={"lsm": torch.randn(9, 32)})


def test_inconsistent_history_between_variables() -> None:
    with pytest.raises(ValueError, match="same history size"):
        dataclasses.replace(_batch(), atmos_vars={"z": torch.randn(1, 3, 4, 17, 32)})


def test_empty_variables_are_allowed() -> None:
    # Running without atmospheric variables is suggested in #176, so an empty dictionary must not
    # trip the validation.
    batch = Batch(
        surf_vars={"2t": torch.randn(1, 2, 17, 32)},
        static_vars={"lsm": torch.randn(17, 32)},
        atmos_vars={},
        metadata=_metadata(),
    )
    assert batch.atmos_vars == {}


def test_derived_batches_stay_valid(test_input_output: tuple[Batch, SavedBatch]) -> None:
    # The operations that construct new batches must not trip the validation.
    batch, _ = test_input_output
    for derived in (batch.to("cpu"), batch.crop(4), batch.regrid(0.45)):
        assert derived.spatial_shape == next(iter(derived.surf_vars.values())).shape[-2:]

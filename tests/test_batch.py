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


def test_batch_rejects_metadata_time_length_mismatch(
    test_input_output: tuple[Batch, SavedBatch],
) -> None:
    batch, _ = test_input_output
    bad_metadata = dataclasses.replace(batch.metadata, time=batch.metadata.time * 2)
    with pytest.raises(ValueError, match=r"`time`"):
        dataclasses.replace(batch, metadata=bad_metadata)


def test_batch_rejects_atmos_levels_mismatch(test_input_output: tuple[Batch, SavedBatch]) -> None:
    batch, _ = test_input_output
    bad_metadata = dataclasses.replace(
        batch.metadata, atmos_levels=batch.metadata.atmos_levels[:-1]
    )
    with pytest.raises(ValueError, match=r"`atmos_levels`"):
        dataclasses.replace(batch, metadata=bad_metadata)


def _batch(n_lat: int = 17, n_lon: int = 32, matrix: bool = False) -> Batch:
    lat = torch.linspace(90, -90, n_lat)
    lon = torch.linspace(0, 360, n_lon + 1)[:-1]
    if matrix:
        lat, lon = lat[:, None].expand(n_lat, n_lon), lon[None, :].expand(n_lat, n_lon)
    return Batch(
        surf_vars={"2t": torch.randn(1, 2, 17, 32)},
        static_vars={"lsm": torch.randn(17, 32)},
        atmos_vars={"z": torch.randn(1, 2, 4, 17, 32)},
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


@pytest.mark.parametrize("n_lat, n_lon, matrix", [(18, 32, False), (17, 33, False), (18, 32, True)])
def test_mismatching_lat_lon(n_lat: int, n_lon: int, matrix: bool) -> None:
    with pytest.raises(ValueError, match="but the variables have spatial shape"):
        _batch(n_lat, n_lon, matrix)


def test_matrix_lat_lon_is_accepted() -> None:
    _batch(matrix=True)


def test_inconsistent_spatial_shape_between_variables() -> None:
    with pytest.raises(AssertionError, match="same spatial shape"):
        dataclasses.replace(_batch(), static_vars={"lsm": torch.randn(9, 32)})


def test_inconsistent_history_between_variables() -> None:
    with pytest.raises(AssertionError, match="same history size"):
        dataclasses.replace(_batch(), atmos_vars={"z": torch.randn(1, 3, 4, 17, 32)})

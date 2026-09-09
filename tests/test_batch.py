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


def _batch(batch_size: int, n_times: int) -> Batch:
    return Batch(
        surf_vars={"2t": torch.randn(batch_size, 2, 17, 32)},
        static_vars={"lsm": torch.randn(17, 32)},
        atmos_vars={"z": torch.randn(batch_size, 2, 4, 17, 32)},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17),
            lon=torch.linspace(0, 360, 33)[:-1],
            time=tuple(datetime(2020, 6, 1, 12, 0) for _ in range(n_times)),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_matching_times_and_batch_size(batch_size: int) -> None:
    # One time per batch element is the documented contract, so this must be accepted.
    assert len(_batch(batch_size, batch_size).metadata.time) == batch_size


@pytest.mark.parametrize("batch_size, n_times", [(1, 2), (1, 100), (2, 1), (3, 2)])
def test_mismatching_times_and_batch_size(batch_size: int, n_times: int) -> None:
    # Without this check, the absolute time embedding in the encoder silently broadcasts the
    # batch dimension to `len(metadata.time)`, which either fails much later with an obscure
    # error or produces predictions of the wrong shape.
    with pytest.raises(ValueError, match="number of times in the metadata"):
        _batch(batch_size, n_times)


def test_inconsistent_batch_sizes_between_variables() -> None:
    batch = _batch(2, 2)
    with pytest.raises(ValueError, match="same batch size"):
        dataclasses.replace(batch, atmos_vars={"z": torch.randn(3, 2, 4, 17, 32)})


def test_derived_batches_stay_valid(test_input_output: tuple[Batch, SavedBatch]) -> None:
    # The operations that construct new batches must not trip the validation.
    batch, _ = test_input_output
    for derived in (batch.to("cpu"), batch.crop(4), batch.regrid(0.45)):
        assert len(derived.metadata.time) == next(iter(derived.surf_vars.values())).shape[0]

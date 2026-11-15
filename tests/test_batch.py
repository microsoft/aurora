"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import SavedBatch

from aurora import Batch, Metadata
from aurora.batch import interpolate_numpy


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
def test_regrid_irregular_grid_and_gradients(device: str) -> None:
    # The poles need extrapolation and longitude zero falls between the last and first columns.
    lat = torch.tensor([75.0, 0.0, -60.0], device=device)
    lon = torch.tensor([30.0, 103.0, 238.0, 305.0], device=device)
    field = torch.arange(12.0, device=device).reshape(4, 3).T.requires_grad_()
    masked = field.detach().clone()
    masked[0, 1] = float("nan")
    batch = Batch(
        surf_vars={"2t": field[None, None]},
        static_vars={"z": field, "masked": masked},
        atmos_vars={"t": torch.stack((field, 2 * field))[None, None]},
        metadata=Metadata(lat=lat, lon=lon, time=(datetime(2023, 1, 1),), atmos_levels=(500, 850)),
    )
    result = batch.regrid(45)
    assert result.metadata.lat.device == field.device
    assert result.metadata.lon.device == field.device
    coords = [
        c.cpu().double().numpy() for c in (lat, lon, result.metadata.lat, result.metadata.lon)
    ]
    for inputs, outputs in (
        (batch.surf_vars, result.surf_vars),
        (batch.static_vars, result.static_vars),
        (batch.atmos_vars, result.atmos_vars),
    ):
        for key, value in inputs.items():
            expected = interpolate_numpy(value.detach().cpu().double().numpy(), *coords)
            assert outputs[key].device == field.device
            assert outputs[key].dtype == torch.float32
            np.testing.assert_allclose(outputs[key].detach().cpu(), expected, rtol=1e-6, atol=1e-6)

    # Each interpolated basis field is a column of the linear interpolation operator.
    basis = np.eye(field.numel()).reshape(-1, *field.shape)
    weights = torch.arange(40.0, device=device).reshape(5, 8)
    (result.static_vars["z"] * weights).sum().backward()
    expected_grad = (
        interpolate_numpy(basis, *coords).reshape(field.numel(), -1) @ weights.cpu().numpy().ravel()
    )
    np.testing.assert_allclose(
        field.grad.cpu(), expected_grad.reshape(field.shape), rtol=1e-6, atol=1e-5
    )


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

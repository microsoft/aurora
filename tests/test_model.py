"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import os
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist

from tests.conftest import SavedBatch

from aurora import Aurora, AuroraSmallPretrained, Batch


@pytest.fixture(scope="session")
def aurora_small() -> Aurora:
    model = AuroraSmallPretrained(use_lora=True)
    model.load_checkpoint(strict=False)  # LoRA parameters not available.
    model = model.double()
    model.eval()
    return model


def test_aurora_small(aurora_small: Aurora, test_input_output: tuple[Batch, SavedBatch]) -> None:
    batch, test_output = test_input_output

    with torch.inference_mode():
        pred = aurora_small.forward(batch)

    def assert_approx_equality(v_out: np.ndarray, v_ref: np.ndarray, tol: float) -> None:
        err = np.abs(v_out - v_ref).mean()
        mag = np.abs(v_ref).mean()
        assert err / mag <= tol

    # For some reason, wind speed and specific humidity are more numerically unstable, so we use a
    # higher tolerance of 0.5% there.
    tolerances = {
        "2t": 1e-4,
        "10u": 5e-3,
        "10v": 5e-3,
        "msl": 1e-4,
        "u": 5e-3,
        "v": 5e-3,
        "t": 1e-4,
        "q": 5e-3,
    }

    # Check the outputs.
    for k in pred.surf_vars:
        assert_approx_equality(
            pred.surf_vars[k].numpy(),
            test_output["surf_vars"][k],
            tolerances[k],
        )
    for k in pred.static_vars:
        assert_approx_equality(
            pred.static_vars[k].numpy(),
            batch.static_vars[k].numpy(),
            1e-10,  # These should be exactly equal.
        )
    for k in pred.atmos_vars:
        assert_approx_equality(
            pred.atmos_vars[k].numpy(),
            test_output["atmos_vars"][k],
            tolerances[k],
        )

    np.testing.assert_allclose(pred.metadata.lon, test_output["metadata"]["lon"])
    np.testing.assert_allclose(pred.metadata.lat, test_output["metadata"]["lat"])
    assert pred.metadata.atmos_levels == tuple(test_output["metadata"]["atmos_levels"])
    assert pred.metadata.time == tuple(test_output["metadata"]["time"])


def test_aurora_small_ddp(
    aurora_small: Aurora, test_input_output: tuple[Batch, SavedBatch]
) -> None:
    batch, test_output = test_input_output

    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("gloo", rank=0, world_size=1)

    aurora_small = torch.nn.parallel.DistributedDataParallel(aurora_small)

    # Just test that it runs.
    with torch.inference_mode():
        aurora_small.forward(batch)


def test_aurora_small_decoder_init() -> None:
    aurora_small = AuroraSmallPretrained(use_lora=True)

    # Check that the decoder heads are properly initialised. The biases should be zero, but the
    # weights shouldn't.
    for layer in [
        *aurora_small.decoder.surf_heads.values(),
        *aurora_small.decoder.atmos_heads.values(),
    ]:
        assert not torch.all(layer.weight == 0)
        assert torch.all(layer.bias == 0)


def test_aurora_small_lat_lon_matrices(
    aurora_small: Aurora, test_input_output: tuple[Batch, SavedBatch]
) -> None:
    batch, test_output = test_input_output

    with torch.inference_mode():
        pred = aurora_small.forward(batch)

        # Modify the batch to have a latitude and longitude matrices.
        n_lat = len(batch.metadata.lat)
        n_lon = len(batch.metadata.lon)
        batch.metadata.lat = batch.metadata.lat[:, None].expand(n_lat, n_lon)
        batch.metadata.lon = batch.metadata.lon[None, :].expand(n_lat, n_lon)

        pred_matrix = aurora_small.forward(batch)

    # Check the outputs.
    for k in pred.surf_vars:
        np.testing.assert_allclose(
            pred.surf_vars[k],
            pred_matrix.surf_vars[k],
            rtol=1e-5,
        )
    for k in pred.static_vars:
        np.testing.assert_allclose(
            pred.static_vars[k],
            pred_matrix.static_vars[k],
            rtol=1e-5,
        )
    for k in pred.atmos_vars:
        np.testing.assert_allclose(
            pred.atmos_vars[k],
            pred_matrix.atmos_vars[k],
            rtol=1e-5,
        )


def test_aurora_small_flags(test_input_output: tuple[Batch, SavedBatch]) -> None:
    batch, test_output = test_input_output

    flag_collections: list[dict] = [
        {},
        {"stabilise_level_agg": True},
        {"timestep": timedelta(hours=12)},
    ]

    preds = []
    for flags in flag_collections:
        model = AuroraSmallPretrained(use_lora=True, **flags)
        model.load_checkpoint(
            "microsoft/aurora",
            "aurora-0.25-small-pretrained.ckpt",
            strict=False,  # LoRA parameters not available.
        )
        model = model.double()
        model.eval()
        with torch.inference_mode():
            preds.append(model.forward(batch).normalise(model.surf_stats))

    # Check that all predictions are different.
    for i, pred1 in enumerate(preds):
        for pred2 in preds[i + 1 :]:
            for k in pred1.surf_vars:
                assert not np.allclose(
                    pred1.surf_vars[k],
                    pred2.surf_vars[k],
                    rtol=5e-2,
                )
            for k in pred1.static_vars:
                np.testing.assert_allclose(
                    pred1.static_vars[k],
                    pred2.static_vars[k],
                    rtol=1e-5,
                )
            for k in pred1.atmos_vars:
                assert not np.allclose(
                    pred1.atmos_vars[k],
                    pred2.atmos_vars[k],
                    rtol=5e-2,
                )

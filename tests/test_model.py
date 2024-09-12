"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import numpy as np
import torch

from tests.conftest import SavedBatch

from aurora import AuroraSmall, Batch


def test_aurora_small(test_input_output: tuple[Batch, SavedBatch]) -> None:
    batch, test_output = test_input_output

    model = AuroraSmall(use_lora=True)

    # Load the checkpoint and run the model.
    model.load_checkpoint(
        "microsoft/aurora",
        "aurora-0.25-small-pretrained.ckpt",
        strict=False,  # LoRA parameters not available.
    )
    model = model.double()
    model.eval()
    with torch.inference_mode():
        pred = model.forward(batch)

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


def test_aurora_small_decoder_init() -> None:
    model = AuroraSmall(use_lora=True)

    # Check that the decoder heads are properly initialised. The biases should be zero, but the
    # weights shouldn't.
    for layer in [*model.decoder.surf_heads.values(), *model.decoder.atmos_heads.values()]:
        assert not torch.all(layer.weight == 0)
        assert torch.all(layer.bias == 0)

"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import pickle
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.interpolate import RegularGridInterpolator as RGI

from aurora import AuroraSmall, Batch, Metadata


class SavedMetadata(TypedDict):
    """Type of metadata of a saved test batch."""

    lat: np.ndarray
    lon: np.ndarray
    time: list[datetime]
    atmos_levels: list[int | float]


class SavedBatch(TypedDict):
    """Type of a saved test batch."""

    surf_vars: dict[str, np.ndarray]
    static_vars: dict[str, np.ndarray]
    atmos_vars: dict[str, np.ndarray]
    metadata: SavedMetadata


def test_aurora_small() -> None:
    model = AuroraSmall(use_lora=True)

    # Load test input.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-small-pretrained-test-input.pickle",
    )
    with open(path, "rb") as f:
        test_input: SavedBatch = pickle.load(f)

    # Load test output.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-small-pretrained-test-output.pickle",
    )
    with open(path, "rb") as f:
        test_output: SavedBatch = pickle.load(f)

    # Load static variables.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-static.pickle",
    )
    with open(path, "rb") as f:
        static_vars: dict[str, np.ndarray] = pickle.load(f)

    def interpolate(v: np.ndarray) -> np.ndarray:
        """Interpolate a static variable `v` to the grid of the test data."""
        rgi = RGI(
            (
                np.linspace(90, -90, v.shape[0]),
                np.linspace(0, 360, v.shape[1], endpoint=False),
            ),
            v,
            method="linear",
            bounds_error=False,
        )
        lat_new, lon_new = np.meshgrid(
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
            indexing="ij",
            sparse=True,
        )
        return rgi((lat_new, lon_new))

    static_vars = {k: interpolate(v) for k, v in static_vars.items()}

    # Construct a proper batch from the test input.
    batch = Batch(
        surf_vars={k: torch.from_numpy(v) for k, v in test_input["surf_vars"].items()},
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={k: torch.from_numpy(v) for k, v in test_input["atmos_vars"].items()},
        metadata=Metadata(
            lat=torch.from_numpy(test_input["metadata"]["lat"]),
            lon=torch.from_numpy(test_input["metadata"]["lon"]),
            atmos_levels=tuple(test_input["metadata"]["atmos_levels"]),
            time=tuple(test_input["metadata"]["time"]),
        ),
    )

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
            static_vars[k],
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

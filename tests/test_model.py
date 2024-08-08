"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import os
import pickle
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch
from huggingface_hub import hf_hub_download

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
    model = AuroraSmall()

    # Load test input.
    path = hf_hub_download(
        repo_id=os.environ["HUGGINGFACE_REPO"],
        filename="aurora-0.25-small-pretrained-test-input.pickle",
    )
    with open(path, "rb") as f:
        test_input: SavedBatch = pickle.load(f)

    # Load test output.
    path = hf_hub_download(
        repo_id=os.environ["HUGGINGFACE_REPO"],
        filename="aurora-0.25-small-pretrained-test-output.pickle",
    )
    with open(path, "rb") as f:
        test_output: SavedBatch = pickle.load(f)

    # Load static variables.
    path = hf_hub_download(
        repo_id=os.environ["HUGGINGFACE_REPO"],
        filename="aurora-0.25-static.pickle",
    )
    with open(path, "rb") as f:
        static_vars: dict[str, np.ndarray] = pickle.load(f)

    # Select the test region for the static variables. For convenience, these are included wholly.
    lat_inds = range(140, 140 + 32)
    lon_inds = range(0, 0 + 64)
    static_vars = {k: v[lat_inds, :][:, lon_inds] for k, v in static_vars.items()}

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
    model.load_checkpoint(os.environ["HUGGINGFACE_REPO"], "aurora-0.25-small-pretrained.ckpt")
    model = model.double()
    model.eval()
    with torch.inference_mode():
        pred = model.forward(batch)

    def assert_approx_equality(v_out, v_ref) -> None:
        err = np.abs(v_out - v_ref).mean()
        mag = np.abs(v_ref).mean()
        print(err / mag)
        assert err / mag <= 1e-4

    # Check the outputs.
    for k in pred.surf_vars:
        assert_approx_equality(pred.surf_vars[k].numpy(), test_output["surf_vars"][k])
    for k in pred.static_vars:
        assert_approx_equality(pred.static_vars[k].numpy(), static_vars[k])
    for k in pred.atmos_vars:
        assert_approx_equality(pred.atmos_vars[k].numpy(), test_output["atmos_vars"][k])

    np.testing.assert_allclose(pred.metadata.lon, test_output["metadata"]["lon"])
    np.testing.assert_allclose(pred.metadata.lat, test_output["metadata"]["lat"])
    assert pred.metadata.atmos_levels == tuple(test_output["metadata"]["atmos_levels"])
    assert pred.metadata.time == tuple(test_output["metadata"]["time"])

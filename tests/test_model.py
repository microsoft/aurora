"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import os
import pickle
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from aurora import AuroraSmall, Batch, Metadata

torch.use_deterministic_algorithms(True)


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

    def all_children(x):
        return [x] + sum([all_children(xi) for xi in x.children()], [])

    for layer in all_children(model):
        if isinstance(layer, torch.nn.LayerNorm):
            layer.eps = 1e-1

    # Load test input.
    path = hf_hub_download(
        repo_id=os.environ["HUGGINGFACE_REPO"],
        filename="aurora-0.25-small-pretrained-test-input.pickle",
    )
    # path = "/home/wessel/feynman/projects/climai_global/notebooks/
    # aurora-0.25-small-pretrained-test-input.pickle"
    with open(path, "rb") as f:
        test_input: SavedBatch = pickle.load(f)

    # Load test output.
    path = hf_hub_download(
        repo_id=os.environ["HUGGINGFACE_REPO"],
        filename="aurora-0.25-small-pretrained-test-output.pickle",
    )
    # path = "/home/wessel/feynman/projects/climai_global/notebooks/
    # aurora-0.25-small-pretrained-test-output.pickle"
    with open(path, "rb") as f:
        test_output: SavedBatch = pickle.load(f)

    # Load static variables.
    path = hf_hub_download(
        repo_id=os.environ["HUGGINGFACE_REPO"],
        # filename="aurora-0.25-static.pickle",
        filename="static_vars_ecmwf_regridded.pickle",
    )
    # path = "/home/wessel/feynman/.data/weather/pde-data-preprocessed/ECMWF-IFS-HR/
    # seqrecord/static_vars_ecmwf_regridded.pickle"
    with open(path, "rb") as f:
        static_vars: dict[str, np.ndarray] = pickle.load(f)

    # Select the test region for the static variables. For convenience, these are included wholly.
    # lat_inds = range(140, 140 + 32 * 4)
    # lon_inds = range(0, 0 + 64 * 4)
    static_vars = {k: v[:-1, :][:, :] for k, v in static_vars.items()}

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
        # pred2 = model.forward(batch)

    # # Check that the outputs are deterministic by just checking the surface-level variables.
    # for k in pred.surf_vars:
    #     np.testing.assert_allclose(pred.surf_vars[k], pred2.surf_vars[k])

    def assert_approx_equality(v_out, v_ref, tol) -> None:
        err = np.abs(v_out - v_ref).mean()
        mag = np.abs(v_ref).mean()
        print(err / mag, tol, mag)
        assert err / mag <= tol

    # For some reason, wind speeds are more numerically unstable, so we use a higher tolerance of
    # 0.5% there.
    tolerances = {
        "2t": 1e-4,
        "10u": 5e-3,
        "10v": 5e-3,
        "msl": 1e-4,
        "u": 5e-3,
        "v": 5e-3,
        "t": 1e-4,
        "q": 1e-4,
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
            0,  # These should be exactly equal.
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

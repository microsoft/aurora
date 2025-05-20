"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import os
import pickle
from datetime import datetime
from typing import Generator, TypedDict

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download

from aurora import Batch, Metadata
from aurora.batch import interpolate_numpy


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


@pytest.fixture()
def test_input_output() -> Generator[tuple[Batch, SavedBatch], None, None]:
    # Load test input.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-small-pretrained-test-input.pickle",
    )
    with open(path, "rb") as f:
        test_input: SavedBatch = pickle.load(f)

    # Load static variables.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-static.pickle",
    )
    with open(path, "rb") as f:
        static_vars: dict[str, np.ndarray] = pickle.load(f)

    # Load test output.
    path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-small-pretrained-test-output.pickle",
    )
    with open(path, "rb") as f:
        test_output: SavedBatch = pickle.load(f)

    # We unfortunately used a time in 1950. Windows cannot produce timestamps for `datetime`s before
    # 1970. We fix this below.
    if os.name == "nt":

        class PatchedDateTime(datetime):
            def timestamp(self) -> float:
                # This is the value of `datetime(1950, 1, 1, 6, 0).timestamp()` on Linux.
                return -631134000.0

        test_input["metadata"]["time"] = [PatchedDateTime(1950, 1, 1, 6, 0)]

    static_vars = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            test_input["metadata"]["lat"],
            test_input["metadata"]["lon"],
        )
        for k, v in static_vars.items()
    }

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

    yield batch, test_output

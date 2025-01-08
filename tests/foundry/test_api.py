"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import datetime

import torch

from aurora import Batch, Metadata
from aurora.foundry import submit


def test_api(tmp_path, mock_foundry_client: dict):
    batch = Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17),
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )

    for i, pred in enumerate(
        submit(
            batch=batch,
            model_name="aurora-0.25-small-pretrained",
            num_steps=4,
            **mock_foundry_client,
        )
    ):
        assert isinstance(pred, Batch)
        assert pred.metadata.rollout_step == i + 1

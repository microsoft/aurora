"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata


def test_aurora_small():
    model = AuroraSmall()

    batch = Batch(
        surf_vars={k: torch.randn(1, 2, 16, 32) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(1, 2, 16, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 16, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17)[:-1],  # Cut off the south pole.
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )

    prediction = model.forward(batch)

    assert isinstance(prediction, Batch)

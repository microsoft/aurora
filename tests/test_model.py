"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata


def test_aurora_small():
    model = AuroraSmall()

    batch = Batch(
        {k: torch.randn(1, 2, 16, 32) for k in ("2t", "10u", "10v", "msl")},
        {k: torch.randn(1, 2, 16, 32) for k in ("lsm", "z", "slt")},
        {k: torch.randn(1, 2, 4, 16, 32) for k in ("z", "u", "v", "t", "q")},
        Metadata(
            torch.linspace(90, -90, 17)[:-1],  # Cut off the south pole.
            torch.linspace(0, 360, 32 + 1)[:-1],
            (datetime(2020, 6, 1, 12, 0),),
            (100, 250, 500, 850),
        ),
    )

    pred = model.forward(batch)

    assert isinstance(pred, Batch)

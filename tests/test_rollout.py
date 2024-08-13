from datetime import datetime, timedelta

import torch

from aurora import AuroraSmall, Batch, Metadata, rollout


def test_rollout():
    model = AuroraSmall()

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

    preds = list(rollout(model, batch, 10))

    assert len(preds) == 10
    for i, pred in enumerate(preds):
        expected = tuple(t + (i + 1) * timedelta(hours=6) for t in batch.metadata.time)
        assert pred.metadata.time == expected

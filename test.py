from datetime import datetime

import torch

from aurora import AuroraSmall, Batch, Metadata

model = AuroraSmall(
    level_condition=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
    dynamic_vars=True,
    atmos_static_vars=True,
    separate_perceiver=("v", "t", "q"),
)

batch = Batch(
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={
        k: torch.randn(17, 32)
        for k in (
            "lsm",
            "z",
            "slt",
        )
    },
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 17),
        lon=torch.linspace(0, 360, 32 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

print("Running model!")
prediction = model.forward(batch)
print("Done!")

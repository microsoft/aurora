import os
from datetime import datetime

import torch

from aurora import AuroraAirPollution, Batch, Metadata

model = AuroraAirPollution()
model.load_checkpoint_local(os.path.expanduser("~/checkpoints/aurora-air-pollution.ckpt"))

batch = Batch(
    surf_vars={k: torch.randn(1, 2, 16, 30) for k in ("2t", "10u", "10v", "msl", "tc_no")},
    static_vars={k: torch.randn(16, 30) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 16, 30) for k in ("z", "u", "v", "co", "so2")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 16),
        lon=torch.linspace(0, 360, 30 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

print("Running model!")
prediction = model.forward(batch)
print("Done!")

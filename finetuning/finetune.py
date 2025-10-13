from datetime import datetime

import torch

from aurora import AuroraPretrained, Batch, Metadata


def loss(pred: Batch) -> torch.Tensor:
    """A sample loss function. You should replace this with your own loss function."""
    surf_values = prediction.surf_vars.values()
    atmos_values = prediction.atmos_vars.values()
    return sum((x * x).sum() for x in tuple(surf_values) + tuple(atmos_values))


model = AuroraPretrained(autocast=True)
model.load_checkpoint()
model.configure_activation_checkpointing()
model.train()
model = model.to("cuda")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(10):
    print(f"Step {i}")

    # Train on random data. You should replace this with your own data.
    batch = Batch(
        surf_vars={k: torch.randn(1, 2, 721, 1440) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(721, 1440) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 13, 721, 1440) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 721),
            lon=torch.linspace(0, 360, 1440 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
        ),
    )

    opt.zero_grad()
    prediction = model.forward(batch.to("cuda"))
    loss_value = loss(prediction)
    loss_value.backward()
    opt.step()

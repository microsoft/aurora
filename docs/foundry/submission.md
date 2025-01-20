# Submitting Predictions

To produce predictions on Azure AI Foundry, the client will communicate with the host through
a blob storage container.

First, create a client that can communicate with your Azure AI Foundry endpoint:

```python
from aurora.foundry import FoundryClient

foundry_client = FoundryClient(
    endpoint="https://endpoint/",
    token="TOKEN",
)
```

Then set up a blob storage container for communication with the host:

```python
from aurora.foundry import BlobStorageChannel

channel = BlobStorageChannel(
    "https://my.blob.core.windows.net/container/folder?<READ_WRITE_SAS_TOKEN>"
)
```

The SAS token needs both read and write rights.
The blob storage container will be used to send the initial condition to the host and to retrieve
the predictions from the host.

```{warning}
It is important that the SAS token has both read and write rights.

To generate a SAS token with read and write rights, navigate to the container in Azure,
go to "Shared access tokens", and select both "Read" and "Write" under "Permissions".
```

You can now submit requests in the following way:

```python
from datetime import datetime

import torch
from aurora import Batch, Metadata

from aurora.foundry import submit


initial_condition = Batch(
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

for pred in submit(
    batch=initial_condition,
    model_name="aurora-0.25-small-pretrained",
    num_steps=4,
    foundry_client=foundry_client,
    channel=channel,
):
    pass  # Do something with `pred`.
```

The above uses the small model and a random initial conditions.
In practice, you want to use the fine-tuned model, `aurora-0.25-finetuned`, and an initial condition from HRES T0.
See the [HRES T0 example](/example_hres_t0).

# Submitting Predictions

To produce predictions on Azure AI Foundry, the client will communicate through
a blob storage container, so `azcopy` needs to be available in the local path.
[See here for instructions on how to install `azcopy`.](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)

First, create a client that can communicate with your Azure AI Foundry endpoint:

```python
from aurora.foundry import FoundryClient

foundry_client = FoundryClient(
    endpoint="https://endpoint/",
    token="TOKEN",
)
```

Then set up a way to communicate with the model running on Foundry.
You likely want to send data back and forth via a folder in a blob storage container:

```python
from aurora.foundry import BlobStorageCommunication

communication = BlobStorageCommunication(
    "https://my.blob.core.windows.net/container/folder?<SAS_TOKEN>"
)
```

The SAS token needs read, write, and list rights.
This API does not automatically delete the model initial condition and predictions that are
uploaded to the blob storage folder.
You will need to do that yourself.

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
    client=communication,
    host=communication,
    foundry_client=foundry_client,
):
    pass  # Do something with `pred`.
```

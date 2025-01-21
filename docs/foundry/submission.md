# Submitting Predictions

To produce Aurora predictions on Azure AI Foundry,
you need an endpoint that hosts Aurora.
To create such an endpoint, find Aurora in the [Azure AI Foundry model catalog](https://ai.azure.com/explore/models),
click "Deploy", and follow the instructions.
Once the endpoint has been deployed,
it will have an endpoint URL and access token.
Then create a `FoundryClient` using this URL and token:

```python
from aurora.foundry import FoundryClient

foundry_client = FoundryClient(
    endpoint="https://endpoint_url/",
    token="TOKEN",
)
```

You will communicate with the endpoint through a blob storage container.
You need to create this blob storage container yourself.
Create one, and generate a URL that includes a SAS token _with both read and write rights_.
Then create `BlobStorageChannel` with the blob storage container URL with SAS appended:

```python
from aurora.foundry import BlobStorageChannel

channel = BlobStorageChannel(
    "https://my.blob.core.windows.net/container/folder?<READ_WRITE_SAS_TOKEN>"
)
```

This blob storage container will be used to send the initial condition to the endpoint
and to retrieve the predictions from the endpoint.

```{warning}
It is important that the SAS token has both read and write rights.

To generate a SAS token with read and write rights, navigate to the container in Azure,
go to "Shared access tokens", and select both "Read" and "Write" under "Permissions".
```

You're all done now!
You can submit requests for predictions in the following way:

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

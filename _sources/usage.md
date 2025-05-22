# Usage

## Installation

For the latest official release, install the `pip` package:

```bash
pip install microsoft-aurora
```

Or install from conda-forge with `conda` / `mamba`:

```bash
mamba install microsoft-aurora -c conda-forge
```

You can also install directly from GitHub:

```bash
git clone https://github.com/microsoft/aurora.git
cd aurora
```

Then create a new virtual environment and install the requirements:

```bash
virtualenv venv -p python3.10
source venv/bin/activate
make install
```

## One-Step Predictions

Making predictions with the model involves three steps:

1. prepare a batch of data,
2. construct the model and load a checkpoint, and
3. run the model on the batch.

We walk through these steps in order.

First, you must construct a batch of data.
A batch of data contains surface-level variables,
static variables, atmospheric variables, and associated metadata.
This batch must be an `aurora.Batch`.
Here is an example that constructs a random batch:

```python
from datetime import datetime

import torch

from aurora import Batch, Metadata

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
```

The exact form of a `Batch` will be explained in detail on [the next page](batch).

Second, you need to construct a model and load a checkpoint.
There are various versions of Aurora, both in terms of model size and in terms of what the model was trained on.
The regular version of Aurora can be constructed as follows:

```python
from aurora import Aurora

model = Aurora()
```

In this example, however, we use a smaller version:

```python
from aurora import AuroraSmallPretrained

model = AuroraSmallPretrained()
```

The checkpoint can then be loaded with `model.load_checkpoint`:

```python
model.load_checkpoint()
```

Instead of loading the default checkpoint,
you can also load another checkpoint by specifying a checkpoint file in a HuggingFace repository.
For example, we could use `aurora-0.25-small-pretrained.ckpt` from the repository
`microsoft/aurora` (which is the default in this case):

```python
model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
```

Typically, you will want to set the model to evaluation mode, which disables e.g. drop-out:

```python
model.eval()
```

A detailed overview of all available models is given [here](models).

Finally, you are ready to run the model!

```python
model = model.to("cuda")

with torch.inference_mode():
    pred = model.forward(batch)
```

Predictions are also of the form of `aurora.Batch`.
For example, `pred.surf_vars["2t"]` gives the predictions for two-meter temperature.

You will need approximately 40 GB of GPU memory for running the regular model on global 0.25 degree data.


## Autoregressive Roll-Outs

To make predictions for more than one step ahead, you can apply the model autoregressively.
This can be done with `aurora.rollout`:

```python
from aurora import rollout

model = model.to("cuda")

with torch.inference_mode():
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=10)]
```

In the list comprehension, we move the prediction after every step immediately
to the CPU to prevent GPU memory buildup.
Every element of `preds` is again of the form of `aurora.Batch`.

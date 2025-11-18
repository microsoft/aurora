# Fine-Tuning

Generally, if you wish to fine-tune Aurora for a specific application,
you should build on the pretrained version:

```python
from aurora import AuroraPretrained

model = AuroraPretrained()
model.load_checkpoint()
```

## Fine-Tuning Environment

We provide a very basic Docker image and fine-tuning loop to get you started.
This Docker image is built from a NVIDIA PyTorch base image,
so is tailored to work for NVIDIA GPUs, and has been tested on an 80 GB A100.
The image can be found at `finetuning/Dockerfile` and the fine-tuning
loop at `finetuning/finetune.py`.
Assuming that you have cloned the Aurora repository, you can build and run
the image by running the following from the root of the repository:

```bash
docker build . -t aurora:latest -f finetuning/Dockerfile
docker run --rm -it -v .:/app/aurora \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    aurora:latest
```

Then, within the image, execute

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
    python finetuning/finetune.py
```

to run the sample fine-tuning loop.
`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` enables CUDA's built-in
asynchronous memory allocator, which is recommended for Aurora.
This loop should run on an A100 with 80 GB of memory.
If you need to reduce memory usage, you could try the following:
(a) split the model and optimiser parameters across multiple GPUs with
[FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html);
(b) use a more memory-efficient optimiser, such as
[Adafactor](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adafactor.html);
(c) split the model activations across multiple GPUs with model parallelism
(you will need to implement this yourself or use an existing framework);
(d) do CPU offloading of model or optimiser parameters; or
(e) run everything in pure `bfloat16` (this might lead to more unstable training).
You could also try to optimise the activation checkpointing strategy, to
see if there is something more to gain there.

For example, on Azure, launch a VM with size `Standard_NC24ads_A100_v4`, image
Ubuntu 24.04 LTS (x64), and 256 GB of disk space.
Then [install CUDA](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup).
Be sure to install the latest supported version of the CUDA Toolkit by
checking `nvidia-smi` after installing the drivers with
`sudo ubuntu-drivers autoinstall` and rebooting.
Best performance is achieved with CUDA Toolkit 13.0 or higher, which
requires drivers that support CUDA 13.0 or higher.
Then install Docker with `sudo apt install docker.io`,
set the right permissions for the current user with
`sudo usermod -a -G docker $USER`,
[install the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and reboot.
You should now be able to clone the repo and build and run the image using
the instructions above.



## Computing Gradients

To compute gradients, you will need an A100 with 80 GB of memory.
In addition, you will need to use reduced precision and gradient checkpointing.
You can do this as follows:

```python
from aurora import AuroraPretrained

model = AuroraPretrained(autocast=True)
model.load_checkpoint()

batch = ...  # Load some data.

model = model.cuda()
model.train()
model.configure_activation_checkpointing()

pred = model.forward(batch)
loss = ...
loss.backward()
```

Here `autocast` enables AMP with `bfloat16` for only the backbone.
This is necessary to be able to fit gradients in memory.

## Exploding Gradients

When fine-tuning, you may run into very large gradient values.
Gradient clipping and internal layer normalisation layers mitigate the impact
of large gradients,
meaning that large gradients will not immediately lead to abnormal model outputs and loss values.
Nevertheless, if gradients do blow up, the model will not learn anymore and eventually the loss value
will also blow up.
You should carefully monitor the value of the gradients to detect exploding gradients.

One cause of exploding gradients is too large values for internal activations.
Typically this can be fixed by judiciously inserting a layer normalisation layer.

We have identified the level aggregation as weak point of the model that can be susceptible
to exploding gradients.
You can stabilise the level aggregation of the model
by setting the following flag in the constructor: `stabilise_level_agg=True`.
Note that `stabilise_level_agg=True` will considerably perturb the model,
so significant additional fine-tuning may be required to get to the desired level of performance.

```python
from aurora import AuroraPretrained
from aurora.normalisation import locations, scales

model = AuroraPretrained(stabilise_level_agg=True)  # Insert extra layer norm. to mitigate exploding gradients.
model.load_checkpoint(strict=False)
```

## Extending Aurora with New Variables

Aurora can be extended with new variables by adjusting the keyword arguments `surf_vars`,
`static_vars`, and `atmos_vars`.
When you add a new variable, you also need to set the normalisation statistics.

```python
from aurora import AuroraPretrained
from aurora.normalisation import locations, scales

model = AuroraPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "new_surf_var"),
    static_vars=("lsm", "z", "slt", "new_static_var"),
    atmos_vars=("z", "u", "v", "t", "q", "new_atmos_var"),
)
model.load_checkpoint(strict=False)

# Normalisation means:
locations["new_surf_var"] = 0.0
locations["new_static_var"] = 0.0
locations["new_atmos_var"] = 0.0

# Normalisation standard deviations:
scales["new_surf_var"] = 1.0
scales["new_static_var"] = 1.0
scales["new_atmos_var"] = 1.0
```

To more efficiently learn new variables, it is recommended to use a separate learning rate for
the patch embeddings of the new variables in the encoder and decoder.
For example, if you are using Adam, you can try `1e-3` for the new patch embeddings
and `3e-4` for the other parameters.

By default, patch embeddings in the encoder for new variables are initialised randomly.
This means that adding new variables to the model perturbs the predictions for the existing
variables.
If you do not want this, you can alternatively initialise the new patch embeddings in the encoder
to zero.
The relevant parameter dictionaries are `model.encoder.{surf,atmos}_token_embeds.weights`.

## Other Model Extensions

It is possible to extend to model in any way you like.
If you do this, you will likely add or remove parameters.
Then `model.load_checkpoint` will error,
because the existing checkpoint now mismatches with the model's parameters.
Simply set `model.load_checkpoint(..., strict=False)` to ignore the mismatches:

```python
from aurora import AuroraPretrained

model = AuroraPretrained(...)

... # Modify `model`.

model.load_checkpoint(strict=False)
```

## Triple Check Your Fine-Tuning Data!

When fine-tuning the model, it is absolutely essential to carefully check your fine-tuning data.

* Are the old (and possibly new) normalisation statistics appropriate for the new data?

* Is any data missing?

* Does the data contains zeros or NaNs?

* Does the data contain any outliers that could possibly interfere with fine-tuning?

_Et cetera._

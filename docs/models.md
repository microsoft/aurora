# Available Models

Weights for models are made available through our [HuggingFace repository `microsoft/aurora`](https://huggingface.co/microsoft/aurora).
We now describe the available models in turn.

## Aurora 0.25° Pretrained

Aurora 0.25° Pretrained is a version of Aurora trained on a wide variety of data.

### Usage

```python
from aurora import Aurora

model = Aurora(use_lora=False)  # Model is not fine-tuned.
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
```

### Recommended Use

Use this version of Aurora if no fine-tuned version exists for your specific data set.
For example, if you wish to make predictions for ERA5 at 0.25° resolution, this version is appropriate.
Note that 0.25° resolution means that the data has dimensions `(721, 1440)`.

Also use Aurora 0.25° Pretrained if you plan to fine-tune Aurora for you specific application,
_even if your application operates at another resolution_.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl` |
| Static variables | `lsm`, `slt`, `z` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.25° Pretrained requires
[static variables from ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).
For convenience, these are also available in
[the HuggingFace repository](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-static.pickle).

## Aurora 0.25° Pretrained Small

Aurora 0.25° Pretrained Small is, as the name suggests, a smaller version of Aurora 0.25° Pretrained.

### Usage

```python
from aurora import AuroraSmall

model = AuroraSmall()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
```

### Recommended Use

Use this model for debugging purposes.
We do not recommend any other use.

## Aurora 0.25° Fine-Tuned

Aurora 0.25° Fine-Tuned is Aurora 0.25° Pretrained fine-tuned on IFS HRES T0.

### Usage

```python
from aurora import Aurora

model = Aurora()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt")
```

### Recommended Use

Use Aurora 0.25° Fine-Tuned if you aim to make predictions for IFS HRES T0.
Aurora 0.25° Fine-Tuned is the best performing version of Aurora at 0.25° resolution.

**Important:**
For optimal performance, it is crucial that you only use Aurora 0.25° Fine-Tuned for IFS HRES T0.
Producing predictions for any other data set will likely give sensible predictions,
but performance may not be optimal anymore.
[Note also that IFS HRES T0 is _not_ the same as IFS HRES analysis.](t0-vs-analysis)

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl` |
| Static variables | `lsm`, `slt`, `z` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.25° Fine-Tuned requires
[static variables from ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).
For convenience, these are also available in
[the HuggingFace repository](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-static.pickle).

(lora-or-no-lora)=
### Notes

If you require more realistic predictions are the expense of slightly higher MSE at longer lead times, you can try turning off LoRA.

| Use LoRA? | Effect |
| - | - |
| Yes | Optimal long-term MSE, but slightly blurrier predictions |
| No | More realistic predictions, but slightly higher long-term MSE |

You can turn off LoRA as follows:

```python
from aurora import Aurora

model = Aurora(use_lora=False)  # Disable LoRA for more realistic samples.
model.load_checkpoint("microsoft/aurora", "aurora-0.25-finetuned.ckpt", strict=False)
```

## Aurora 0.1° Fine-Tuned

Aurora 0.1° Fine-Tuned is a high-resolution version of Aurora.

### Usage

```python
from aurora import AuroraHighRes

model = AuroraHighRes()
model.load_checkpoint("microsoft/aurora", "aurora-0.1-finetuned.ckpt")
```

### Recommended Use

Use Aurora 0.1° Fine-Tuned if you aim to make predictions for IFS HRES T0 at 0.1° resolution.
Note that 0.1° resolution means that the data should have dimensions `(1801, 3600)`.
Aurora 0.1° Fine-Tuned is the best performing version of Aurora at 0.1° resolution.

**Important:**
For optimal performance, it is crucial that you only use Aurora 0.1° Fine-Tuned for IFS HRES analysis.
Producing predictions for any other data set will likely give sensible predictions,
but performance may be significantly affected.
[Note also that IFS HRES T0 is _not_ the same as IFS HRES analysis.](t0-vs-analysis)

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl` |
| Static variables | `lsm`, `slt`, `z` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.1° Fine-Tuned requires
[static variables from IFS HRES analysis](https://rda.ucar.edu/datasets/ds113.1/) regridded
to 0.1° resolution.
Because of differences between implementations of regridding methods, the resulting static
variables might not be exactly equal to the ones we used during training.
For this reason we also uploaded
[the exact static variables which we used during training](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.1-static.pickle).
To use these, you must remove an exception to the normalisation by instantiating
the model in the following way:

```python
from aurora import AuroraHighRes

model = AuroraHighRes(surf_stats=None)  # Use static variables from HF repo.
model.load_checkpoint("microsoft/aurora", "aurora-0.1-finetuned.ckpt")
```

### Notes

[Like for Aurora 0.25° Fine-Tuned](lora-or-no-lora),
you can turn off LoRA to obtain more realistic predictions at the expensive of slightly higher long-term MSE:

```python
from aurora import AuroraHighRes

model = AuroraHighRes(use_lora=False)  # Disable LoRA for more realistic samples.
model.load_checkpoint("microsoft/aurora", "aurora-0.1-finetuned.ckpt", strict=False)
```

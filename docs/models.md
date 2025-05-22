# Available Models

Weights for models are made available through our [HuggingFace repository `microsoft/aurora`](https://huggingface.co/microsoft/aurora).
We now describe the available models in turn.

## Aurora 0.25° Pretrained

Aurora 0.25° Pretrained is a version of Aurora trained on a wide variety of data.

### Usage

```python
from aurora import AuroraPretrained

model = AuroraPretrained()
model.load_checkpoint()
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

## Aurora 0.25° Small Pretrained

Aurora 0.25° Small Pretrained is, as the name suggests, a smaller version of Aurora 0.25° Pretrained.

### Usage

```python
from aurora import AuroraSmallPretrained

model = AuroraSmallPretrained()
model.load_checkpoint()
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
model.load_checkpoint()
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
model.load_checkpoint(strict=False)
```

## Aurora 0.25° 12-Hour Pretrained

Aurora 0.25° 12-Hour Pretrained is Aurora 0.25° Pretrained with a 12-hour lead time.

### Usage

```python
from aurora import Aurora12hPretrained

model = Aurora12hPretrained()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 0.25° 12-Hour Pretrained if you wish to make predictions with a 12-hour lead time.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl` |
| Static variables | `lsm`, `slt`, `z` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.25° 12-Hour Pretrained requires
[static variables from ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).
For convenience, these are also available in
[the HuggingFace repository](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-static.pickle).


## Aurora 0.1° Fine-Tuned

Aurora 0.1° Fine-Tuned is a high-resolution version of Aurora.

### Usage

```python
from aurora import AuroraHighRes

model = AuroraHighRes()
model.load_checkpoint()
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


Due to differences between implementations of regridding methods, it is recommended to use
[the exact static variables which we used during training](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.1-static.pickle).

It is also possible to use the
[static variables from IFS HRES analysis](https://rda.ucar.edu/datasets/ds113.1/) regridded
to 0.1° resolution.
However, these static variables will not be exactly equal to the ones we used, which might impact
performance.
If you download the static variables yourself, you must adjust the normalisation statistics.
You can do that in the following way:

```python
from aurora import AuroraHighRes

model = AuroraHighRes(
    # Use manually downloaded and regridded static variables.
    surf_stats={"z": (-3.270407e03, 6.540335e04)},
)

model.load_checkpoint()
```

The specific values above should work reasonably.
<!-- Jupyter book complains that the below link doesn't work, but it does. -->
See [the API](api.rst#aurora.Aurora.__init__) for a description of `surf_vars`.
Generally, the first value in the tuple should be `min(static_z)`
and the second value `max(static_z) - min(static_z)`.

### Notes

[Like for Aurora 0.25° Fine-Tuned](lora-or-no-lora),
you can turn off LoRA to obtain more realistic predictions at the expensive of slightly higher long-term MSE:

```python
from aurora import AuroraHighRes

model = AuroraHighRes(use_lora=False)  # Disable LoRA for more realistic samples.
model.load_checkpoint(strict=False)
```

(aurora-air-pollution)=
## Aurora 0.4° Air Pollution

Aurora 0.4° Air Pollution is Aurora 0.25° Pretrained fine-tuned on
[CAMS analysis data](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts).
This version of Aurora is capable of making air pollution forecasts.

### Usage

```python
from aurora import AuroraAirPollution

model = AuroraAirPollution()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 0.4° Air Pollution if you aim to make predictions for CAMS analysis.

**Important:**
For optimal performance, it is crucial that you only run Aurora 0.4° Air Pollution on CAMS analysis data.
Producing predictions for any other data set might give sensible predictions,
but performance may not be optimal anymore.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl`, `pm1`, `pm2p5`, `pm10`, `tcco`, `tc_no`, `tcno2`, `tcso2`, `gtco3` |
| Static variables | `lsm`, `slt`, `z`, `static_ammonia`, `static_ammonia_log`, `static_co`, `static_co_log`, `static_nox`, `static_nox_log`, `static_so2`, `static_so2_log`  |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z`, `co`, `no`, `no2`, `so2`, `go3` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.4° Air Pollution requires
[static variables from the HuggingFace repository](https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.4-air-pollution-static.pickle).

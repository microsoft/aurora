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

Use Aurora 0.1° Fine-Tuned if you aim to make predictions for IFS HRES analysis at 0.1° resolution.
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
Note that 0.4° resolution means that the data should have dimensions `(451, 900)`.

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

(aurora-wave)=
## Aurora 0.25° Wave

Aurora 0.25° Wave is Aurora 0.25° Pretrained fine-tuned on
[HRES-WAM ocean wave data](https://www.ecmwf.int/en/forecasts/datasets/set-ii).
This version of Aurora is capable of making ocean wave forecasts.

### Usage

```python
from aurora import AuroraWave

model = AuroraWave()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 0.25° Wave if you aim to make predictions for HRES-WAM analysis data combined with HRES T0.

**Important:**
Some specific postprocessing applies to the HRES-WAM data.
See Section C.5 of the [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09005-y/MediaObjects/41586_2025_9005_MOESM1_ESM.pdf).

**Important:**
For optimal performance, it is crucial that you only run Aurora 0.25° Wave on batches with all
meteorological variables taken from HRES T0 and all ocean wave variables taken from HRES-WAM
analysis.
Producing predictions for any other combination might give sensible predictions,
but performance may not be optimal anymore.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `swh`, `mwd`, `mwp`, `pp1d`, `shww`, `mdww`, `mpww`, `shts`, `mdts`, `mpts`, `swh1`, `mwd1`, `mwp1`, `swh2`, `mwd2`, `mwp2`, `10u_wave`, `10v_wave`, `wind` |
| Static variables | `lsm`, `slt`, `z`, `wmb`, `lat_mask` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.25° Wave requires
[static variables from the HuggingFace repository](https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.25-wave-static.pickle).

(aurora-025-v15)=
## Aurora 1.5 (0.25°)

Aurora 1.5 is a new fine-tuned version of Aurora with substantially expanded surface variables,
variable lead-time support, and prescribed solar insolation.

### Usage

```python
from aurora import AuroraV1p5

model = AuroraV1p5()
model.load_checkpoint()
```

### Recommended Use

We recommend using Aurora 1.5 over Aurora 0.25° Fine-Tuned since its accuracy is improved and
it provides access to the expanded variable set and hourly lead time.
This 1.5-family model makes deterministic forecasts with IFS HRES T0 data at 0.25° resolution.
It was built starting with Aurora 0.25° Pretrained but extensively fine-tuned with ERA5 data for the
new variables and subsequently on IFS analysis data for real-time use.

Aurora 1.5 extends Aurora with:

- **26 surface-level variables** (including solar insolation), up from 4 in Aurora 0.25°;
- **36 static variables** covering land-surface properties, vegetation, soil type, and orography;
- **variable lead-time embeddings**, enabling predictions at any lead time as fine as one hour
  rather than being restricted to the fixed 6-hour base timestep;
- **7 output-only surface variables** that the model predicts but that are not required in the
  input (wind gusts, boundary layer height, radiation fluxes, precipitation, and snowfall).
  These are automatically zero-padded by the model during rollout.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level input variables | `2t`, `10u`, `10v`, `msl`, `2d`, `tcwv`, `tcc`, `100u`, `100v`, `sp`, `lcc`, `mcc`, `hcc`, `skt`, `stl1`, `swvl1`, `ci`, `scaled_sd`, `insolation` |
| Surface-level output-only variables | `i10fg`, `blh`, `uvb_1h`, `ssrd_1h`, `ttr_1h`, `scaled_tp_1h`, `scaled_sf_1h` |
| Static variables | `lsm`, `z`, `anor`, `isor`, `cvh`, `cl`, `dl`, `cvl`, `slor`, `slt_0`–`slt_7`, `sdfor`, `sdor`, `tvh_0`, `tvh_3`–`tvh_6`, `tvh_18`, `tvh_19`, `tvl_0`–`tvl_2`, `tvl_7`, `tvl_9`–`tvl_11`, `tvl_13`, `tvl_16`, `tvl_17` |
| Atmospheric variables | `z`, `u`, `v`, `t`, `q` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |

The `scaled_sd` (snow depth) and the output-only variables beginning with `scaled_` use a
log-transform; the model handles this internally as it does for certain atmospheric chemistry
parameters. Despite the variable name, the user should provide un-scaled snow depth as input.
The `insolation` variable is a prescribed solar insolation value computed automatically from
the batch's valid time and does not need to be provided by the user.

### Static Variables

Aurora 1.5 requires an extended set of 36 static variables available from the
[HuggingFace repository](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-v1.5-static.pickle).

### Hourly Sub-Steps

Aurora 1.5 supports variable lead-time embeddings, enabling predictions at lead times
finer than the 6-hour base timestep. To produce hourly predictions during rollout, pass
`fine_lead_times` to `rollout`:

```python
from aurora import rollout

steps = 4  # 4 main 6-hour steps.
# Within every main step, also predict the following hours. Combined with the four main steps,
# this leads to hourly predictions up to 24 hours.
fine_lead_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

with torch.inference_mode():
    preds = [
        pred.to("cpu")
        for pred in rollout(model, batch, steps=steps, fine_lead_times=fine_lead_times)
    ]
```

The last entry in `fine_lead_times` must equal the model's base timestep (6 hours).
Only the prediction at the final sub-step is fed back autoregressively; intermediate sub-step
predictions are produced from the same previous-step input and do not update the autoregressive
state.

### Rollout Input Clipping

To help guard against a continuous rollout of Aurora 1.5 eventually drifting to unrealistic physical
states, the model `rollout` includes an option `apply_rollout_input_clipping` (default: `True`)
which clips certain surface variables to known minimum and/or maximum bounds (e.g., preventing
cloud cover from going outside of `[0, 1]`). This clipping is *only* applied to the data in a
Batch being fed back into the model to match the training regimen - it is possible that the
model's forward pass will generate predictions outside of the clipped bounds.
`AuroraV1p5` defines the default clipping used in training; it can be overridden with the
`rollout_input_clipping` keyword argument.

## Aurora 1.5 Ensemble (0.25°)

Aurora 1.5 Ensemble is the stochastic ensemble version of Aurora 1.5.

### Usage

```python
from aurora import AuroraV1p5Ensemble

model = AuroraV1p5Ensemble()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 1.5 Ensemble when you need probabilistic forecasts or an ensemble of
plausible future states. It has the same variable set and capabilities as Aurora 1.5,
with the addition of stochastic noise injection for ensemble variability.

Ensemble members are generated by running the model multiple times.
Because the noise conditioning is stochastic, each forward pass produces a
distinct, physically plausible trajectory. The noise is injected via the backbone and
accumulates continuously across both sub-steps and main autoregressive steps, giving
smooth and temporally coherent ensemble members.
The model may be run on perturbed initial conditions from the ECMWF ENS model or on the same
initial conditions, with the former option providing more spread at early lead times.

The `AuroraV1p5Ensemble` class is identical to `AuroraV1p5` in terms of variables,
pressure levels, and rollout behaviour (including `fine_lead_times` support).
See [Aurora 1.5](aurora-025-v15) for the full variable table and sub-step guidance.

### Noise Accumulation

When using `rollout` with `fine_lead_times`, noise accumulation is enabled by default
(`use_noise_accumulation=True`). This keeps the noise correlated across sub-steps for
smoother intra-step transitions while using independent effective noise between main steps,
matching the training regimen. Set `use_noise_accumulation=False` to draw independent
noise at each sub-step instead, though this is not recommended.

# Form of a Batch

You must feed data to the model in the form of a `aurora.Batch`.
We now explain the exact form of `aurora.Batch`.

## Overall Structure

Batches contain four things:

1. some surface-level variables,
2. some static variables,
3. some atmospheric variables all at the same collection of pressure levels, and
4. metadata describing these variables: latitudes, longitudes,
    the pressure levels of the atmospheric variables, and the time of the data.

All variables in a batch are unnormalised.
Normalisation happens internally in the model.

Before we explain the four components in detail, here is an example with randomly generated data:

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

## `Batch.surf_vars`

`Batch.surf_vars` is a dictionary mapping names of surface-level variables to the numerical values
of the variables.
The surface-level variables must be of the form `(b, t, h, w)` where `b` is the batch size,
`t` the history dimension, `h` the number of latitudes, and `w` the number of longitudes.

All Aurora models produce the prediction for the next step from the current _and_ previous step.
`surf_vars[:, 1, :, :]` must correspond to the current step,
and `surf_vars[:, 0, :, :]` must correspond to the previous step, so the step before that.

The following surface-level variables are allowed:

| Name | Description |
| - | - |
| `2t` | Two-meter temperature in `K` |
| `10u` | Ten-meter eastward wind speed in `m/s` |
| `10v` | Ten-meter southward wind speed in `m/s` |
| `msl` | Mean sea-level pressure in `Pa` |

For [Aurora 0.4° Air Pollution](aurora-air-pollution), the following surface-level variables are
also allowed:

| Name | Description |
| - | - |
| `pm1` | Particulate matter less than `1 um` in `kg/m^3` |
| `pm2p5` | Particulate matter less than `2.5 um` in `kg/m^3` |
| `pm10` | Particulate matter less than `10 um` in `kg/m^3` |
| `tcco` | Total column carbon monoxide in `kg/m^2` |
| `tc_no` | Total column nitrogen monoxide in `kg/m^2` |
| `tcno2` | Total column nitrogen dioxide in `kg/m^2` |
| `tcso2` | Total column sulphur dioxide in `kg/m^2` |
| `gtco3` | Total column ozone in `kg/m^2` |

## `Batch.static_vars`

`Batch.static_vars` is a dictionary mapping names of static variables to the
numerical values of the variables.
The static variables must be of the form `(h, w)` where `h` is the number of latitudes
and `w` the number of longitudes.

The following static variables are allowed:

| Name | Description |
| - | - |
| `lsm` | [Land-sea mask](https://codes.ecmwf.int/grib/param-db/172) |
| `slt` | [Soil type](https://codes.ecmwf.int/grib/param-db/43) |
| `z` | Surface-level geopotential in `m^2/s^2` |

## `Batch.atmos_vars`

`Batch.atmos_vars` is a dictionary mapping names of atmospheric variables to the
numerical values of the variables.
The atmospheric variables must be of the form `(b, t, c, h, w)` where `b` is the batch size,
`t` the history dimension, `c` the number of pressure levels, `h` the number of latitudes,
and `w` the number of longitudes.
All atmospheric variables must contain the same collection of pressure levels in the same order.

The following atmospheric variables are allowed:

| Name | Description |
| - | - |
| `t` | Temperature in `K` |
| `u` | Eastward wind speed in `m/s` |
| `v` | Southward wind speed in `m/s` |
| `q` | Specific humidity in `kg/kg` |
| `z` | Geopotential in `m^2/s^2` |

For [Aurora 0.4° Air Pollution](aurora-air-pollution), the following atmospheric variables are
also allowed:

| Name | Description |
| - | - |
| `co` | Carbon monoxide in `kg/kg` |
| `no` | Nitrogen monoxide in `kg/kg` |
| `no2` | Nitrogen dioxide in `kg/kg` |
| `so2` | Sulphur dioxide in `kg/kg` |
| `go3` | Ozone in `kg/kg` |

## `Batch.metadata`

`Batch.metadata` must be a `Metadata`, which contains the following fields:

* `Metadata.lat` is the vector of latitudes.
    The latitudes must be _decreasing_.
    The latitudes can either include both endpoints, like `linspace(90, -90, 721)`,
    or not include the south pole, like `linspace(90, -90, 721)[:-1]`.
    For curvilinear grids, this can also be a matrix, in which case the foregoing conditions
    apply to every _column_.
* `Metadata.lon` is the vector of longitudes.
    The longitudes must be _increasing_.
    The longitudes must be in the range `[0, 360)`, so they can include zero and cannot include 360.
    For curvilinear grids, this can also be a matrix, in which case the foregoing conditions
    apply to every _row_.
* `Metadata.atmos_levels` is a `tuple` of the pressure levels of the atmospheric variables in hPa.
    Note that these levels must be in exactly correspond to the order of the atmospheric variables.
    Note also that `Metadata.atmos_levels` should be a `tuple`, not a `list`.
* `Metadata.time` is a `tuple` with, for each batch element, a `datetime.datetime` representing the time of the data.
    If the batch size is one, then this will be a one-element `tuple`, e.g. `(datetime(2024, 1, 1, 12, 0),)`.
    Since all Aurora models require variables for the current _and_ previous step,
    `Metadata.time` corresponds to the time of the _current_ step.
    Specifically, `Metadata.time[i]` corresponds to the time of `Batch.surf_vars[i, -1]`.

## Model Output

The output of `aurora.forward(batch)` will again be a `Batch`.
This batch is of exactly the same form, with only one difference:
the history dimension will have size one.

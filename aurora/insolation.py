"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Approximate solar insolation computation based on orbital mechanics.
Reference: https://brian-rose.github.io/ClimateLaboratoryBook/courseware/insolation.html
"""

from datetime import datetime
from typing import Sequence, Union

import numpy as np

__all__ = ["insolation"]


def insolation(
    dates: Union[Sequence[datetime], np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    s0: float = 1.0,
    daily: bool = False,
    enforce_2d: bool = False,
    clip_zero: bool = False,
) -> np.ndarray:
    """Calculate approximate solar insolation for given dates, latitudes, and longitudes.

    Uses 1995 orbital elements (standard in the climate modelling community).

    Args:
        dates: 1-D sequence of datetime-like objects.
        lat: 1-D or 2-D array of latitudes in degrees (`-90` to `90`).
        lon: 1-D or 2-D array of longitudes in degrees (`0` to `360`). If 2-D, must have the same
            shape as `lat`.
        s0: Scaling factor (solar constant). Defaults to `1.0`.
        daily: If `True`, return the daily maximum solar radiation (depends only on latitude and day
            of year). Defaults to `False`.
        enforce_2d: If `True` and `lat` / `lon` are 1-D, broadcast them into 2-D meshgrids. Defaults
            to `False`.
        clip_zero: If `True`, set negative (night-time) values to zero. Defaults to `False`.

    Returns:
        :class:`numpy.ndarray`: Insolation array of shape `(len(dates), *lat.shape)`.
    """
    if lat.ndim != lon.ndim:
        raise ValueError("`lat` and `lon` must have the same number of dimensions.")
    if lat.ndim == 1 and enforce_2d:
        lon, lat = np.meshgrid(lon, lat)
    if lat.shape != lon.shape:  # Assert same shape after potential broadcasting
        raise ValueError(f"Shape mismatch between `lat` (`{lat.shape}`) and `lon` (`{lon.shape}`).")
    n_dim = len(lat.shape)
    lat = lat.astype(np.float32)  # No mutation - safe to redefine the local variable.

    # Constants for year 1995.
    eps = 23.4441 * np.pi / 180.0  # Obliquity of Earth
    ecc = 0.016715  # Eccentricity of Earth's orbit
    om = 282.7 * np.pi / 180.0  # Longitude of perihelion
    beta = np.sqrt(1 - ecc**2.0)

    # Day of year as a float.
    dates_arr = np.array(dates, dtype="datetime64")
    start_years = dates_arr.astype("datetime64[Y]")
    days_arr = ((dates_arr - start_years) / np.timedelta64(1, "D")).astype(np.float32)
    for _ in range(n_dim):
        days_arr = np.expand_dims(days_arr, -1)

    if daily:
        days_arr = 0.5 + np.round(days_arr)
        new_lon = lon.astype(np.float32, copy=True)  # Copy to safely mutate.
        new_lon[:] = 0.0
    else:
        new_lon = lon.astype(np.float32)  # No mutation - safe to redefine the local variable.

    # Longitude of Earth relative to the orbit (1st-order approximation).
    lambda_m0 = ecc * (1.0 + beta) * np.sin(om)
    lambda_m = lambda_m0 + 2.0 * np.pi * (days_arr - 80.5) / 365.0
    lambda_ = lambda_m + 2.0 * ecc * np.sin(lambda_m - om)

    # Solar declination.
    dec = np.arcsin(np.sin(eps) * np.sin(lambda_)).astype("float32")
    # Hour angle.
    h = (2 * np.pi * (days_arr + new_lon / 360.0)).astype("float32", copy=False)
    # Earth-Sun distance factor.
    rho = ((1.0 - ecc**2.0) / (1.0 + ecc * np.cos(lambda_ - om))).astype("float32", copy=False)

    # Insolation.
    diff = np.sin(np.pi / 180.0 * lat[None, ...]) * np.sin(dec) - np.cos(
        np.pi / 180.0 * lat[None, ...]
    ) * np.cos(dec) * np.cos(h)
    sol = s0 * diff * rho**-2.0
    if clip_zero:
        sol[sol < 0.0] = 0.0

    return sol

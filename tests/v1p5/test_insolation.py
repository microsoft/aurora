"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the insolation computation.
"""

from datetime import datetime

import numpy as np
import pytest

from aurora import insolation


def test_shape_1d():
    dates = [datetime(2023, 6, 15, 12)]
    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 360, 37, endpoint=False)
    sol = insolation(dates, lat, lon)
    assert sol.shape == (1, 37)


def test_shape_2d_enforce():
    dates = [datetime(2023, 6, 15, 12)]
    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 360, 72, endpoint=False)
    sol = insolation(dates, lat, lon, enforce_2d=True)
    assert sol.shape == (1, 37, 72)


def test_shape_multidates():
    dates = [datetime(2023, 1, 1), datetime(2023, 7, 1)]
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(0, 360, 10, endpoint=False)
    sol = insolation(dates, lat, lon, enforce_2d=True)
    assert sol.shape == (2, 5, 10)


def test_scaling_factor():
    dates = [datetime(2023, 6, 15, 12)]
    lat = np.array([0.0])
    lon = np.array([0.0])
    sol1 = insolation(dates, lat, lon, s0=1.0)
    sol2 = insolation(dates, lat, lon, s0=2.0)
    np.testing.assert_allclose(sol2, 2.0 * sol1)


def test_clip_zero():
    dates = [datetime(2023, 6, 15, 0)]  # Midnight UTC
    lat = np.linspace(-90, 90, 37)
    lon = np.linspace(0, 360, 72, endpoint=False)
    sol = insolation(dates, lat, lon, enforce_2d=True, clip_zero=True)
    assert (sol >= 0.0).all()


def test_daily_ignores_longitude():
    dates = [datetime(2023, 6, 15, 12)]
    lat = np.array([45.0])
    lon1 = np.array([0.0])
    lon2 = np.array([180.0])
    sol1 = insolation(dates, lat, lon1, daily=True)
    sol2 = insolation(dates, lat, lon2, daily=True)
    np.testing.assert_allclose(sol1, sol2)


def test_dimension_mismatch_raises():
    dates = [datetime(2023, 6, 15, 12)]
    lat = np.linspace(-90, 90, 5)
    lon = np.zeros((5, 10))
    with pytest.raises(ValueError, match="same number of dimensions"):
        insolation(dates, lat, lon)


def test_shape_mismatch_2d_raises():
    dates = [datetime(2023, 6, 15, 12)]
    lat = np.zeros((5, 10))
    lon = np.zeros((3, 10))
    with pytest.raises(ValueError, match="(?i)shape mismatch"):
        insolation(dates, lat, lon)

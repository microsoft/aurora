"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for log-transform and log-untransform functions.
"""

import torch

from aurora.normalisation import log_transform, log_untransform


def test_roundtrip():
    x = torch.rand(10) * 5.0  # Positive values
    result = log_untransform(log_transform(x))
    torch.testing.assert_close(result, x, rtol=1e-5, atol=1e-6)


def test_zero_maps_to_zero():
    x = torch.tensor([0.0])
    scaled = log_transform(x)
    torch.testing.assert_close(scaled, torch.tensor([0.0]), atol=1e-7, rtol=0.0)


def test_monotonic():
    x = torch.linspace(0, 10, 100)
    scaled = log_transform(x)
    assert (scaled[1:] > scaled[:-1]).all()


def test_untransform_of_zero():
    # log_transform(0) == 0, so log_untransform(0) should be 0.
    result = log_untransform(torch.tensor([0.0]))
    torch.testing.assert_close(result, torch.tensor([0.0]), atol=1e-7, rtol=0.0)

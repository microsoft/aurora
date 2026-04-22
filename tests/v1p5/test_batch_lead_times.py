"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for lead_times plumbing through Batch operations.
"""

import torch

from ._helpers import _make_batch


def test_normalise_preserves_lead_times():
    lt = torch.tensor([6.0])
    batch = _make_batch(lead_times=lt)
    normed = batch.normalise(surf_stats={})
    torch.testing.assert_close(normed.lead_times, lt)


def test_unnormalise_preserves_lead_times():
    lt = torch.tensor([6.0])
    batch = _make_batch(lead_times=lt)
    unnormed = batch.unnormalise(surf_stats={})
    torch.testing.assert_close(unnormed.lead_times, lt)


def test_crop_preserves_lead_times():
    lt = torch.tensor([6.0])
    batch = _make_batch(lead_times=lt)
    cropped = batch.crop(4)
    torch.testing.assert_close(cropped.lead_times, lt)


def test_to_device_preserves_lead_times():
    lt = torch.tensor([6.0])
    batch = _make_batch(lead_times=lt)
    moved = batch.to("cpu")
    torch.testing.assert_close(moved.lead_times, lt)

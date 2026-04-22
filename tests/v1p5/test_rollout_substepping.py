"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for rollout sub-stepping with variable lead times.
"""

from datetime import timedelta

import pytest
import torch

from ._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora import Aurora, rollout


def test_fine_lead_times_produces_more_outputs():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )

    steps = 2
    fine_lead_times = [3.0, 6.0]
    with torch.inference_mode():
        preds = list(rollout(model, batch, steps, fine_lead_times=fine_lead_times))

    # Should be `steps * len(fine_lead_times)` outputs.
    assert len(preds) == steps * len(fine_lead_times)


def test_fine_lead_times_correct_output_times():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )

    fine_lead_times = [2.0, 4.0, 6.0]
    with torch.inference_mode():
        preds = list(rollout(model, batch, steps=2, fine_lead_times=fine_lead_times))

    base_time = batch.metadata.time[0]
    for i, lt in enumerate(fine_lead_times):
        expected = base_time + timedelta(hours=lt)
        assert preds[i].metadata.time[0] == expected


def test_fine_lead_times_requires_variable_lead_time():
    # Create a plain `Aurora` model (`variable_lead_time=False` by default).
    model = Aurora(use_lora=False)
    model.eval()
    batch = _make_batch(
        surf_vars=("2t", "10u", "10v", "msl"),
        static_vars=("lsm", "z"),
        atmos_vars=("z", "u", "v", "t", "q"),
    )
    with pytest.raises(ValueError, match="variable_lead_time"):
        list(rollout(model, batch, steps=1, fine_lead_times=[3.0, 6.0]))


def test_standard_rollout_still_works():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )

    steps = 3
    with torch.inference_mode():
        preds = list(rollout(model, batch, steps))

    assert len(preds) == steps
    for i, pred in enumerate(preds):
        expected_time = tuple(t + (i + 1) * timedelta(hours=6) for t in batch.metadata.time)
        assert pred.metadata.time == expected_time
        assert pred.metadata.rollout_step == i + 1

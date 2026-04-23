"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the AuroraV1p5 forward pass.
"""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch

from ._helpers import _ATMOS_VARS, _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora.insolation import insolation


def test_forward_produces_all_vars():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))

    # Model should predict all `surf_vars` including output-only ones.
    for v in _SURF_VARS:
        assert v in pred.surf_vars, f"Missing surface variable: {v}"
    for v in _ATMOS_VARS:
        assert v in pred.atmos_vars, f"Missing atmospheric variable: {v}"


def test_forward_advances_time():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))

    expected_time = tuple(t + timedelta(hours=6) for t in batch.metadata.time)
    assert pred.metadata.time == expected_time
    assert pred.metadata.rollout_step == 1


def test_variable_lead_time_changes_output_time():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )

    with torch.inference_mode():
        pred3 = model.forward(batch, lead_times=torch.tensor([3.0]))
        pred6 = model.forward(batch, lead_times=torch.tensor([6.0]))

    expected3 = tuple(t + timedelta(hours=3) for t in batch.metadata.time)
    expected6 = tuple(t + timedelta(hours=6) for t in batch.metadata.time)
    assert pred3.metadata.time == expected3
    assert pred6.metadata.time == expected6


def test_missing_lead_times_raises():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    with pytest.raises(ValueError, match="lead_times"):
        model.forward(batch)


def test_insolation_is_recomputed():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )

    with torch.inference_mode(), patch("aurora.model.aurora.insolation", wraps=insolation) as mock:
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))

    mock.assert_called()
    assert torch.isfinite(pred.surf_vars["insolation"]).all()


def test_log_transformed_vars_are_nonnegative():
    model = _make_small_v1p5()
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    # Make the input positive for log-transformed vars.
    for k in batch.surf_vars:
        if k.startswith("scaled_"):
            batch.surf_vars[k] = batch.surf_vars[k].abs()

    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))

    # `log_unscale(x) = eps * (exp(x) - 1)` with `eps = 1e-3`, so the theoretical minimum is `-eps`\
    # (when x -> -inf). Allow that margin.
    for k in pred.surf_vars:
        if k.startswith("scaled_"):
            message = f"Log-transformed var `{k}` has unexpected negative values."
            assert (pred.surf_vars[k] >= -1e-3).all(), message

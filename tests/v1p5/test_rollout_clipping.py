"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for rollout input clipping.
"""

import torch

from ._helpers import BATCH, HISTORY, H, W, _make_batch, _make_small_v1p5


def test_clipping_applied():
    model_clipped = _make_small_v1p5(
        rollout_input_clipping={"2t": {"min": -10.0, "max": 10.0}},
    )
    pred = _make_batch(lead_times=torch.tensor([6.0]))
    pred.surf_vars["2t"] = torch.full((BATCH, HISTORY, H, W), 100.0)
    clipped = model_clipped.apply_rollout_input_clipping(pred)
    assert clipped.surf_vars["2t"].max() <= 10.0


def test_no_clipping_when_none():
    model = _make_small_v1p5(rollout_input_clipping=None)
    pred = _make_batch(lead_times=torch.tensor([6.0]))
    original_val = pred.surf_vars["2t"].clone()
    result = model.apply_rollout_input_clipping(pred)
    torch.testing.assert_close(result.surf_vars["2t"], original_val)

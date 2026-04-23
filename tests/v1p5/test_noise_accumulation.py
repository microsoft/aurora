"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for noise accumulation in stochastic V1.5 models.
"""

import torch

from ._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora import rollout


def test_reset_noise_clears_cache():
    model = _make_small_v1p5(stochastic=True)
    model.backbone._noise_cache.append(torch.randn(2, 4))
    assert len(model.backbone._noise_cache) == 1
    model.reset_noise()
    assert len(model.backbone._noise_cache) == 0


def test_set_noise_accumulation_configures():
    model = _make_small_v1p5(stochastic=True)
    model.set_noise_accumulation(n=5)
    assert model.backbone._accumulate_noise is True
    assert model.backbone._noise_cache_size == 5
    assert len(model.backbone._noise_cache) == 0


def test_set_noise_accumulation_disable():
    model = _make_small_v1p5(stochastic=True)
    model.set_noise_accumulation(n=5)
    assert model.backbone._accumulate_noise is True
    model.set_noise_accumulation(n=0)
    assert model.backbone._accumulate_noise is False


def test_stochastic_forward_runs():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )
    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.tensor([6.0]))
    for v in _SURF_VARS:
        assert v in pred.surf_vars


def test_noise_accumulation_in_rollout():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    batch = _make_batch(
        surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
    )

    fine_lead_times = [3.0, 6.0]
    model.set_noise_accumulation(n=2)
    # Forward the model manually to populate the noise cache.
    with torch.inference_mode():
        for lt in fine_lead_times:
            _ = model.forward(batch, lead_times=torch.tensor([lt]))
    assert model.backbone._noise_cache_size == 2

    # Forward with rollout to check noise accumulation is enabled and then properly disabled
    # after.
    with torch.inference_mode():
        preds = list(
            rollout(
                model,
                batch,
                steps=1,
                fine_lead_times=fine_lead_times,
                use_noise_accumulation=True,
            )
        )
    assert len(preds) == 2
    assert model.backbone._noise_cache_size == 0

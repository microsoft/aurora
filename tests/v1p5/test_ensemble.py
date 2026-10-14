"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for ensemble rollout and tiling utility functions.
"""

import itertools

import pytest
import torch

from ._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, BATCH, _make_batch, _make_small_v1p5
from aurora import Aurora, rollout_ensemble, split_batch, tile_batch
from aurora.model.film import AdaptiveLayerNorm


def _unzero_adaptive_layer_norms(model: Aurora, std: float = 0.1) -> None:
    """Nudge every `AdaptiveLayerNorm`'s modulation away from its zero initialisation.

    A freshly constructed model is exactly insensitive to its conditioning signal, and hence to
    the injected noise, because `AdaptiveLayerNorm.ln_modulation` is zero-initialised.
    """
    for m in model.modules():
        if isinstance(m, AdaptiveLayerNorm):
            with torch.no_grad():
                m.ln_modulation[-1].weight.normal_(std=std)
                m.ln_modulation[-1].bias.normal_(std=std)


def test_tile_and_split_batch_roundtrip():
    b, n = 2, 3
    batch = _make_batch(batch_size=b)

    tiled = tile_batch(batch, n)
    assert len(tiled.metadata.time) == n * b
    for v in (*tiled.surf_vars.values(), *tiled.atmos_vars.values()):
        assert v.shape[0] == n * b

    members = split_batch(tiled, n)
    assert len(members) == n
    for member in members:
        assert member.metadata.time == batch.metadata.time
        for k, v in member.surf_vars.items():
            torch.testing.assert_close(v, batch.surf_vars[k])
        for k, v in member.atmos_vars.items():
            torch.testing.assert_close(v, batch.atmos_vars[k])

    # `batch` has batch size `b`, which is not divisible by `n`.
    with pytest.raises(ValueError):
        split_batch(batch, n)


@pytest.mark.parametrize("stochastic", [True, False])
def test_forward_ensemble(stochastic: bool):
    torch.manual_seed(0)
    model = _make_small_v1p5(stochastic=stochastic)
    _unzero_adaptive_layer_norms(model)  # Otherwise, the noise has no effect on the output.
    model.eval()
    surf_vars = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)
    batch = _make_batch(surf_vars=surf_vars)
    n = 3
    batch = tile_batch(batch, n)
    b = next(iter(batch.surf_vars.values())).shape[0]

    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.full((b,), 6.0))
    members = split_batch(pred, n)

    if stochastic:
        # Every member receives independent noise, so members must differ.
        for member1, member2 in itertools.combinations(members, 2):
            assert (member1.surf_vars["2t"] - member2.surf_vars["2t"]).abs().mean() > 1e-2
    else:
        # Check that all are equal.
        for m in range(1, n):
            torch.testing.assert_close(
                members[0].surf_vars["2t"], members[m].surf_vars["2t"], atol=1e-3, rtol=1e-3
            )


def test_rollout_ensemble():
    num_ensemble_members = 3
    torch.manual_seed(0)
    model = _make_small_v1p5(stochastic=True)
    _unzero_adaptive_layer_norms(model)  # Otherwise, the noise has no effect on the output.
    model.eval()
    batch = _make_batch(surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF))
    steps = 2

    with torch.inference_mode():
        preds = list(rollout_ensemble(model, batch, steps, num_ensemble_members))

    assert len(preds) == steps
    for members in preds:
        assert len(members) == num_ensemble_members
        for member in members:
            for v in member.surf_vars.values():
                assert v.shape[0] == BATCH
        # Every member receives independent noise, so members must differ.
        for member1, member2 in itertools.combinations(members, 2):
            assert (member1.surf_vars["2t"] - member2.surf_vars["2t"]).abs().mean() > 1e-2


@pytest.mark.parametrize("stochastic, num_ensemble_members", [(True, 0), (False, 3)])
def test_rollout_ensemble_raises(stochastic: bool, num_ensemble_members: int):
    model = _make_small_v1p5(stochastic=stochastic)
    with pytest.raises(ValueError):
        next(rollout_ensemble(model, _make_batch(), 2, num_ensemble_members))

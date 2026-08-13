"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for internal ensemble members (`num_ensemble_members`).
"""

import warnings
from datetime import datetime

import pytest
import torch

from ._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora import Aurora, Batch, Metadata, rollout
from aurora.batch import _split_batch, _tile_batch
from aurora.model.film import AdaptiveLayerNorm


def _unzero_adaptive_layer_norms(model: Aurora, std: float = 0.1) -> None:
    """Nudge every `AdaptiveLayerNorm`'s modulation away from its zero initialisation.

    At construction, `AdaptiveLayerNorm.ln_modulation` is exactly zero-initialised (the
    `adaLN-Zero` trick), which makes a freshly-built, untrained model exactly insensitive to its
    conditioning signal `c` -- which is what carries the ensemble noise. Without this, no output
    difference a test observes between ensemble members can be attributed to noise, since noise
    provably has zero effect on such a model.
    """
    for m in model.modules():
        if isinstance(m, AdaptiveLayerNorm):
            with torch.no_grad():
                m.ln_modulation[-1].weight.normal_(std=std)
                m.ln_modulation[-1].bias.normal_(std=std)


def _make_ensemble_test_batch(b: int = 2) -> Batch:
    """A small batch with a configurable batch size `b`, used to test tiling/splitting."""
    h, w = 8, 8
    return Batch(
        surf_vars={"2t": torch.randn(b, 2, h, w)},
        static_vars={"lsm": torch.randn(h, w)},
        atmos_vars={"z": torch.randn(b, 2, 2, h, w)},
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=tuple(datetime(2023, 6, 15, i, 0) for i in range(b)),
            atmos_levels=(500, 850),
        ),
    )


def test_tile_and_split_batch_roundtrip():
    b, n = 2, 3
    batch = _make_ensemble_test_batch(b)

    tiled = _tile_batch(batch, n)

    v = tiled.surf_vars["2t"]
    assert v.shape[0] == n * b
    for m in range(n):
        torch.testing.assert_close(v[m * b : (m + 1) * b], batch.surf_vars["2t"])

    v = tiled.atmos_vars["z"]
    assert v.shape[0] == n * b
    for m in range(n):
        torch.testing.assert_close(v[m * b : (m + 1) * b], batch.atmos_vars["z"])

    assert len(tiled.metadata.time) == n * b
    for m in range(n):
        assert tiled.metadata.time[m * b : (m + 1) * b] == batch.metadata.time

    # Static variables have no batch dimension and are untouched.
    torch.testing.assert_close(tiled.static_vars["lsm"], batch.static_vars["lsm"])

    # Splitting undoes the tiling: every member is identical to the original, standard-shaped
    # batch (tiling itself introduces no randomness).
    members = _split_batch(tiled, n)
    assert len(members) == n
    for member in members:
        torch.testing.assert_close(member.surf_vars["2t"], batch.surf_vars["2t"])
        torch.testing.assert_close(member.atmos_vars["z"], batch.atmos_vars["z"])
        assert member.metadata.time == batch.metadata.time


def test_num_ensemble_members_must_be_positive():
    with pytest.raises(ValueError, match="num_ensemble_members"):
        _make_small_v1p5(num_ensemble_members=0)


def test_num_ensemble_members_warns_without_stochastic():
    with pytest.warns(UserWarning, match="stochastic"):
        _make_small_v1p5(num_ensemble_members=2, stochastic=False)


def test_num_ensemble_members_no_warning_with_stochastic():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _make_small_v1p5(num_ensemble_members=2, stochastic=True)


def test_forward_returns_single_batch_when_num_ensemble_members_one():
    model = _make_small_v1p5()
    model.eval()
    surf_vars = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)
    batch = _make_batch(surf_vars=surf_vars)

    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.full((1,), 6.0))

    assert isinstance(pred, Batch)


def test_forward_returns_list_of_standard_shaped_batches():
    n = 3
    model = _make_small_v1p5(stochastic=True, num_ensemble_members=n)
    model.eval()
    surf_vars = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)
    batch = _make_batch(surf_vars=surf_vars)
    b = next(iter(batch.surf_vars.values())).shape[0]

    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.full((b,), 6.0))

    assert isinstance(pred, list)
    assert len(pred) == n
    for member in pred:
        assert isinstance(member, Batch)
        for v in member.surf_vars.values():
            assert v.shape[0] == b
        for v in member.static_vars.values():
            # Static variables have no batch dimension.
            assert v.dim() == 2


def test_forward_ensemble_members_differ_when_stochastic():
    n = 3
    torch.manual_seed(0)
    model = _make_small_v1p5(stochastic=True, num_ensemble_members=n)
    # Un-zero the modulation so noise has a real, appreciable effect (see helper docstring);
    # otherwise this test cannot distinguish genuine noise sensitivity from incidental
    # floating-point batching noise (see `test_forward_ensemble_members_identical_without_stochastic`).
    _unzero_adaptive_layer_norms(model)
    model.eval()
    surf_vars = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)
    batch = _make_batch(surf_vars=surf_vars)
    b = next(iter(batch.surf_vars.values())).shape[0]

    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.full((b,), 6.0))

    # Threshold well above the ~1e-3 floating-point batching floor established in
    # `test_forward_ensemble_members_identical_without_stochastic`, so a pass here can only be
    # explained by the injected noise actually differing per member, not incidental rounding.
    for i in range(n):
        for j in range(i + 1, n):
            diff = (pred[i].surf_vars["2t"] - pred[j].surf_vars["2t"]).abs().max()
            assert diff > 1e-2


def test_forward_ensemble_members_identical_without_stochastic():
    n = 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _make_small_v1p5(stochastic=False, num_ensemble_members=n)
    model.eval()
    surf_vars = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)
    batch = _make_batch(surf_vars=surf_vars)
    b = next(iter(batch.surf_vars.values())).shape[0]

    with torch.inference_mode():
        pred = model.forward(batch, lead_times=torch.full((b,), 6.0))

    for m in range(1, n):
        # Loose tolerance: floating-point ops (e.g. batched matmul/softmax reductions) are not
        # strictly invariant to how many other (tiled) rows share the batch, so bitwise equality
        # isn't guaranteed even though the members are mathematically identical computations.
        torch.testing.assert_close(
            pred[0].surf_vars["2t"], pred[m].surf_vars["2t"], atol=1e-3, rtol=1e-3
        )


def test_rollout_yields_list_of_standard_shaped_batches_across_steps():
    n = 2
    model = _make_small_v1p5(stochastic=True, num_ensemble_members=n)
    model.eval()
    surf_vars = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)
    batch = _make_batch(surf_vars=surf_vars)
    b = next(iter(batch.surf_vars.values())).shape[0]

    with torch.inference_mode():
        preds = list(rollout(model, batch, steps=3))

    assert len(preds) == 3
    for step_pred in preds:
        assert isinstance(step_pred, list)
        assert len(step_pred) == n
        for member in step_pred:
            for v in member.surf_vars.values():
                assert v.shape[0] == b

    # The model's ensemble configuration is restored after the roll-out completes.
    assert model.num_ensemble_members == n


def test_rollout_restores_num_ensemble_members_on_early_close():
    n = 2
    model = _make_small_v1p5(stochastic=True, num_ensemble_members=n)
    model.eval()
    surf_vars = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)
    batch = _make_batch(surf_vars=surf_vars)

    with torch.inference_mode():
        gen = rollout(model, batch, steps=5)
        next(gen)
        gen.close()

    assert model.num_ensemble_members == n


def test_rollout_num_ensemble_members_one_is_unaffected():
    model = _make_small_v1p5()
    model.eval()
    surf_vars = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)
    batch = _make_batch(surf_vars=surf_vars)
    b = next(iter(batch.surf_vars.values())).shape[0]

    with torch.inference_mode():
        preds = list(rollout(model, batch, steps=2))

    for pred in preds:
        assert isinstance(pred, Batch)
        for v in pred.surf_vars.values():
            assert v.shape[0] == b
    assert model.num_ensemble_members == 1

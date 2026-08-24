"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the `generator` argument of `Aurora.forward`, which controls the noise injection of
stochastic Aurora 1.5 models (issue #191).
"""

import warnings

import pytest
import torch

from ._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora import rollout

_INPUT_SURF_VARS = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)


def _record_noise(model, n_forwards=3, generator=None, batch_size=1):
    """Run `n_forwards` identical forward passes and return the noise samples drawn."""
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS, batch_size=batch_size)
    lead_times = torch.full((batch_size,), 6.0)
    recorded = []
    original = model.backbone._sample_noise

    def recording_sample_noise(shape, device, dtype, generator=None):
        noise = original(shape, device, dtype, generator)
        recorded.append(noise.clone())
        return noise

    model.backbone._sample_noise = recording_sample_noise
    try:
        with torch.inference_mode():
            for _ in range(n_forwards):
                model.forward(batch, lead_times=lead_times, generator=generator)
    finally:
        model.backbone._sample_noise = original
    return recorded


def _seqs_equal(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _member_seq(recorded, i):
    return [noise[i] for noise in recorded]


def test_single_generator_reseed_reproduces_noise():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    generator = torch.Generator().manual_seed(42)
    first = _record_noise(model, generator=generator)
    generator.manual_seed(42)
    second = _record_noise(model, generator=generator)
    assert _seqs_equal(first, second)


def test_single_generator_advances_without_reseed():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    generator = torch.Generator().manual_seed(42)
    first = _record_noise(model, generator=generator)
    second = _record_noise(model, generator=generator)
    assert not _seqs_equal(first, second)


def test_same_seed_fresh_instance_reproduces_noise():
    # The core requirement of issue #191: a given ensemble member reproduces the same noise
    # sequence across model runs, here even across fresh model instances.
    model_a = _make_small_v1p5(stochastic=True)
    model_b = _make_small_v1p5(stochastic=True)
    model_a.eval()
    model_b.eval()
    generator_a = torch.Generator().manual_seed(42)
    generator_b = torch.Generator().manual_seed(42)
    assert _seqs_equal(
        _record_noise(model_a, generator=generator_a),
        _record_noise(model_b, generator=generator_b),
    )


def test_single_generator_independent_of_global_rng():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    generator = torch.Generator().manual_seed(42)
    torch.manual_seed(0)
    first = _record_noise(model, generator=generator)
    generator.manual_seed(42)
    torch.manual_seed(999)
    second = _record_noise(model, generator=generator)
    assert _seqs_equal(first, second)


def test_generator_none_preserves_global_rng_semantics():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    torch.manual_seed(0)
    first = _record_noise(model)
    torch.manual_seed(0)
    second = _record_noise(model)
    torch.manual_seed(999)
    third = _record_noise(model)
    assert _seqs_equal(first, second)
    assert not _seqs_equal(first, third)


def test_tuple_member_noise_independent_of_batch_composition():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    generators = tuple(torch.Generator().manual_seed(seed) for seed in (1, 2, 3))
    joint = _record_noise(model, generator=generators, batch_size=3)
    alone = _record_noise(model, generator=(torch.Generator().manual_seed(3),), batch_size=1)
    assert _seqs_equal(_member_seq(joint, 2), _member_seq(alone, 0))


def test_tuple_and_single_generator_are_not_interchangeable():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    single = _record_noise(model, generator=torch.Generator().manual_seed(42), batch_size=2)
    generators = (torch.Generator().manual_seed(42), torch.Generator().manual_seed(42))
    per_member = _record_noise(model, generator=generators, batch_size=2)
    # Identically seeded per-member generators give every member the same noise, whereas a single
    # generator drives one stream across the whole batch.
    assert _seqs_equal(_member_seq(per_member, 0), _member_seq(per_member, 1))
    assert not _seqs_equal(_member_seq(single, 0), _member_seq(single, 1))
    assert not _seqs_equal(_member_seq(single, 1), _member_seq(per_member, 1))


def test_tuple_none_member_independent_of_other_generators():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    torch.manual_seed(123)
    first = _record_noise(model, generator=(torch.Generator().manual_seed(42), None), batch_size=2)
    torch.manual_seed(123)
    second = _record_noise(model, generator=(torch.Generator().manual_seed(7), None), batch_size=2)
    # The seeded member changes with its seed, but the `None` member draws from the global RNG
    # and must not be affected by the other member's generator.
    assert not _seqs_equal(_member_seq(first, 0), _member_seq(second, 0))
    assert _seqs_equal(_member_seq(first, 1), _member_seq(second, 1))


def test_tuple_length_mismatch_raises_without_consuming_rng():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    generators = tuple(torch.Generator().manual_seed(seed) for seed in (1, 2, 3))
    states = [g.get_state().clone() for g in generators]
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS, batch_size=2)
    with torch.inference_mode(), pytest.raises(ValueError, match="Expected 2 generators"):
        model.forward(batch, lead_times=torch.full((2,), 6.0), generator=generators)
    for state, g in zip(states, generators):
        assert torch.equal(state, g.get_state())


def test_accumulation_reproduces_after_reseed_and_reset_noise():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    model.set_noise_accumulation(n=2)
    generator = torch.Generator().manual_seed(42)
    first = _record_noise(model, generator=generator)
    # Reproducing a run requires *both* re-seeding the generator and flushing the noise cache.
    generator.manual_seed(42)
    model.reset_noise()
    second = _record_noise(model, generator=generator)
    # Re-seeding alone is not enough: leftover cached noise changes how many samples are drawn.
    generator.manual_seed(42)
    third = _record_noise(model, generator=generator)
    model.set_noise_accumulation(n=0)
    assert _seqs_equal(first, second)
    assert not _seqs_equal(first, third)


def test_non_stochastic_model_warns_once_and_ignores_generator():
    model = _make_small_v1p5()
    model.eval()
    generator = torch.Generator().manual_seed(42)
    state = generator.get_state().clone()
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS)
    lead_times = torch.tensor([6.0])
    with torch.inference_mode():
        with pytest.warns(UserWarning, match="`generator` is ignored"):
            model.forward(batch, lead_times=lead_times, generator=generator)
        # The warning is only emitted on the first offending forward call.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.forward(batch, lead_times=lead_times, generator=generator)
        assert not any("`generator` is ignored" in str(w.message) for w in caught)
    assert torch.equal(state, generator.get_state())


def test_rollout_passes_generator_through():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    generator = torch.Generator().manual_seed(42)

    def record_rollout():
        batch = _make_batch(surf_vars=_INPUT_SURF_VARS)
        recorded = []
        original = model.backbone._sample_noise

        def recording_sample_noise(shape, device, dtype, generator=None):
            noise = original(shape, device, dtype, generator)
            recorded.append(noise.clone())
            return noise

        model.backbone._sample_noise = recording_sample_noise
        try:
            with torch.inference_mode():
                for _ in rollout(model, batch, steps=2, generator=generator):
                    pass
        finally:
            model.backbone._sample_noise = original
        return recorded

    first = record_rollout()
    generator.manual_seed(42)
    model.reset_noise()
    second = record_rollout()
    assert len(first) == 2
    assert _seqs_equal(first, second)


def test_dtype_change_invalidates_noise_cache():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    model.set_noise_accumulation(n=2)
    _record_noise(model, n_forwards=1)
    model.double()
    with pytest.warns(UserWarning, match="clearing noise cache"):
        _record_noise(model, n_forwards=1)
    model.set_noise_accumulation(n=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_indexless_cuda_generator_accepted_on_cuda_model():
    model = _make_small_v1p5(stochastic=True).cuda()
    model.eval()
    # `torch.Generator(device="cuda")` reports device `cuda` without an index, which must be
    # treated as compatible with the model's `cuda:0`, like PyTorch does.
    generator = (torch.Generator(device="cuda").manual_seed(42),)
    recorded = _record_noise(model, n_forwards=1, generator=generator)
    assert recorded[0].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_device_mismatch_raises_without_consuming_rng():
    model = _make_small_v1p5(stochastic=True).cuda()
    model.eval()
    generators = (torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    states = [g.get_state().clone() for g in generators]
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS, batch_size=2)
    with torch.inference_mode(), pytest.raises(ValueError, match="on device"):
        model.forward(batch, lead_times=torch.full((2,), 6.0), generator=generators)
    for state, g in zip(states, generators):
        assert torch.equal(state, g.get_state())

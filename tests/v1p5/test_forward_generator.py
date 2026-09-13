"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the `generator` argument of `Aurora.forward`.
"""

import pytest
import torch

from ._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora import rollout

_INPUT_SURF_VARS = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)


@pytest.fixture
def model():
    model = _make_small_v1p5(stochastic=True)
    model.eval()
    return model


def _record_noise(model, run):
    """Call `run` and return the noise samples drawn."""
    recorded = []
    sample_noise = model.backbone._sample_noise

    def record(*args):
        recorded.append(sample_noise(*args))
        return recorded[-1]

    model.backbone._sample_noise = record
    with torch.inference_mode():
        run()
    model.backbone._sample_noise = sample_noise
    return recorded


def _forward_noise(model, generator, batch_size=1, n_forwards=3):
    """Return the noise samples drawn by `n_forwards` forward passes."""
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS, batch_size=batch_size)
    lead_times = torch.full((batch_size,), 6.0)
    return _record_noise(
        model,
        lambda: [
            model.forward(batch, lead_times=lead_times, generator=generator)
            for _ in range(n_forwards)
        ],
    )


def _equal(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _member(recorded, i):
    return [noise[i] for noise in recorded]


@pytest.mark.parametrize("n", [0, 2])
def test_single_generator(model, n):
    model.set_noise_accumulation(n)
    generator = torch.Generator().manual_seed(42)
    torch.manual_seed(0)
    first = _forward_noise(model, generator)
    # Re-seeding the generator and flushing the cache reproduces the noise, regardless of the
    # global RNG.
    generator.manual_seed(42)
    model.reset_noise()
    torch.manual_seed(1)
    assert _equal(first, _forward_noise(model, generator))
    # Without re-seeding, the generator keeps advancing.
    assert not _equal(first, _forward_noise(model, generator))


def test_tuple_of_generators(model):
    def forward_noise(seeds, batch_size):
        torch.manual_seed(0)
        generators = tuple(None if s is None else torch.Generator().manual_seed(s) for s in seeds)
        return _forward_noise(model, generators, batch_size=batch_size)

    first = forward_noise((1, None, 3), batch_size=3)
    second = forward_noise((2, None, 3), batch_size=3)
    alone = forward_noise((3,), batch_size=1)
    # The noise of a member depends only on its own generator, ...
    assert not _equal(_member(first, 0), _member(second, 0))
    assert _equal(_member(first, 2), _member(second, 2))
    assert _equal(_member(first, 2), _member(alone, 0))
    # ... and a `None` entry uses the global RNG.
    assert _equal(_member(first, 1), _member(second, 1))


def test_tuple_length_mismatch(model):
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS, batch_size=2)
    with torch.inference_mode(), pytest.raises(ValueError, match="one generator per batch element"):
        model.forward(batch, lead_times=torch.full((2,), 6.0), generator=(torch.Generator(),))


def test_rollout(model):
    generator = torch.Generator().manual_seed(42)
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS)

    def run():
        list(rollout(model, batch, steps=2, generator=generator))

    first = _record_noise(model, run)
    generator.manual_seed(42)
    assert len(first) == 2
    assert _equal(first, _record_noise(model, run))

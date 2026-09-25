"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the `generator` argument of `Aurora.forward` and `rollout`.
"""

from typing import Any, Callable, Sequence
from unittest.mock import patch

import pytest
import torch

from ._helpers import _OUTPUT_ONLY_SURF, _SURF_VARS, _make_batch, _make_small_v1p5
from aurora import AuroraV1p5, rollout
from aurora.model.swin3d import NoiseGenerator

_INPUT_SURF_VARS = tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)


@pytest.fixture
def model() -> AuroraV1p5:
    return _make_small_v1p5(stochastic=True).eval()


def _record_noise(model: AuroraV1p5, run: Callable[[], Any]) -> list[torch.Tensor]:
    """Call `run` and return the noise samples drawn."""
    recorded: list[torch.Tensor] = []
    sample_noise = model.backbone._sample_noise

    def record(*args: Any) -> torch.Tensor:
        recorded.append(sample_noise(*args))
        return recorded[-1]

    with torch.inference_mode(), patch.object(model.backbone, "_sample_noise", record):
        run()
    return recorded


def _forward_noise(
    model: AuroraV1p5, generator: NoiseGenerator, batch_size: int = 1, n_forwards: int = 3
) -> list[torch.Tensor]:
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


def _equal_sequences_of_tensors(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _member(recorded: Sequence[torch.Tensor], i: int) -> list[torch.Tensor]:
    return [noise[i] for noise in recorded]


@pytest.mark.parametrize("n", [0, 2])
def test_single_generator(model: AuroraV1p5, n: int) -> None:
    model.set_noise_accumulation(n)
    generator = torch.Generator().manual_seed(42)
    torch.manual_seed(0)
    first = _forward_noise(model, generator)
    # Re-seeding the generator and flushing the cache reproduces the noise, regardless of the
    # global RNG.
    generator.manual_seed(42)
    model.reset_noise()
    torch.manual_seed(1)
    assert _equal_sequences_of_tensors(first, _forward_noise(model, generator))
    # Without re-seeding, the generator keeps advancing.
    model.reset_noise()
    assert not _equal_sequences_of_tensors(first, _forward_noise(model, generator))


def test_tuple_of_generators(model: AuroraV1p5) -> None:
    def forward_noise(seeds: Sequence[int | None], batch_size: int) -> list[torch.Tensor]:
        torch.manual_seed(0)
        generators = tuple(None if s is None else torch.Generator().manual_seed(s) for s in seeds)
        return _forward_noise(model, generators, batch_size=batch_size)

    first = forward_noise((1, None, 3), batch_size=3)
    second = forward_noise((2, None, 3), batch_size=3)
    alone = forward_noise((3,), batch_size=1)
    # The noise of a member depends only on its own generator, ...
    assert not _equal_sequences_of_tensors(_member(first, 0), _member(second, 0))
    assert _equal_sequences_of_tensors(_member(first, 2), _member(second, 2))
    assert _equal_sequences_of_tensors(_member(first, 2), _member(alone, 0))
    # ... and a `None` entry uses the global RNG.
    assert _equal_sequences_of_tensors(_member(first, 1), _member(second, 1))


def test_tuple_length_mismatch(model: AuroraV1p5) -> None:
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS, batch_size=2)
    with torch.inference_mode(), pytest.raises(ValueError, match="one generator per batch element"):
        model.forward(batch, lead_times=torch.full((2,), 6.0), generator=(torch.Generator(),))


@pytest.mark.parametrize("fine_lead_times", [None, [3.0, 6.0]])
def test_rollout(model: AuroraV1p5, fine_lead_times: list[float] | None) -> None:
    generator = torch.Generator().manual_seed(42)
    batch = _make_batch(surf_vars=_INPUT_SURF_VARS)

    def run() -> None:
        list(rollout(model, batch, steps=2, fine_lead_times=fine_lead_times, generator=generator))

    first = _record_noise(model, run)
    # Re-seeding the generator reproduces the noise, ...
    generator.manual_seed(42)
    assert _equal_sequences_of_tensors(first, _record_noise(model, run))
    # ... and without re-seeding, the generator keeps advancing.
    assert not _equal_sequences_of_tensors(first, _record_noise(model, run))

"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import numpy as np
import pytest
import torch

from aurora.model.aurora import AuroraSmallPretrained


@pytest.fixture
def model(request):
    return AuroraSmallPretrained(max_history_size=request.param)


@pytest.fixture
def checkpoint():
    return {
        "encoder.surf_token_embeds.weights.0": torch.rand((2, 1, 2, 4, 4)),
        "encoder.atmos_token_embeds.weights.0": torch.rand((2, 1, 2, 4, 4)),
    }


# Check both history sizes which are divisible by 2 (original shape) and not.
@pytest.mark.parametrize("model", [4, 5], indirect=True)
def test_adapt_checkpoint_max_history(model, checkpoint):
    # Checkpoint starts with history dim., `shape[2]`, equal to 2.
    assert checkpoint["encoder.surf_token_embeds.weights.0"].shape[2] == 2
    model.adapt_checkpoint_max_history_size(checkpoint)

    for name, weight in checkpoint.items():
        assert weight.shape[2] == model.max_history_size
        for j in range(weight.shape[2]):
            if j >= checkpoint[name].shape[2]:
                np.testing.assert_allclose(weight[:, :, j, :, :], 0 * weight[:, :, j, :, :])
            else:
                np.testing.assert_allclose(weight[:, :, j, :, :], checkpoint[name][:, :, j, :, :])


@pytest.mark.parametrize("model", [1], indirect=True)
def test_adapt_checkpoint_max_history_fail(model, checkpoint):
    """Check that an assertion error is thrown when trying to load a larger checkpoint to a
    smaller history size."""
    with pytest.raises(AssertionError):
        model.adapt_checkpoint_max_history_size(checkpoint)


@pytest.mark.parametrize("model", [4], indirect=True)
def test_adapt_checkpoint_max_history_twice(model, checkpoint):
    """Test adapting the checkpoint twice to ensure that the second time should not change the
    weights."""
    model.adapt_checkpoint_max_history_size(checkpoint)
    model.adapt_checkpoint_max_history_size(checkpoint)

    for name, weight in checkpoint.items():
        assert weight.shape[2] == model.max_history_size
        for j in range(weight.shape[2]):
            if j >= checkpoint[name].shape[2]:
                np.testing.assert_allclose(weight[:, :, j, :, :], 0 * weight[:, :, j, :, :])
            else:
                np.testing.assert_allclose(weight[:, :, j, :, :], checkpoint[name][:, :, j, :, :])

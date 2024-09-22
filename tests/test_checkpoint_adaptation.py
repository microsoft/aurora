"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import pytest
import torch

from aurora.model.aurora import AuroraSmall


@pytest.fixture
def model(request):
    return AuroraSmall(max_history_size=request.param)


@pytest.fixture
def checkpoint():
    return {
        "encoder.surf_token_embeds.weights.0": torch.rand((2, 1, 2, 4, 4)),
        "encoder.atmos_token_embeds.weights.0": torch.rand((2, 1, 2, 4, 4)),
    }


# check both history sizes which are divisible by 2 (original shape) and not
@pytest.mark.parametrize("model", [4, 5], indirect=True)
def test_adapt_checkpoint_max_history(model, checkpoint):
    # checkpoint starts with history dim, shape[2], as size 2
    assert checkpoint["encoder.surf_token_embeds.weights.0"].shape[2] == 2
    adapted_checkpoint = model.adapt_checkpoint_max_history_size(checkpoint)

    for name, weight in adapted_checkpoint.items():
        assert weight.shape[2] == model.max_history_size
        for j in range(weight.shape[2]):
            if j >= checkpoint[name].shape[2]:
                assert torch.equal(weight[:, :, j, :, :], torch.zeros_like(weight[:, :, j, :, :]))
            else:
                assert torch.equal(
                    weight[:, :, j, :, :],
                    checkpoint[name][:, :, j % checkpoint[name].shape[2], :, :],
                )


# check that assert is thrown when trying to load a larger checkpoint to a smaller history size
@pytest.mark.parametrize("model", [1], indirect=True)
def test_adapt_checkpoint_max_history_fail(model, checkpoint):
    with pytest.raises(AssertionError):
        model.adapt_checkpoint_max_history_size(checkpoint)


# test adapting the checkpoint twice to ensure that the second time should not change the weights
@pytest.mark.parametrize("model", [4], indirect=True)
def test_adapt_checkpoint_max_history_twice(model, checkpoint):
    adapted_checkpoint = model.adapt_checkpoint_max_history_size(checkpoint)
    adapted_checkpoint = model.adapt_checkpoint_max_history_size(adapted_checkpoint)

    for name, weight in adapted_checkpoint.items():
        assert weight.shape[2] == model.max_history_size
        for j in range(weight.shape[2]):
            if j >= checkpoint[name].shape[2]:
                assert torch.equal(weight[:, :, j, :, :], torch.zeros_like(weight[:, :, j, :, :]))
            else:
                assert torch.equal(
                    weight[:, :, j, :, :],
                    checkpoint[name][:, :, j % checkpoint[name].shape[2], :, :],
                )

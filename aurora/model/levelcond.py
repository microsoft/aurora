"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import Callable

import torch
import torch.nn as nn

from aurora.normalisation import level_to_str

__all__ = ["LevelConditioned"]


class LevelConditioned(nn.Module):
    """A module with pressure-level-specific parameters."""

    def __init__(
        self,
        construct_module: Callable[[], nn.Module],
        levels: tuple[int | float, ...],
        levels_dim: int,
    ) -> None:
        """Instantiate.

        Args:
            construct_module (Callable[[], :class:`nn.Module`]): Function that construct a new
                instance of the module that should have pressure-level-specific parameters.
            levels (tuple[int | float, ...]): All possible pressure levels
            levels_dim (int): Dimension of the input that ranges of pressure levels.
        """
        super().__init__()
        self.levels_dim = levels_dim
        self.layers = torch.nn.ParameterDict(
            {level_to_str(level): construct_module() for level in levels}
        )

    def forward(
        self, x: torch.Tensor, *args, levels: tuple[int | float, ...], **kw_args
    ) -> torch.Tensor:
        """Run the module.

        Args:
            x (:class:`torch.Tensor`): Input.
            *args (object): Further arguments.
            levels (tuple[int | float, ...]): Pressure levels in input `x`.
            **kw_args (dict): Further keyword arguments.

        Returns:
            :class:`torch.Tensor`:  Output of applying the module to `x`, where the appropriate
                modules with pressure-level-specific parameters are applied to the appropriate
                elements in `x` along dimension `self.levels_dim`.
        """
        # Resolve `self.levels_dim` to a normal index.
        levels_dim = self.levels_dim
        while levels_dim < 0:
            levels_dim += len(x.shape)

        if x.shape[levels_dim] != len(levels):
            raise ValueError("Incorrect number of pressure levels.")

        def index(i: int) -> tuple[slice | int, ...]:
            return levels_dim * (slice(None),) + (i,)

        return torch.stack(
            [
                self.layers[level_to_str(level)](x[index(i)], *args, **kw_args)
                for i, level in enumerate(levels)
            ],
            dim=levels_dim,
        )

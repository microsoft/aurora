"""
Copyright (c) Microsoft Corporation. Licensed under the MIT license.
"""

import math
from typing import Literal

import torch
from torch import nn

LoraMode = Literal["single", "all"]


class LoRA(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert r > 0, "r must be non-negative"
        self.lora_alpha = alpha
        self.r = r

        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        self.scaling = self.lora_alpha / self.r

        self.reset_parameters()

    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        return (
            self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        ) * self.scaling


class LoRARollout(nn.Module):
    """Module for per rollout step LoRA finetuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        max_steps: int = 40,
        mode: LoraMode = "single",
    ):
        super().__init__()

        self.mode = mode
        self.max_steps = max_steps
        lora_layers = max_steps if mode == "all" else 1
        self.loras = nn.ModuleList(
            [
                LoRA(in_features, out_features, r=r, alpha=alpha, dropout=dropout)
                for _ in range(lora_layers)
            ]
        )

    def forward(self, x: torch.Tensor, step: int):
        assert step >= 0, f"Step must be non-negative, found {step}."

        if step >= self.max_steps:
            return 0

        if self.mode == "single":
            return self.loras[0](x)
        elif self.mode == "all":
            return self.loras[step](x)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

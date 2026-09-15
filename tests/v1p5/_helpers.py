"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Shared helpers and constants for Aurora 1.5 tests.
"""

from datetime import datetime
from typing import Any

import torch

from aurora import AuroraV1p5, Batch, Metadata

# Minimal variable sets for a lightweight Aurora 1.5-like model.
_SURF_VARS = ("2t", "10u", "10v", "msl", "scaled_tp_1h", "insolation")
_STATIC_VARS = ("lsm", "z")
_ATMOS_VARS = ("z", "u", "v", "t", "q")
_OUTPUT_ONLY_SURF = ("scaled_tp_1h",)

H, W = 17, 32  # Spatial dims divisible by patch_size=4 after crop.
N_LEVELS = 4
BATCH = 1
HISTORY = 2


def _make_batch(
    surf_vars: tuple[str, ...] = _SURF_VARS,
    static_vars: tuple[str, ...] = _STATIC_VARS,
    atmos_vars: tuple[str, ...] = _ATMOS_VARS,
    batch_size: int = BATCH,
) -> Batch:
    """Create a minimal synthetic batch."""
    return Batch(
        surf_vars={k: torch.randn(batch_size, HISTORY, H, W) for k in surf_vars},
        static_vars={k: torch.randn(H, W) for k in static_vars},
        atmos_vars={k: torch.randn(batch_size, HISTORY, N_LEVELS, H, W) for k in atmos_vars},
        metadata=Metadata(
            lat=torch.linspace(90, -90, H),
            lon=torch.linspace(0, 360, W + 1)[:-1],
            time=(datetime(2023, 6, 15, 12, 0),) * batch_size,
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def _make_small_v1p5(**overrides: Any) -> AuroraV1p5:
    """Create a small Aurora 1.5 model for fast unit tests."""
    defaults: dict[str, Any] = dict(
        surf_vars=_SURF_VARS,
        static_vars=_STATIC_VARS,
        atmos_vars=_ATMOS_VARS,
        output_only_surf_vars=_OUTPUT_ONLY_SURF,
        encoder_depths=(2, 2),
        encoder_num_heads=(4, 8),
        decoder_depths=(2, 2),
        decoder_num_heads=(8, 4),
        embed_dim=64,
        num_heads=4,
        use_lora=False,
        autocast=False,
        use_fp16_safe_attention=False,
    )
    defaults.update(overrides)
    return AuroraV1p5(**defaults)

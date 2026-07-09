"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for AuroraV1p5 subclasses.
"""

from ._helpers import _ATMOS_VARS, _OUTPUT_ONLY_SURF, _STATIC_VARS, _SURF_VARS
from aurora import AuroraV1p5Ensemble


def test_ensemble_is_stochastic():
    small_kw = dict(
        surf_vars=_SURF_VARS,
        static_vars=_STATIC_VARS,
        atmos_vars=_ATMOS_VARS,
        output_only_surf_vars=_OUTPUT_ONLY_SURF,
        encoder_depths=(2, 2),
        encoder_num_heads=(4, 8),
        decoder_depths=(2, 2),
        decoder_num_heads=(8, 4),
        embed_dim=128,
        num_heads=4,
        use_lora=False,
        autocast=False,
        use_fp16_safe_attention=False,
    )
    model = AuroraV1p5Ensemble(**small_kw)
    assert model.backbone.stochastic is True

"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for Aurora V1.5 features: insolation, log scaling, fp16-safe attention, variable lead-time
models, rollout sub-stepping, and noise accumulation.
"""

import dataclasses
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from aurora import (
    AuroraV1p5,
    AuroraV1p5Ensemble,
    Batch,
    Metadata,
    insolation,
    rollout,
)
from aurora.model.util import fp16_safe_scaled_dot_product_attention
from aurora.normalisation import log_scale, log_unscale

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal variable sets for a lightweight V1.5-like model.
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
    lead_times: torch.Tensor | None = None,
) -> Batch:
    """Create a minimal synthetic batch."""
    return Batch(
        surf_vars={k: torch.randn(BATCH, HISTORY, H, W) for k in surf_vars},
        static_vars={k: torch.randn(H, W) for k in static_vars},
        atmos_vars={k: torch.randn(BATCH, HISTORY, N_LEVELS, H, W) for k in atmos_vars},
        metadata=Metadata(
            lat=torch.linspace(90, -90, H),
            lon=torch.linspace(0, 360, W + 1)[:-1],
            time=(datetime(2023, 6, 15, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
        lead_times=lead_times,
    )


def _make_small_v1p5(**overrides: Any) -> AuroraV1p5:
    """Create a small V1.5 model for fast unit tests."""
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


# ---------------------------------------------------------------------------
# Insolation tests
# ---------------------------------------------------------------------------


class TestInsolation:
    def test_shape_1d(self):
        dates = [datetime(2023, 6, 15, 12)]
        lat = np.linspace(-90, 90, 37)
        lon = np.linspace(0, 360, 37, endpoint=False)
        sol = insolation(dates, lat, lon)
        assert sol.shape == (1, 37)

    def test_shape_2d_enforce(self):
        dates = [datetime(2023, 6, 15, 12)]
        lat = np.linspace(-90, 90, 37)
        lon = np.linspace(0, 360, 72, endpoint=False)
        sol = insolation(dates, lat, lon, enforce_2d=True)
        assert sol.shape == (1, 37, 72)

    def test_shape_multidates(self):
        dates = [datetime(2023, 1, 1), datetime(2023, 7, 1)]
        lat = np.linspace(-90, 90, 5)
        lon = np.linspace(0, 360, 10, endpoint=False)
        sol = insolation(dates, lat, lon, enforce_2d=True)
        assert sol.shape == (2, 5, 10)

    def test_scaling_factor(self):
        dates = [datetime(2023, 6, 15, 12)]
        lat = np.array([0.0])
        lon = np.array([0.0])
        sol1 = insolation(dates, lat, lon, s0=1.0)
        sol2 = insolation(dates, lat, lon, s0=2.0)
        np.testing.assert_allclose(sol2, 2.0 * sol1)

    def test_clip_zero(self):
        dates = [datetime(2023, 6, 15, 0)]  # Midnight UTC
        lat = np.linspace(-90, 90, 37)
        lon = np.linspace(0, 360, 72, endpoint=False)
        sol = insolation(dates, lat, lon, enforce_2d=True, clip_zero=True)
        assert (sol >= 0.0).all()

    def test_daily_ignores_longitude(self):
        dates = [datetime(2023, 6, 15, 12)]
        lat = np.array([45.0])
        lon1 = np.array([0.0])
        lon2 = np.array([180.0])
        sol1 = insolation(dates, lat, lon1, daily=True)
        sol2 = insolation(dates, lat, lon2, daily=True)
        np.testing.assert_allclose(sol1, sol2)

    def test_dimension_mismatch_raises(self):
        dates = [datetime(2023, 6, 15, 12)]
        lat = np.linspace(-90, 90, 5)
        lon = np.zeros((5, 10))
        with pytest.raises(ValueError, match="same number of dimensions"):
            insolation(dates, lat, lon)

    def test_shape_mismatch_2d_raises(self):
        dates = [datetime(2023, 6, 15, 12)]
        lat = np.zeros((5, 10))
        lon = np.zeros((3, 10))
        with pytest.raises(ValueError, match="(?i)shape mismatch"):
            insolation(dates, lat, lon)


# ---------------------------------------------------------------------------
# Log scale / unscale tests
# ---------------------------------------------------------------------------


class TestLogScaling:
    def test_roundtrip(self):
        x = torch.rand(10) * 5.0  # Positive values
        result = log_unscale(log_scale(x))
        torch.testing.assert_close(result, x, rtol=1e-5, atol=1e-6)

    def test_zero_maps_to_zero(self):
        x = torch.tensor([0.0])
        scaled = log_scale(x)
        torch.testing.assert_close(scaled, torch.tensor([0.0]), atol=1e-7, rtol=0.0)

    def test_monotonic(self):
        x = torch.linspace(0, 10, 100)
        scaled = log_scale(x)
        assert (scaled[1:] > scaled[:-1]).all()

    def test_unscale_of_zero(self):
        # log_scale(0) == 0, so log_unscale(0) should be 0.
        result = log_unscale(torch.tensor([0.0]))
        torch.testing.assert_close(result, torch.tensor([0.0]), atol=1e-7, rtol=0.0)


# ---------------------------------------------------------------------------
# fp16-safe attention tests
# ---------------------------------------------------------------------------


class TestFP16SafeAttention:
    def test_matches_standard_in_float32(self):
        torch.manual_seed(42)
        B, H, L, D = 2, 4, 16, 32
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        ref = F.scaled_dot_product_attention(q, k, v)
        out = fp16_safe_scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_standard_with_mask(self):
        torch.manual_seed(42)
        B, H, L, D = 2, 4, 8, 16
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        mask = torch.zeros(L, L)
        mask[0, L // 2 :] = float("-inf")

        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = fp16_safe_scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_no_inf_in_fp16(self):
        torch.manual_seed(42)
        B, H, L, D = 1, 2, 8, 16
        # Use large values that could cause overflow in fp16.
        q = (torch.randn(B, H, L, D) * 10).half()
        k = (torch.randn(B, H, L, D) * 10).half()
        v = (torch.randn(B, H, L, D)).half()

        out = fp16_safe_scaled_dot_product_attention(q, k, v)
        assert torch.isfinite(out).all()

    def test_custom_scale(self):
        torch.manual_seed(42)
        B, H, L, D = 1, 2, 8, 16
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        out1 = fp16_safe_scaled_dot_product_attention(q, k, v, scale=0.5)
        out2 = fp16_safe_scaled_dot_product_attention(q, k, v, scale=1.0)
        # Different scales should produce different outputs.
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# AuroraV1p5 forward pass tests
# ---------------------------------------------------------------------------


class TestAuroraV1p5Forward:
    def test_forward_produces_all_vars(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            # Exclude output-only vars from input.
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
            lead_times=torch.tensor([6.0]),
        )
        with torch.inference_mode():
            pred = model.forward(batch)

        # Model should predict all `surf_vars` including output-only ones.
        for v in _SURF_VARS:
            assert v in pred.surf_vars, f"Missing surface variable: {v}"
        for v in _ATMOS_VARS:
            assert v in pred.atmos_vars, f"Missing atmospheric variable: {v}"

    def test_forward_advances_time(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
            lead_times=torch.tensor([6.0]),
        )
        with torch.inference_mode():
            pred = model.forward(batch)

        expected_time = tuple(t + timedelta(hours=6) for t in batch.metadata.time)
        assert pred.metadata.time == expected_time
        assert pred.metadata.rollout_step == 1

    def test_variable_lead_time_changes_output_time(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
        )

        with torch.inference_mode():
            batch3 = dataclasses.replace(batch, lead_times=torch.tensor([3.0]))
            pred3 = model.forward(batch3)
            batch6 = dataclasses.replace(batch, lead_times=torch.tensor([6.0]))
            pred6 = model.forward(batch6)

        expected3 = tuple(t + timedelta(hours=3) for t in batch.metadata.time)
        expected6 = tuple(t + timedelta(hours=6) for t in batch.metadata.time)
        assert pred3.metadata.time == expected3
        assert pred6.metadata.time == expected6

    def test_missing_lead_times_raises(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
            lead_times=None,
        )
        with pytest.raises(ValueError, match="lead_times"):
            model.forward(batch)

    def test_insolation_is_recomputed(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
            lead_times=torch.tensor([6.0]),
        )

        with torch.inference_mode():
            pred = model.forward(batch)

        # Insolation should have been recomputed analytically, not merely passed through the
        # network.  Check it contains finite values.
        assert torch.isfinite(pred.surf_vars["insolation"]).all()

    def test_log_scaled_vars_are_nonnegative(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
            lead_times=torch.tensor([6.0]),
        )
        # Make the input positive for log-scaled vars.
        for k in batch.surf_vars:
            if k.startswith("scaled_"):
                batch.surf_vars[k] = batch.surf_vars[k].abs()

        with torch.inference_mode():
            pred = model.forward(batch)

        # `log_unscale(x) = eps * (exp(x) - 1)` with `eps = 1e-3`, so the theoretical minimum
        # is `-eps` (when x -> -inf).  Allow that margin.
        for k in pred.surf_vars:
            if k.startswith("scaled_"):
                message = f"Log-scaled var `{k}` has unexpected negative values."
                assert (pred.surf_vars[k] >= -1e-3).all(), message


# ---------------------------------------------------------------------------
# AuroraV1p5 subclasses
# ---------------------------------------------------------------------------


class TestAuroraV1p5Subclasses:
    def test_ensemble_is_stochastic(self):
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


# ---------------------------------------------------------------------------
# Rollout sub-stepping tests
# ---------------------------------------------------------------------------


class TestRolloutSubStepping:
    def test_fine_lead_times_produces_more_outputs(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
        )

        steps = 2
        fine_lead_times = [3.0, 6.0]
        with torch.inference_mode():
            preds = list(rollout(model, batch, steps, fine_lead_times=fine_lead_times))

        # Should be `steps * len(fine_lead_times)` outputs.
        assert len(preds) == steps * len(fine_lead_times)

    def test_fine_lead_times_correct_output_times(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
        )

        fine_lead_times = [2.0, 4.0, 6.0]
        with torch.inference_mode():
            preds = list(rollout(model, batch, steps=2, fine_lead_times=fine_lead_times))

        base_time = batch.metadata.time[0]
        for i, lt in enumerate(fine_lead_times):
            expected = base_time + timedelta(hours=lt)
            assert preds[i].metadata.time[0] == expected

    def test_fine_lead_times_requires_variable_lead_time(self):
        # Create a plain `Aurora` model (`variable_lead_time=False` by default).
        from aurora import Aurora

        model = Aurora(use_lora=False)
        model.eval()
        batch = _make_batch(
            surf_vars=("2t", "10u", "10v", "msl"),
            static_vars=("lsm", "z"),
            atmos_vars=("z", "u", "v", "t", "q"),
        )
        with pytest.raises(ValueError, match="variable_lead_time"):
            list(rollout(model, batch, steps=1, fine_lead_times=[3.0, 6.0]))

    def test_standard_rollout_still_works(self):
        model = _make_small_v1p5()
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
        )

        steps = 3
        with torch.inference_mode():
            preds = list(rollout(model, batch, steps))

        assert len(preds) == steps
        for i, pred in enumerate(preds):
            expected_time = tuple(t + (i + 1) * timedelta(hours=6) for t in batch.metadata.time)
            assert pred.metadata.time == expected_time
            assert pred.metadata.rollout_step == i + 1


# ---------------------------------------------------------------------------
# Rollout input clipping tests
# ---------------------------------------------------------------------------


class TestRolloutInputClipping:
    def test_clipping_applied(self):
        model = _make_small_v1p5()
        model.eval()

        # Create a prediction-like batch with out-of-range values.
        pred = _make_batch(lead_times=torch.tensor([6.0]))
        # For `AuroraV1p5` defaults, `scaled_tp_1h` won't be in clipping; let's test with a model
        # that has clipping for a known var.
        model_clipped = _make_small_v1p5(
            rollout_input_clipping={"2t": {"min": -10.0, "max": 10.0}},
        )
        pred.surf_vars["2t"] = torch.full((BATCH, HISTORY, H, W), 100.0)
        clipped = model_clipped.apply_rollout_input_clipping(pred)
        assert clipped.surf_vars["2t"].max() <= 10.0

    def test_no_clipping_when_none(self):
        model = _make_small_v1p5(rollout_input_clipping=None)
        pred = _make_batch(lead_times=torch.tensor([6.0]))
        original_val = pred.surf_vars["2t"].clone()
        result = model.apply_rollout_input_clipping(pred)
        torch.testing.assert_close(result.surf_vars["2t"], original_val)


# ---------------------------------------------------------------------------
# Noise accumulation tests
# ---------------------------------------------------------------------------


class TestNoiseAccumulation:
    def test_reset_noise_clears_cache(self):
        model = _make_small_v1p5(stochastic=True)
        model.backbone._noise_cache.append(torch.randn(2, 4))
        assert len(model.backbone._noise_cache) == 1
        model.reset_noise()
        assert len(model.backbone._noise_cache) == 0

    def test_set_noise_accumulation_configures(self):
        model = _make_small_v1p5(stochastic=True)
        model.set_noise_accumulation(True, n=5)
        assert model.backbone._accumulate_noise is True
        assert model.backbone._noise_cache_size == 5
        assert len(model.backbone._noise_cache) == 0

    def test_set_noise_accumulation_disable(self):
        model = _make_small_v1p5(stochastic=True)
        model.set_noise_accumulation(True, n=5)
        assert model.backbone._accumulate_noise is True
        model.set_noise_accumulation(False)
        assert model.backbone._accumulate_noise is False

    def test_stochastic_forward_runs(self):
        model = _make_small_v1p5(stochastic=True)
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
            lead_times=torch.tensor([6.0]),
        )
        with torch.inference_mode():
            pred = model.forward(batch)
        for v in _SURF_VARS:
            assert v in pred.surf_vars

    def test_noise_accumulation_in_rollout(self):
        model = _make_small_v1p5(stochastic=True)
        model.eval()
        batch = _make_batch(
            surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF),
        )

        fine_lead_times = [3.0, 6.0]
        model.set_noise_accumulation(True, n=2)
        # Forward the model manually to populate the noise cache.
        with torch.inference_mode():
            for lt in fine_lead_times:
                _ = model.forward(dataclasses.replace(batch, lead_times=torch.tensor([lt])))
        assert model.backbone._noise_cache_size == 2

        # Forward with rollout to check noise accumulation is enabled and then properly disabled
        # after.
        with torch.inference_mode():
            preds = list(
                rollout(
                    model,
                    batch,
                    steps=1,
                    fine_lead_times=fine_lead_times,
                    use_noise_accumulation=True,
                )
            )
        assert len(preds) == 2
        assert model.backbone._noise_cache_size == 0


# ---------------------------------------------------------------------------
# Batch lead_times plumbing
# ---------------------------------------------------------------------------


class TestBatchLeadTimes:
    def test_normalise_preserves_lead_times(self):
        lt = torch.tensor([6.0])
        batch = _make_batch(lead_times=lt)
        normed = batch.normalise(surf_stats={})
        torch.testing.assert_close(normed.lead_times, lt)

    def test_unnormalise_preserves_lead_times(self):
        lt = torch.tensor([6.0])
        batch = _make_batch(lead_times=lt)
        unnormed = batch.unnormalise(surf_stats={})
        torch.testing.assert_close(unnormed.lead_times, lt)

    def test_crop_preserves_lead_times(self):
        lt = torch.tensor([6.0])
        batch = _make_batch(lead_times=lt)
        cropped = batch.crop(4)
        torch.testing.assert_close(cropped.lead_times, lt)

    def test_to_device_preserves_lead_times(self):
        lt = torch.tensor([6.0])
        batch = _make_batch(lead_times=lt)
        moved = batch.to("cpu")
        torch.testing.assert_close(moved.lead_times, lt)

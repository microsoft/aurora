"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Tests for the fp16-safe scaled dot-product attention implementation.
"""

import torch
import torch.nn.functional as F

from aurora.model.util import fp16_safe_scaled_dot_product_attention


def test_matches_standard_in_float32():
    torch.manual_seed(42)
    B, H, L, D = 2, 4, 16, 32
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    ref = F.scaled_dot_product_attention(q, k, v)
    out = fp16_safe_scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_matches_standard_with_mask():
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


def test_no_inf_in_fp16():
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 8, 16
    # Use large values that could cause overflow in fp16. Large sigma caused by large values in
    # q @ k^T, which could cause inf after softmax.
    q = (torch.randn(B, H, L, D) * 10).half()
    k = (torch.randn(B, H, L, D) * 10).half()
    v = (torch.randn(B, H, L, D)).half()

    out = fp16_safe_scaled_dot_product_attention(q, k, v)
    assert torch.isfinite(out).all()


def test_custom_scale():
    torch.manual_seed(42)
    B, H, L, D = 1, 2, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    out1 = fp16_safe_scaled_dot_product_attention(q, k, v, scale=0.5)
    out2 = fp16_safe_scaled_dot_product_attention(q, k, v, scale=1.0)
    # Different scales should produce different outputs.
    assert not torch.allclose(out1, out2)

# NOTE: written but not hardware-verified — smoke-test before use
"""
test_attention.py — pytest smoke tests for the NumPy and PyTorch attention
implementations.

Run:
    pytest test_attention.py -v

All tests use small, deterministic random inputs and run in under 30 seconds
on any CPU.
"""

import numpy as np
import torch
import pytest

from attention_numpy import (
    scaled_dot_product_attention as np_sdpa,
    multi_head_attention as np_mha,
    make_causal_mask as np_causal_mask,
    softmax as np_softmax,
)
from attention_torch import (
    scaled_dot_product_attention as th_sdpa,
    multi_head_attention as th_mha,
    make_causal_mask as th_causal_mask,
)


ATOL = 1e-4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sdpa_inputs():
    """Small deterministic Q, K, V arrays for single-head attention tests."""
    rng = np.random.default_rng(0)
    B, T, d_k = 2, 6, 16
    Q = rng.standard_normal((B, T, d_k))
    K = rng.standard_normal((B, T, d_k))
    V = rng.standard_normal((B, T, d_k))
    return Q, K, V


@pytest.fixture
def mha_inputs():
    """Small deterministic inputs for multi-head attention tests."""
    rng = np.random.default_rng(1)
    B, T, d_model, n_heads = 2, 8, 32, 4
    x = rng.standard_normal((B, T, d_model))
    WQ = rng.standard_normal((d_model, d_model))
    WK = rng.standard_normal((d_model, d_model))
    WV = rng.standard_normal((d_model, d_model))
    WO = rng.standard_normal((d_model, d_model))
    return x, WQ, WK, WV, WO, n_heads


# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
        out = np_softmax(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(2), atol=1e-6)

    def test_monotone(self):
        """Higher logit should produce higher softmax output."""
        x = np.array([[1.0, 2.0, 3.0]])
        out = np_softmax(x, axis=-1)
        assert out[0, 0] < out[0, 1] < out[0, 2]

    def test_numerical_stability_large_values(self):
        """Softmax should not overflow for large positive inputs."""
        x = np.array([[1000.0, 1001.0, 999.0]])
        out = np_softmax(x, axis=-1)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(1), atol=1e-6)


# ---------------------------------------------------------------------------
# Single-head SDPA — shape and value tests
# ---------------------------------------------------------------------------

class TestSingleHeadSdpa:
    def test_output_shape_numpy(self, sdpa_inputs):
        Q, K, V = sdpa_inputs
        out, weights = np_sdpa(Q, K, V)
        assert out.shape == Q.shape, f"Expected {Q.shape}, got {out.shape}"

    def test_weight_shape_numpy(self, sdpa_inputs):
        Q, K, V = sdpa_inputs
        _, weights = np_sdpa(Q, K, V)
        B, T, _ = Q.shape
        assert weights.shape == (B, T, T)

    def test_weights_sum_to_one_numpy(self, sdpa_inputs):
        Q, K, V = sdpa_inputs
        _, weights = np_sdpa(Q, K, V)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(weights.shape[:-1]),
                                   atol=1e-5)

    def test_output_shape_torch(self, sdpa_inputs):
        Q, K, V = sdpa_inputs
        Q_t = torch.tensor(Q, dtype=torch.float64)
        K_t = torch.tensor(K, dtype=torch.float64)
        V_t = torch.tensor(V, dtype=torch.float64)
        out, _ = th_sdpa(Q_t, K_t, V_t)
        assert tuple(out.shape) == Q.shape

    def test_numpy_torch_agree(self, sdpa_inputs):
        Q, K, V = sdpa_inputs
        out_np, weights_np = np_sdpa(Q, K, V)
        Q_t = torch.tensor(Q, dtype=torch.float64)
        K_t = torch.tensor(K, dtype=torch.float64)
        V_t = torch.tensor(V, dtype=torch.float64)
        out_th, weights_th = th_sdpa(Q_t, K_t, V_t)
        np.testing.assert_allclose(out_np, out_th.numpy(), atol=ATOL)
        np.testing.assert_allclose(weights_np, weights_th.numpy(), atol=ATOL)

    def test_causal_mask_numpy_torch_agree(self, sdpa_inputs):
        Q, K, V = sdpa_inputs
        T = Q.shape[1]
        mask_np = np_causal_mask(T)
        out_np, _ = np_sdpa(Q, K, V, mask=mask_np)

        Q_t = torch.tensor(Q, dtype=torch.float64)
        K_t = torch.tensor(K, dtype=torch.float64)
        V_t = torch.tensor(V, dtype=torch.float64)
        mask_th = th_causal_mask(T).double()
        out_th, _ = th_sdpa(Q_t, K_t, V_t, mask=mask_th)
        np.testing.assert_allclose(out_np, out_th.numpy(), atol=ATOL)


# ---------------------------------------------------------------------------
# Causal mask properties
# ---------------------------------------------------------------------------

class TestCausalMask:
    def test_lower_tri_zero_numpy(self):
        T = 5
        mask = np_causal_mask(T)
        lower = np.tril(mask)
        np.testing.assert_array_equal(lower, np.zeros((T, T)))

    def test_upper_tri_negative_numpy(self):
        T = 5
        mask = np_causal_mask(T)
        upper = np.triu(mask, k=1)
        assert np.all(upper < 0)

    def test_causal_property(self, sdpa_inputs):
        """Position 0 output must be unchanged when future keys are perturbed."""
        Q, K, V = sdpa_inputs
        T = Q.shape[1]
        mask = np_causal_mask(T)

        out_orig, _ = np_sdpa(Q, K, V, mask=mask)

        K_mod = K.copy()
        K_mod[:, 1:, :] += 1000.0   # large perturbation to all future positions
        out_mod, _ = np_sdpa(Q, K_mod, V, mask=mask)

        np.testing.assert_allclose(
            out_orig[:, 0, :], out_mod[:, 0, :], atol=ATOL,
            err_msg="Causal property violated: position 0 output changed when "
                    "future keys were perturbed."
        )


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape_numpy(self, mha_inputs):
        x, WQ, WK, WV, WO, n_heads = mha_inputs
        out = np_mha(x, WQ, WK, WV, WO, n_heads)
        assert out.shape == x.shape

    def test_output_shape_torch(self, mha_inputs):
        x, WQ, WK, WV, WO, n_heads = mha_inputs
        x_t  = torch.tensor(x,  dtype=torch.float64)
        WQ_t = torch.tensor(WQ, dtype=torch.float64)
        WK_t = torch.tensor(WK, dtype=torch.float64)
        WV_t = torch.tensor(WV, dtype=torch.float64)
        WO_t = torch.tensor(WO, dtype=torch.float64)
        out = th_mha(x_t, WQ_t, WK_t, WV_t, WO_t, n_heads)
        assert tuple(out.shape) == x.shape

    def test_numpy_torch_agree(self, mha_inputs):
        x, WQ, WK, WV, WO, n_heads = mha_inputs
        T = x.shape[1]
        mask_np = np_causal_mask(T)
        mask_th = th_causal_mask(T).double()

        out_np = np_mha(x, WQ, WK, WV, WO, n_heads, mask=mask_np)

        x_t  = torch.tensor(x,  dtype=torch.float64)
        WQ_t = torch.tensor(WQ, dtype=torch.float64)
        WK_t = torch.tensor(WK, dtype=torch.float64)
        WV_t = torch.tensor(WV, dtype=torch.float64)
        WO_t = torch.tensor(WO, dtype=torch.float64)
        out_th = th_mha(x_t, WQ_t, WK_t, WV_t, WO_t, n_heads, mask=mask_th)

        np.testing.assert_allclose(out_np, out_th.numpy(), atol=ATOL,
                                   err_msg="Multi-head outputs differ between NumPy and PyTorch.")

    def test_invalid_n_heads_raises(self, mha_inputs):
        x, WQ, WK, WV, WO, _ = mha_inputs
        with pytest.raises(AssertionError):
            np_mha(x, WQ, WK, WV, WO, n_heads=7)   # 32 % 7 != 0


# ---------------------------------------------------------------------------
# Scaling sanity check
# ---------------------------------------------------------------------------

class TestScaling:
    def test_scaling_reduces_score_magnitude(self):
        """Unscaled scores should have higher variance than scaled scores."""
        rng = np.random.default_rng(5)
        B, T, d_k = 1, 10, 128
        Q = rng.standard_normal((B, T, d_k))
        K = rng.standard_normal((B, T, d_k))

        scores_unscaled = (Q @ K.swapaxes(-2, -1))
        scores_scaled   = (Q @ K.swapaxes(-2, -1)) / np.sqrt(d_k)

        assert scores_unscaled.std() > scores_scaled.std(), (
            "Scaling should reduce score standard deviation."
        )

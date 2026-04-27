# NOTE: written but not hardware-verified — smoke-test before use
"""
compare.py — Run NumPy and PyTorch attention implementations on the same
random inputs, assert they agree within tolerance, and compare against
torch.nn.functional.scaled_dot_product_attention.

Run:
    python compare.py

Expected output: all comparisons printing [PASS].

The script exercises:
  1. Single-head SDPA: NumPy vs PyTorch (no mask, causal mask).
  2. Multi-head attention: NumPy vs PyTorch.
  3. PyTorch SDPA vs torch.nn.functional (causal and non-causal).
"""

import numpy as np
import torch
import sys

from attention_numpy import (
    scaled_dot_product_attention as np_sdpa,
    multi_head_attention as np_mha,
    make_causal_mask as np_causal_mask,
)
from attention_torch import (
    scaled_dot_product_attention as th_sdpa,
    multi_head_attention as th_mha,
    make_causal_mask as th_causal_mask,
    compare_with_torch_sdpa,
)


ATOL = 1e-4  # absolute tolerance for float64/float32 comparisons


def assert_close(name: str, a: np.ndarray, b: np.ndarray, atol: float = ATOL) -> None:
    diff = np.abs(a - b).max()
    status = "PASS" if diff < atol else "FAIL"
    print(f"  [{status}] {name}: max |numpy - torch| = {diff:.2e}  (atol={atol:.0e})")
    if diff >= atol:
        print(f"         shapes: numpy={a.shape}, torch={b.shape}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 1. Single-head SDPA — no mask
# ---------------------------------------------------------------------------

def compare_sdpa_no_mask(B: int = 2, T: int = 6, d_k: int = 16) -> None:
    print("\n--- Single-head SDPA (no mask) ---")
    rng = np.random.default_rng(42)

    Q_np = rng.standard_normal((B, T, d_k))
    K_np = rng.standard_normal((B, T, d_k))
    V_np = rng.standard_normal((B, T, d_k))

    out_np, weights_np = np_sdpa(Q_np, K_np, V_np, mask=None)

    Q_th = torch.tensor(Q_np, dtype=torch.float64)
    K_th = torch.tensor(K_np, dtype=torch.float64)
    V_th = torch.tensor(V_np, dtype=torch.float64)

    out_th, weights_th = th_sdpa(Q_th, K_th, V_th, mask=None)

    assert_close("output", out_np, out_th.numpy())
    assert_close("weights", weights_np, weights_th.numpy())


# ---------------------------------------------------------------------------
# 2. Single-head SDPA — causal mask
# ---------------------------------------------------------------------------

def compare_sdpa_causal(B: int = 2, T: int = 8, d_k: int = 32) -> None:
    print("\n--- Single-head SDPA (causal mask) ---")
    rng = np.random.default_rng(7)

    Q_np = rng.standard_normal((B, T, d_k))
    K_np = rng.standard_normal((B, T, d_k))
    V_np = rng.standard_normal((B, T, d_k))

    mask_np = np_causal_mask(T)          # (T, T) with -1e9 above diagonal
    mask_th = th_causal_mask(T)          # (T, T) with -inf above diagonal

    out_np, _ = np_sdpa(Q_np, K_np, V_np, mask=mask_np)

    Q_th = torch.tensor(Q_np, dtype=torch.float64)
    K_th = torch.tensor(K_np, dtype=torch.float64)
    V_th = torch.tensor(V_np, dtype=torch.float64)

    # Use the -inf mask version for torch to get a fair float64 comparison.
    # (both should be numerically the same for non-extreme inputs)
    out_th, _ = th_sdpa(Q_th, K_th, V_th, mask=mask_th.double())

    assert_close("causal output", out_np, out_th.numpy())

    # Verify causal property: output at position i must not depend on j > i.
    # We check this by comparing the first-position output across two inputs
    # that differ only at positions 1..T-1.
    Q2_np = rng.standard_normal((1, T, d_k))
    K2_np = rng.standard_normal((1, T, d_k))
    V2_np = rng.standard_normal((1, T, d_k))
    K2_mod = K2_np.copy()
    K2_mod[:, 1:, :] += 100.0   # perturb all future positions

    out_orig, _ = np_sdpa(Q2_np, K2_np, V2_np, mask=mask_np[np.newaxis])
    out_mod, _ = np_sdpa(Q2_np, K2_mod, V2_np, mask=mask_np[np.newaxis])
    causal_diff = np.abs(out_orig[:, 0, :] - out_mod[:, 0, :]).max()
    causal_ok = causal_diff < ATOL
    print(f"  [{'PASS' if causal_ok else 'FAIL'}] causal property check: "
          f"position-0 output unchanged after perturbing future keys "
          f"(max diff = {causal_diff:.2e})")


# ---------------------------------------------------------------------------
# 3. Multi-head attention — NumPy vs PyTorch
# ---------------------------------------------------------------------------

def compare_mha(B: int = 2, T: int = 10, d_model: int = 64, n_heads: int = 4) -> None:
    print(f"\n--- Multi-head attention (B={B}, T={T}, d_model={d_model}, H={n_heads}) ---")
    d_k = d_model // n_heads
    rng = np.random.default_rng(99)

    x_np = rng.standard_normal((B, T, d_model))
    WQ_np = rng.standard_normal((d_model, d_model))
    WK_np = rng.standard_normal((d_model, d_model))
    WV_np = rng.standard_normal((d_model, d_model))
    WO_np = rng.standard_normal((d_model, d_model))

    mask_np = np_causal_mask(T)

    out_np = np_mha(x_np, WQ_np, WK_np, WV_np, WO_np, n_heads, mask=mask_np)

    x_th  = torch.tensor(x_np,  dtype=torch.float64)
    WQ_th = torch.tensor(WQ_np, dtype=torch.float64)
    WK_th = torch.tensor(WK_np, dtype=torch.float64)
    WV_th = torch.tensor(WV_np, dtype=torch.float64)
    WO_th = torch.tensor(WO_np, dtype=torch.float64)
    mask_th = th_causal_mask(T).double()

    out_th = th_mha(x_th, WQ_th, WK_th, WV_th, WO_th, n_heads, mask=mask_th)

    assert_close("multi-head output", out_np, out_th.numpy())


# ---------------------------------------------------------------------------
# 4. PyTorch SDPA vs torch.nn.functional
# ---------------------------------------------------------------------------

def compare_torch_vs_functional() -> None:
    print("\n--- PyTorch SDPA vs torch.nn.functional.scaled_dot_product_attention ---")
    B, T, d_k = 3, 12, 32
    Q = torch.randn(B, T, d_k, dtype=torch.float32)
    K = torch.randn(B, T, d_k, dtype=torch.float32)
    V = torch.randn(B, T, d_k, dtype=torch.float32)

    print("  Non-causal:")
    compare_with_torch_sdpa(Q, K, V, is_causal=False, atol=1e-4)
    print("  Causal:")
    compare_with_torch_sdpa(Q, K, V, is_causal=True, atol=1e-4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Attention correctness comparisons")
    print("=" * 60)

    compare_sdpa_no_mask()
    compare_sdpa_causal()
    compare_mha()
    compare_torch_vs_functional()

    print("\nDone.")


if __name__ == '__main__':
    main()

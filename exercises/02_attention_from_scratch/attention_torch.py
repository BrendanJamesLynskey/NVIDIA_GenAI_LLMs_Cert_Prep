# NOTE: written but not hardware-verified — smoke-test before use
"""
attention_torch.py — Scaled dot-product attention and multi-head attention
implemented in PyTorch, mirroring attention_numpy.py exactly.

The function signatures and shape conventions are identical to the NumPy
versions so that compare.py can call both and diff the outputs.

At the end of each function, an optional comparison against
torch.nn.functional.scaled_dot_product_attention is included for reference.
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Single-head scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention in PyTorch.

        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Parameters
    ----------
    Q : Tensor, shape (..., T_q, d_k)
    K : Tensor, shape (..., T_k, d_k)
    V : Tensor, shape (..., T_k, d_v)
    mask : Optional[Tensor], shape (..., T_q, T_k) — additive mask, -inf
           for blocked positions.

    Returns
    -------
    output  : Tensor, shape (..., T_q, d_v)
    weights : Tensor, shape (..., T_q, T_k)
    """
    d_k = Q.size(-1)
    scale = d_k ** 0.5

    # Q K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (..., T_q, T_k)

    if mask is not None:
        scores = scores + mask

    weights = torch.softmax(scores, dim=-1)   # (..., T_q, T_k)
    output = torch.matmul(weights, V)          # (..., T_q, d_v)

    return output, weights


def make_causal_mask(T: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Causal (upper-triangular) additive mask of shape (T, T).

    Entries above the diagonal are -inf; others are 0.  Using -inf rather
    than -1e9 is more numerically precise and aligns with PyTorch's own
    convention in torch.nn.Transformer.
    """
    mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
    return mask


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

def multi_head_attention(
    x: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    W_O: torch.Tensor,
    n_heads: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multi-head self-attention in PyTorch.

    Parameters
    ----------
    x   : Tensor, shape (B, T, d_model)
    W_Q : Tensor, shape (d_model, d_model)
    W_K : Tensor, shape (d_model, d_model)
    W_V : Tensor, shape (d_model, d_model)
    W_O : Tensor, shape (d_model, d_model)
    n_heads : int
    mask : Optional[Tensor], shape (T, T) or (B, H, T, T)

    Returns
    -------
    output : Tensor, shape (B, T, d_model)
    """
    B, T, d_model = x.shape
    assert d_model % n_heads == 0, (
        f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
    )
    d_k = d_model // n_heads

    # Full-model projections.
    Q_full = x @ W_Q   # (B, T, d_model)
    K_full = x @ W_K
    V_full = x @ W_V

    # Split into heads: (B, T, d_model) -> (B, H, T, d_k)
    def split_heads(z: torch.Tensor) -> torch.Tensor:
        return z.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, H, T, d_k)

    Q_h = split_heads(Q_full)
    K_h = split_heads(K_full)
    V_h = split_heads(V_full)

    # Broadcast 2-D mask to (1, 1, T, T).
    if mask is not None and mask.dim() == 2:
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
    else:
        mask_expanded = mask

    attn_output, _ = scaled_dot_product_attention(Q_h, K_h, V_h, mask=mask_expanded)
    # attn_output: (B, H, T, d_k)

    # Merge heads: (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, d_model)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, d_model)

    output = attn_output @ W_O  # (B, T, d_model)

    return output


# ---------------------------------------------------------------------------
# Comparison helper: our implementation vs torch.nn.functional
# ---------------------------------------------------------------------------

def compare_with_torch_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    atol: float = 1e-4,
) -> None:
    """
    Run our scaled_dot_product_attention against
    torch.nn.functional.scaled_dot_product_attention and print the max
    absolute difference.

    torch.nn.functional.scaled_dot_product_attention accepts an `is_causal`
    flag directly and builds the mask internally, so we pass is_causal=True
    for a fair comparison when a causal mask is desired.

    Note: F.scaled_dot_product_attention does NOT return attention weights
    (by design, to allow FlashAttention-style fused kernels), so we can only
    compare the output tensor.
    """
    T = Q.size(-2)

    # Our implementation.
    mask = make_causal_mask(T, device=Q.device) if is_causal else None
    our_output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)

    # PyTorch reference.
    ref_output = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

    diff = (our_output - ref_output).abs().max().item()
    status = "PASS" if diff < atol else "FAIL"
    print(f"  [{status}] max |our - torch.nn.functional| = {diff:.2e}  (atol={atol:.0e})")

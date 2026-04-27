# NOTE: written but not hardware-verified — smoke-test before use
"""
attention_numpy.py — Scaled dot-product attention and multi-head attention
implemented entirely in NumPy.

No PyTorch, no deep-learning framework of any kind.  Every matrix multiply and
softmax is written explicitly so that tensor shapes are fully transparent.

Shape conventions used throughout:
    B  — batch size
    T  — sequence length (number of tokens)
    d  — d_model (residual-stream dimension)
    H  — number of attention heads
    d_k — key/query dimension per head  = d_model // H
    d_v — value dimension per head      = d_model // H  (equal to d_k here)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional


# ---------------------------------------------------------------------------
# Utility: manual softmax
# ---------------------------------------------------------------------------

def softmax(x: NDArray, axis: int = -1) -> NDArray:
    """
    Numerically stable softmax along *axis*.

    We subtract the maximum value along *axis* before exponentiation to
    prevent overflow.  This is mathematically equivalent to the standard
    definition:

        softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    but avoids exp() blowing up for large positive inputs.

    Shape: unchanged from input.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Single-head scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: NDArray,
    K: NDArray,
    V: NDArray,
    mask: Optional[NDArray] = None,
) -> tuple[NDArray, NDArray]:
    """
    Scaled dot-product attention as defined in Vaswani et al. (2017).

        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Parameters
    ----------
    Q : NDArray, shape (..., T_q, d_k)
        Query matrix.  T_q is the number of query positions.
    K : NDArray, shape (..., T_k, d_k)
        Key matrix.  T_k is the number of key/value positions.
    V : NDArray, shape (..., T_k, d_v)
        Value matrix.  d_v is the value dimension (often equal to d_k).
    mask : Optional[NDArray], shape (..., T_q, T_k) or broadcastable.
        Additive mask applied before softmax.  Use -1e9 (or -np.inf) for
        positions that should be masked out.  A causal mask sets all
        upper-triangular entries (j > i) to -1e9.

    Returns
    -------
    output : NDArray, shape (..., T_q, d_v)
        Attended output.
    weights : NDArray, shape (..., T_q, T_k)
        Attention weight matrix (after softmax) — useful for visualisation
        and for the compare.py correctness check.

    Notes
    -----
    The scaling factor 1/sqrt(d_k) is critical.  Without it, dot products
    grow in magnitude as d_k increases, pushing the softmax into saturated
    regions where gradients vanish.  See notes/02_transformer_architecture.md
    for the full explanation.
    """
    d_k = Q.shape[-1]
    scale = np.sqrt(d_k).astype(np.float64)

    # Q K^T: (..., T_q, d_k) @ (..., d_k, T_k) -> (..., T_q, T_k)
    scores = Q @ K.swapaxes(-2, -1) / scale  # raw attention logits

    # Apply mask (additive, so -1e9 entries become ~0 after softmax).
    if mask is not None:
        scores = scores + mask

    # Softmax over the key dimension (axis=-1).
    weights = softmax(scores, axis=-1)  # (..., T_q, T_k)

    # Weighted sum of values: (..., T_q, T_k) @ (..., T_k, d_v) -> (..., T_q, d_v)
    output = weights @ V

    return output, weights


def make_causal_mask(T: int) -> NDArray:
    """
    Build a causal (autoregressive) mask of shape (T, T).

    Entry [i, j] is -1e9 when j > i (future positions), 0.0 otherwise.
    After softmax, future positions receive attention weight ≈ 0.

    The mask is upper-triangular with the diagonal and everything below it
    set to 0 (allowed), and everything above the diagonal set to -1e9 (blocked).

    Example for T=4:
        [[ 0.   -1e9  -1e9  -1e9]
         [ 0.    0.   -1e9  -1e9]
         [ 0.    0.    0.   -1e9]
         [ 0.    0.    0.    0. ]]
    """
    # np.triu(..., k=1) selects entries strictly above the diagonal.
    mask = np.triu(np.full((T, T), -1e9), k=1)
    return mask


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

def multi_head_attention(
    x: NDArray,
    W_Q: NDArray,
    W_K: NDArray,
    W_V: NDArray,
    W_O: NDArray,
    n_heads: int,
    mask: Optional[NDArray] = None,
) -> NDArray:
    """
    Multi-head self-attention.

    The model dimension d_model is split into *n_heads* parallel "heads",
    each operating on a d_k = d_model // n_heads dimensional subspace.
    Outputs from all heads are concatenated and projected back to d_model
    by the output projection W_O.

        MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
        where head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)

    In practice Q, K, and V all come from the same input x (self-attention).

    Parameters
    ----------
    x   : NDArray, shape (B, T, d_model)
        Input sequence (batch of token embeddings, possibly with positional
        encoding added).
    W_Q : NDArray, shape (d_model, d_model)
        Query projection weight.
    W_K : NDArray, shape (d_model, d_model)
        Key projection weight.
    W_V : NDArray, shape (d_model, d_model)
        Value projection weight.
    W_O : NDArray, shape (d_model, d_model)
        Output projection weight (applied after concatenating heads).
    n_heads : int
        Number of attention heads H.  Must divide d_model evenly.
    mask : Optional[NDArray], shape (T, T) or (B, H, T, T)
        Additive attention mask (causal or padding).

    Returns
    -------
    output : NDArray, shape (B, T, d_model)

    Shape walkthrough
    -----------------
    B=batch, T=seq_len, d=d_model, H=n_heads, d_k=d//H

    x           : (B, T, d)
    x @ W_Q     : (B, T, d)  — full-model Q projection
    reshape     : (B, T, H, d_k)
    transpose   : (B, H, T, d_k)  — bring heads to batch dimension
    attention   : (B, H, T, d_k)
    transpose   : (B, T, H, d_k)
    reshape     : (B, T, d)  — concatenate heads
    @ W_O       : (B, T, d)
    """
    B, T, d_model = x.shape
    assert d_model % n_heads == 0, (
        f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
    )
    d_k = d_model // n_heads

    # Project to Q, K, V in the full model dimension.
    # (B, T, d_model) @ (d_model, d_model) -> (B, T, d_model)
    Q_full = x @ W_Q
    K_full = x @ W_K
    V_full = x @ W_V

    # Reshape and transpose to separate heads.
    # (B, T, d_model) -> (B, T, H, d_k) -> (B, H, T, d_k)
    def split_heads(z: NDArray) -> NDArray:
        return z.reshape(B, T, n_heads, d_k).transpose(0, 2, 1, 3)

    Q_h = split_heads(Q_full)  # (B, H, T, d_k)
    K_h = split_heads(K_full)  # (B, H, T, d_k)
    V_h = split_heads(V_full)  # (B, H, T, d_k)

    # Expand mask to (B, H, T, T) if it is 2-D.
    if mask is not None and mask.ndim == 2:
        mask_expanded = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, T, T)
    else:
        mask_expanded = mask

    # Attend independently in each head.
    # (B, H, T, d_k) each — attention_numpy handles batched dims automatically.
    attn_output, _ = scaled_dot_product_attention(Q_h, K_h, V_h, mask=mask_expanded)
    # attn_output shape: (B, H, T, d_k)

    # Merge heads: transpose back and reshape.
    # (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, d_model)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, d_model)

    # Output projection.
    # (B, T, d_model) @ (d_model, d_model) -> (B, T, d_model)
    output = attn_output @ W_O

    return output

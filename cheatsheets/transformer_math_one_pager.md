# Transformer Math — One-Pager

## Tensor shapes (decoder-only, single layer)

| Tensor | Shape | Notes |
|---|---|---|
| Input tokens | `[B, S]` | B = batch, S = seq len |
| Token embedding | `[B, S, d_model]` | Embedding matrix E ∈ ℝ^{V×d_model} |
| Q / K / V (per head) | `[B, S, d_k]` | d_k = head_dim = d_model / n_heads |
| Attention scores | `[B, n_heads, S, S]` | Pre-softmax; scale by 1/√d_k |
| Attention output | `[B, S, d_model]` | After W_O projection |
| FFN hidden | `[B, S, d_ff]` | d_ff = 4·d_model (standard) or ≈2.67·d_model (SwiGLU) |
| Residual stream | `[B, S, d_model]` | Unchanged width throughout all N layers |
| Logits | `[B, S, V]` | Unembedding; often weight-tied with E |

---

## Parameter count formulas

```
Embedding:         V × d_model
Q/K/V projections: 3 × d_model × d_k × n_heads  =  3 × d_model²
Output projection: d_model × d_model             =  d_model²
Attention total:   4 × d_model²

FFN (standard):    2 × d_model × d_ff            =  8 × d_model²
FFN (SwiGLU):      3 × d_model × d_ff_swiglu     ≈  8 × d_model²  (d_ff_swiglu ≈ 8/3 × d_model)

Per-layer total:   ≈ 12 × d_model²
Full model:        n_layers × 12 × d_model²  +  V × d_model
```

**Worked example — 7B-class model** (LLaMA-style):
`n_layers=32, d_model=4096, d_ff=11008, n_heads=32, head_dim=128`

```
Attention per layer:  4 × 4096²          = 67,108,864  ≈ 67 M
FFN per layer:        ~3 × 4096 × 11008  ≈ 135 M  (SwiGLU: 3 matrices)
Per layer:            ≈ 202 M
32 layers:            ≈ 6.46 B
Embedding (V=32000):  32000 × 4096       ≈ 131 M
Total:                ≈ 6.6 B  (rounds to "7B" with unembedding)
```

---

## FLOPs (forward pass, approximate)

```
FLOPs ≈ 2 × n_params × n_tokens
```

Rule of thumb: each parameter participates in ~2 multiply-adds per token. For a 7B model processing 1 token: ≈ 14 × 10⁹ FLOPs.

---

## KV cache size formula

```
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_element
```

**Worked example — 7B at 4k context, FP16** (n_kv_heads = n_heads = 32):

```
2 × 32 × 32 × 128 × 4096 × 2 = 2,147,483,648 bytes  ≈ 2 GB per sequence
```

GQA/MQA reduces `n_kv_heads`. LLaMA-3 70B: n_kv_heads=8 vs n_heads=64 → 8× KV cache reduction.

---

## Dimensionality choices

| Choice | Typical value | Why |
|---|---|---|
| head_dim (d_k) | 64–128 | Flash-attention tile efficiency; hardware alignment |
| d_ff (standard) | 4 × d_model | Empirically near-optimal from original paper |
| d_ff (SwiGLU) | ≈ 8/3 × d_model | Three matrices; keeps total params ≈ same as standard |
| Vocab size V | 32 000–128 000 | BPE/SentencePiece; larger V → better tokenisation efficiency |

---

## Memory to load weights

```
weight_bytes = n_params × bytes_per_param
```

| Precision | bytes/param | 7B model |
|---|---|---|
| FP32 | 4 | 28 GB |
| BF16 / FP16 | 2 | 14 GB |
| INT8 | 1 | 7 GB |
| INT4 | 0.5 | 3.5 GB |

RTX 4000 Ada (20 GB): 7B at BF16 fits (~14 GB); leaves ~6 GB for KV cache.
RTX 3080 (10 GB): 7B at INT4 (~3.5 GB weights) + ~6 GB KV budget.

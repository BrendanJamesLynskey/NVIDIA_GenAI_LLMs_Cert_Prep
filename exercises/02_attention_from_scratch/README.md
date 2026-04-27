# 02 — Attention From Scratch

This exercise implements scaled dot-product attention and multi-head attention twice — once in NumPy with an explicit manual softmax, and once in PyTorch — and then asserts that both implementations agree to within a numerical tolerance of 1e-4. A comparison script also validates the PyTorch implementation against `torch.nn.functional.scaled_dot_product_attention`. The goal is to make tensor shapes, the scaling factor, and causal masking concrete before relying on framework abstractions. This exercise maps directly to the attention section of [notes/02\_transformer\_architecture.md](../../notes/02_transformer_architecture.md).

---

## Hardware requirements

No GPU required for the NumPy or CPU-PyTorch paths. The implementations run on any machine with Python 3.9 or later. If PyTorch detects a CUDA device it will be ignored — all tensors are kept on CPU.

Minimum confirmed: any modern CPU with 4 GB RAM.

---

## The maths

### Scaled dot-product attention

Given input sequence X ∈ ℝ^{T × d\_model}, three learned linear projections produce:

| Matrix | Formula | Shape |
|---|---|---|
| Q (queries) | X W\_Q | T × d\_k |
| K (keys)    | X W\_K | T × d\_k |
| V (values)  | X W\_V | T × d\_v |

The attention output is:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) · V
```

**Step-by-step:**

1. `Q K^T` — dot product of every query with every key. Shape: T × T. Entry [i, j] measures how much query i attends to key j.
2. `/ sqrt(d_k)` — scale to prevent dot products from growing large with d\_k, which would saturate softmax and cause near-zero gradients. Without scaling, softmax(QK^T / 1) approximates a one-hot distribution for large d\_k.
3. `+ mask` — add -∞ to positions that should be blocked. For causal (autoregressive) masking the mask is upper-triangular: entry [i, j] = -∞ when j > i, meaning token i cannot attend to any future token j.
4. `softmax(...)` — convert each row of the score matrix to a probability distribution over keys. Masked positions become ≈ 0.
5. `· V` — take a weighted sum of the value vectors. Shape: T × d\_v.

### Shape table

| Symbol | Meaning | Typical value (GPT-2 base) |
|---|---|---|
| B | batch size | 1–32 |
| T | sequence length | up to 1024 |
| d\_model | residual stream dimension | 768 |
| H | number of attention heads | 12 |
| d\_k = d\_v = d\_model / H | per-head dimension | 64 |

### Multi-head attention

Rather than one large attention, the model dimension d\_model is split across H heads, each operating in a d\_k = d\_model / H dimensional subspace:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
where head_i = Attention(Q W_{Q,i}, K W_{K,i}, V W_{V,i})
```

Implementation procedure:
1. Project x to full-model Q, K, V: (B, T, d\_model) × (d\_model, d\_model) → (B, T, d\_model).
2. Reshape and transpose to separate heads: (B, T, d\_model) → (B, H, T, d\_k).
3. Run scaled dot-product attention independently for each head (batched along the H dimension).
4. Transpose and reshape to merge heads: (B, H, T, d\_k) → (B, T, d\_model).
5. Apply output projection W\_O: (B, T, d\_model) → (B, T, d\_model).

### Key points

- **d\_k scaling matters.** With d\_k = 64 and random unit-normal Q and K, the raw dot product QK^T has standard deviation ≈ 8 (= sqrt(64)). Without scaling, softmax sees inputs with std ≈ 8 and saturates to near-one-hot. With scaling by 1/sqrt(64) = 0.125, softmax inputs have std ≈ 1 and the gradient flows normally.
- **Causal mask is upper-triangular -inf.** After softmax, -inf becomes 0 weight. Using -1e9 (NumPy version) versus -inf (PyTorch version) are functionally equivalent for typical inputs; -inf is numerically preferable.
- **Multi-head: split d\_model across heads, attend independently, concat.** Each head learns to attend to a different aspect of the context. The per-head dimensionality d\_k is smaller, but the total parameter count is identical to a single-head attention with the same d\_model.

---

## File layout

| File | Purpose |
|---|---|
| `attention_numpy.py` | `scaled_dot_product_attention`, `multi_head_attention`, `make_causal_mask`, `softmax` — pure NumPy |
| `attention_torch.py` | Same functions in PyTorch; `compare_with_torch_sdpa` for reference comparison |
| `compare.py` | Runs all comparisons, asserts outputs match, prints PASS/FAIL |
| `test_attention.py` | pytest smoke tests |
| `requirements.txt` | `numpy`, `torch`, `pytest` |

---

## Setup

```bash
cd exercises/02_attention_from_scratch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No downloads at first run.

---

## Run

**Comparison script (prints PASS/FAIL for each check):**

```bash
python compare.py
```

**Smoke tests:**

```bash
pytest test_attention.py -v
```

---

## Expected output

`compare.py`:

```
Single-head SDPA (no mask)
  [PASS] output: max |numpy - torch| = ~1e-15
  [PASS] weights: max |numpy - torch| = ~1e-15
Single-head SDPA (causal mask)
  [PASS] causal output: max |numpy - torch| = ~1e-15
  [PASS] causal property check: position-0 output unchanged...
Multi-head attention
  [PASS] multi-head output: max |numpy - torch| = ~1e-15
PyTorch SDPA vs torch.nn.functional...
  [PASS] non-causal: max |our - torch.nn.functional| = ~1e-6
  [PASS] causal: max |our - torch.nn.functional| = ~1e-6
```

The NumPy-vs-PyTorch differences are sub-1e-14 (float64 arithmetic is identical). The PyTorch-vs-`torch.nn.functional` difference is slightly larger because `F.scaled_dot_product_attention` may use a fused kernel internally (FlashAttention if CUDA is available, or a different float32 accumulation order on CPU), but should be well within 1e-4.

Rough estimate: all tests complete in under 10 seconds on any modern CPU.

---

## What to study from this

- **Shape manipulation is the hard part.** The attention formula is three lines; making the shapes work for batched multi-head attention is most of the implementation effort. Being able to trace (B, T, d\_model) → (B, H, T, d\_k) → (B, T, d\_model) by hand is a reliable exam preparation technique.
- **Softmax saturation is the reason for the scaling.** This comes up directly in NCA/NCP exam questions framed as "why do we divide by sqrt(d\_k)?". The distractor answer is usually "to normalise the output magnitude" — but it is the gradient (via softmax) that is stabilised, not the output magnitude.
- **Causal masking enforces the autoregressive property.** Encoder-only models (BERT) use no causal mask; decoder-only models (GPT) use the upper-triangular -inf mask. This distinction is a reliable exam topic.
- **Full treatment:** [notes/02\_transformer\_architecture.md](../../notes/02_transformer_architecture.md) and [LLM\_Hub\_Transformer\_Architecture](https://github.com/BrendanJamesLynskey/LLM_Hub_Transformer_Architecture).

---

## Further reading

- Vaswani et al. (2017), "Attention Is All You Need": <https://arxiv.org/abs/1706.03762>
- Dao et al. (2022), "FlashAttention": <https://arxiv.org/abs/2205.14135>
- PyTorch documentation, `torch.nn.functional.scaled_dot_product_attention`: <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>

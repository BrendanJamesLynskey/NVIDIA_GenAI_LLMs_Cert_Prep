# Quantisation and KV Cache

## — QUANTISATION —

### Precision formats

| Format | Bytes/param | Dynamic range | Hardware support | Notes |
|---|---|---|---|---|
| FP32 | 4 | ~1.2×10⁻³⁸ – 3.4×10³⁸ | All | Training baseline |
| BF16 | 2 | Same exponent as FP32 | Ampere+ | Preferred for training stability |
| FP16 | 2 | ~6×10⁻⁸ – 6.5×10⁴ | All | Serving baseline |
| FP8 E4M3 | 1 | Narrow; high precision | Hopper (H100+) | Inference weights+activations |
| FP8 E5M2 | 1 | Wider range; less precision | Hopper (H100+) | Gradient storage in training |
| INT8 | 1 | −128 to 127 | Ampere+ (A10G, A100) | SmoothQuant W8A8 |
| FP4 E2M1 (NVFP4) | 0.5 | Very narrow | Blackwell native; Ada/Hopper via TRT-LLM | W4A8; needs calibration or QAT |
| INT4 AWQ/GPTQ | 0.5 | −8 to 7 | Any (dequant to FP16 for compute) | Weight-only; consumer GPUs |

### Weight-only vs weight-and-activation

| Type | Method | What runs at low precision | Benefit | Limitation |
|---|---|---|---|---|
| **Weight-only** | AWQ, GPTQ | Weights stored INT4; dequant to FP16 before matmul | Reduces memory bandwidth | Compute still FP16; no tensor-core gain |
| **Weight+activation** | SmoothQuant (INT8), FP8, NVFP4 | Both weights and activations; matmul runs at target precision | Full tensor-core throughput benefit | Activations have dynamic range; needs calibration |

### Calibration vs QAT

- **PTQ (post-training quantisation):** small calibration dataset → compute scaling factors. Fast; accuracy drop of 0.5–2% at INT4.
- **QAT (quantisation-aware training):** simulated quantisation in forward pass during fine-tune. Better accuracy; costs a training run.

### Hardware generation support

| Generation | Tensor core precisions | Notes |
|---|---|---|
| Ampere (A100, RTX 3080) | FP16, BF16, INT8, TF32 | No FP8; no INT4 tensor cores (weight-only INT4 via dequant) |
| Ada Lovelace (RTX 4000 Ada) | FP16, BF16, INT8, TF32 | FP8 via TRT-LLM NVFP4 emulation; no native Blackwell FP4 |
| Hopper (H100) | + FP8 (native) | FP8 tensor cores; TRT-LLM FP8 path primary |
| Blackwell (B100/B200) | + FP4 native (NVFP4) | W4A8 first-class; highest throughput per watt |

### AWQ vs GPTQ vs FP8

| | AWQ | GPTQ | FP8 |
|---|---|---|---|
| Bit width | INT4 weight | INT4 weight | 8-bit weight+act |
| Approach | Activation-aware scaling protects salient weights | Layer-wise second-order (Hessian) minimisation | Hardware-native; calibration scaling |
| Accuracy | Better than GPTQ at same bit-width | Slightly lower than AWQ typically | Near-lossless on Hopper |
| Hardware target | Consumer GPU (weight-only) | Consumer GPU (weight-only) | H100 / Hopper |
| TRT-LLM support | Yes | Yes | Yes (primary Hopper path) |

---

## — KV CACHE —

### Memory formula

```
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_element
```

**Worked example — 7B model, FP16, 4k context:**
`n_layers=32, n_kv_heads=32, head_dim=128, seq_len=4096, bytes=2`

```
2 × 32 × 32 × 128 × 4096 × 2 = 2,147,483,648 bytes  ≈ 2 GB per sequence
```

At batch size 8: ~16 GB KV cache alone — exceeds RTX 4000 Ada's full 20 GB.

GQA cuts `n_kv_heads`: LLaMA-3 70B uses 8 KV heads vs 64 query heads → 8× smaller KV cache.

### PagedAttention

- Pre-PagedAttention: KV cache allocated as contiguous blocks; over-provisioned by 20–80% (unknown sequence length).
- **PagedAttention (vLLM, 2023):** KV cache split into fixed-size *pages*; non-contiguous; block table maps logical→physical.
- Benefits: near-zero internal fragmentation; prefix sharing across requests (copy-on-write); larger effective batch sizes.

### KV cache quantisation

KV cache can be quantised independently of weights — typically **FP8 or INT8**. ~2× memory reduction.

| | Weight quant | KV quant |
|---|---|---|
| What it reduces | Model weight memory | KV cache memory |
| When it matters | At load time | At long context / large batch |
| Accuracy impact | 0.5–2% (INT4); near-zero (INT8/FP8) | Small; FP8 KV is near-lossless in practice |
| TRT-LLM | Separate flag from weight quant |  |

---

## Memory budget: RTX 4000 Ada (20 GB)

```
Available:          20 GB
7B weights (BF16):  ~14 GB   → remaining: ~6 GB for KV cache
7B weights (INT4):  ~3.5 GB  → remaining: ~16.5 GB for KV cache

KV cache @ FP16, 4k context, n_kv_heads=32:  2 GB/sequence
→ 7B BF16: fits ~3 sequences at 4k context
→ 7B INT4: fits ~8 sequences at 4k context

KV cache @ INT8 (half the FP16 cost):  1 GB/sequence
→ 7B BF16: fits ~6 sequences at 4k context

13B weights (BF16):  ~26 GB  → does not fit
13B weights (INT4):  ~7 GB   → ~13 GB remaining; usable at moderate context
```

# Inference Optimisation

Model Optimisation is the single largest NCP-GENL domain at 17%. The exam expects working knowledge of the full inference stack: how memory is consumed, how throughput is maximised, what each optimisation technique actually does, and which frameworks implement which techniques. This note is correspondingly detailed.

Cross-reference: [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM) for TensorRT-LLM depth; [LLM\_Hub\_Local\_LLM\_Hosting](https://github.com/BrendanJamesLynskey/LLM_Hub_Local_LLM_Hosting) for vLLM, Ollama, TGI, and llama.cpp comparisons.

---

## The Throughput / Latency / Memory Triangle

Inference optimisation involves three coupled constraints:

- **Throughput** — tokens generated per second across all concurrent requests. Maximised by large batch sizes and high arithmetic intensity.
- **Latency** — two sub-components matter separately:
  - **TTFT (time to first token)** — the prefill phase; processes the entire input prompt in parallel (bounded by compute).
  - **TPOT (time per output token)** — the autoregressive decode phase; generates one token per forward pass (bounded by memory bandwidth, because weight matrices must be streamed from HBM each step with a batch of 1).
- **Memory** — GPU HBM capacity sets the hard ceiling on model size and KV cache.

These constraints trade off: increasing batch size improves throughput but increases latency for early requests in the batch. Reducing model precision frees memory and often improves throughput but may degrade quality. Every optimisation technique in this note occupies a specific position in this triangle.

---

## KV Cache and Its Memory Cost

During autoregressive decoding, attention requires access to the key and value tensors computed for every previous token. Recomputing these from scratch at each step would be prohibitively expensive; instead they are cached — the KV cache.

The memory footprint of the KV cache for a single sequence is:

```
KV_cache_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_element
```

The factor of 2 is for keys and values. For a model with 32 layers, 32 KV heads, head dimension 128, sequence length 4096, and FP16 (2 bytes per element):

```
2 × 32 × 32 × 128 × 4096 × 2 = 2,147,483,648 bytes ≈ 2 GB per sequence
```

This is not a small number. At batch size 8, that is 16 GB just for the KV cache — exceeding the RTX 4000 Ada's entire 20 GB. Long context makes this worse non-linearly: doubling the sequence length doubles the KV cache. Models with grouped-query attention (GQA) or multi-query attention (MQA) reduce `n_kv_heads` substantially — Llama 3 70B uses 8 KV heads against 64 query heads, cutting KV cache by 8×.

The KV cache dominates the memory budget at long context and modest model sizes. On Brendan's hardware (10 GB and 20 GB), the practical context-length ceiling for even a 7B model at FP16 without KV quantisation is in the low thousands of tokens once the model weights occupy their share.

---

## PagedAttention and vLLM

Before PagedAttention (Kwon et al., 2023 — arXiv:2309.06180), KV caches were allocated as contiguous memory blocks reserved at the start of a request. Because sequence length is not known in advance, systems over-provisioned, wasting 20–80% of KV cache memory to internal fragmentation.

**PagedAttention** borrows the OS virtual-memory paging concept. The KV cache is divided into fixed-size *blocks* (pages), each holding the keys and values for a fixed number of tokens. Pages are allocated on demand and stored non-contiguously; a block table maps logical page indices to physical memory locations. Consequences:

- Near-zero internal fragmentation — only the last page of a sequence may be partially filled.
- **KV cache sharing** — prompt prefixes shared across requests (e.g., a system prompt used by many parallel queries) can be stored once and referenced by multiple sequences via copy-on-write.
- Large effective batch sizes — more requests can be in flight simultaneously with the same GPU memory.

vLLM's throughput improvement over FasterTransformer (the prior state of the art) was reported at 2–4× at the same latency level.

---

## Continuous / In-Flight Batching

Static batching processes a fixed batch of requests in lockstep — the batch is held until every sequence finishes generating, then a new batch starts. This wastes GPU cycles whenever a short sequence finishes early and its slot sits idle waiting for the longest sequence in the batch.

**Continuous batching** (also called in-flight batching or iteration-level scheduling) releases a slot as soon as a sequence finishes and immediately inserts a new request. From the GPU's perspective, every forward pass is at (near) peak utilisation. The serving engine manages a dynamic set of active sequences whose membership can change every token step.

TensorRT-LLM implements in-flight batching natively; its C++ runtime manages the batch state, KV cache allocation, and request scheduling at the iteration level. vLLM implements the same concept. For TensorRT-LLM internals, see [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM).

---

## FlashAttention

Standard attention computes the full `Q × K^T` score matrix in HBM before applying softmax and multiplying by V. For sequence length L and head dimension d, the score matrix is O(L²) in memory. At L = 8192 and 32 heads, this is impractically large and requires multiple round-trips between HBM and the compute units.

**FlashAttention v1** (Dao et al., 2022 — arXiv:2205.14135) reformulated attention as a tiled algorithm that keeps intermediate results in on-chip SRAM. The full L×L matrix is never materialised in HBM. This is IO-optimal for exact attention — it reduces HBM read/write volume from O(L²) to O(L), at the cost of more arithmetic (recomputing softmax normalisers). Measured speedups: up to 3× on GPT-2 training at 1K tokens; training sequence lengths up to 16K became practical.

**FlashAttention v2** (Dao, 2023 — arXiv:2307.08691) addressed the 25–40% GPU utilisation ceiling of v1. Three changes: (1) fewer non-matrix operations by restructuring the softmax rescaling; (2) parallelisation across sequence dimension even within a single head; (3) better warp-level work distribution within a thread block. Result: ~2× speedup over v1, reaching 50–73% of theoretical peak FLOPs/s on A100.

**FlashAttention v3** targets H100-class hardware specifically (using FP8 tensor cores and the async pipeline features of Hopper). On consumer hardware (Ampere/Ada), v2 is the relevant implementation.

FlashAttention is a training and prefill optimisation. It does not change decode memory — the KV cache is still needed during autoregressive generation — but it dramatically accelerates the prefill phase and makes long-context training feasible.

---

## Quantisation

Quantisation reduces the numerical precision of weights and/or activations, shrinking memory footprint and often improving throughput by using lower-precision matrix-multiply units.

### Weight formats

| Format | Bits/weight | Typical use case |
|--------|-------------|-----------------|
| FP16/BF16 | 16 | Baseline serving |
| INT8 (W8A8) | 8 weight + 8 activation | SmoothQuant; good accuracy, ~2× memory |
| FP8 (W8A8) | 8 weight + 8 activation | H100 FP8 tensor cores; TensorRT-LLM native |
| INT4 AWQ | 4 weight + 16 activation | Weight-only; consumer GPU serving |
| INT4 GPTQ | 4 weight + 16 activation | Weight-only; post-training |
| NVFP4 (W4A8) | 4 weight + 8 activation | TensorRT-LLM on Ada/Hopper |

**Weight-only quantisation** (AWQ, GPTQ) quantises weights to INT4 but dequantises to FP16 before the matmul. This reduces memory bandwidth pressure (loading fewer bytes per weight) and thus improves decode throughput on bandwidth-limited hardware, but the compute itself still runs in FP16. There is no benefit from narrower compute units.

**Weight-and-activation quantisation** (INT8 SmoothQuant, FP8) runs the matmul itself at lower precision, exploiting NVIDIA's INT8/FP8 tensor core throughput. This can improve throughput beyond the memory-bandwidth improvement alone. The challenge is that activations have dynamic range that varies per token, making uniform quantisation lossy; SmoothQuant migrates quantisation difficulty from activations to weights by applying a per-channel scale.

**Calibration**: post-training quantisation (PTQ) requires a small calibration dataset to determine scaling factors. Quantisation-aware training (QAT) fine-tunes with simulated quantisation in the forward pass, producing better accuracy at the cost of a training run.

**Accuracy/latency trade-off**: INT4 weight-only typically incurs 0.5–2% accuracy degradation on standard benchmarks; INT8 weight-and-activation is often near-lossless; FP4/NVFP4 requires careful calibration or QAT. Always evaluate on task-specific metrics, not just perplexity.

### KV Cache Quantisation

KV cache can be quantised independently of weights — typically to INT8 or FP8 — reducing the memory cost of long contexts without affecting the model weights at all. This is particularly valuable on consumer hardware where the KV cache competes with weights for the same HBM. TensorRT-LLM exposes KV cache quantisation as a separate option from weight quantisation.

---

## Speculative Decoding

Autoregressive decoding is sequential: each token is generated one at a time. Speculative decoding breaks this by using a small, fast *draft model* to generate k candidate tokens, then verifying all k with the large *target model* in a single forward pass. Correct draft tokens are accepted; the first incorrect token causes rejection from that point, and the target model's corrected token is taken instead.

Under the right conditions (draft model well-aligned with target, k in the range 3–7), speculative decoding can reduce TPOT by 2–3× with no change in output distribution — the verification step guarantees the same distribution as the target model alone.

**Medusa** (Cai et al., 2024) replaces the draft model with extra decoding heads added directly to the target model, jointly trained to predict tokens at positions +1, +2, …+k. No separate draft model is needed; the extra heads add minimal latency to each target forward pass.

**EAGLE** (Li et al., 2024) uses an auto-regressive draft model trained to predict one step ahead at the feature (hidden state) level rather than the token level, achieving higher acceptance rates than Medusa in practice. EAGLE-3 is the current iteration, supported natively in TensorRT-LLM.

The speedup versus verification cost trade-off depends on the draft acceptance rate and the ratio of draft-to-target model size. A poorly-calibrated draft that generates mostly rejected tokens adds overhead without acceleration.

---

## Multi-LoRA Inference

Fine-tuned models are often LoRA adapters on a shared base. Serving multiple adapters simultaneously with one base model copy avoids reloading the base weights for each adapter switch.

**S-LoRA** (Sheng et al., 2023) and **Punica** introduced the concept of batching requests across different adapters in a single forward pass. The base model forward pass is shared; adapter deltas are applied per-request using custom CUDA kernels that fuse the LoRA A×B computation into the same pass. The adapters themselves (typically small — rank-16 LoRA on a 7B model is a few hundred MB per adapter) are stored in GPU or CPU memory and swapped as needed.

vLLM supports multi-LoRA serving natively. This is particularly relevant for platforms that fine-tune per-user or per-task and want to host tens or hundreds of adapters without proportional memory cost.

---

## Inference Frameworks

| Framework | Primary strength | Hardware target | Notes |
|-----------|-----------------|-----------------|-------|
| **TensorRT-LLM** | NVIDIA-optimised throughput; FP8/FP4; in-flight batching | NVIDIA (A10G, A100, H100, Ada) | Engine builder + runtime; Triton backend for serving. See [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM) |
| **vLLM** | PagedAttention; OpenAI-compatible API; broad model support | NVIDIA + AMD | Best for high-throughput serving of standard architectures |
| **SGLang** | Structured generation; RadixAttention for prefix caching; fast TTFT | NVIDIA | Strong for agentic and multi-turn workloads |
| **TGI (Text Generation Inference)** | Hugging Face ecosystem integration | NVIDIA + AMD + Intel | Simpler deployment; less tuning flexibility |
| **llama.cpp** | CPU + consumer GPU; GGUF quantisation | CPU, NVIDIA, Apple Metal | Widest hardware support; lower throughput than GPU-native |
| **Ollama** | Zero-config local serving; wraps llama.cpp | CPU, NVIDIA, Apple | Best for individual developer use; not production-grade |

For a detailed comparison, see [LLM\_Hub\_Local\_LLM\_Hosting](https://github.com/BrendanJamesLynskey/LLM_Hub_Local_LLM_Hosting).

**When to reach for which**: TensorRT-LLM is the right choice when maximising NVIDIA GPU utilisation in a production setting and the target hardware is NVIDIA. vLLM is the right choice for rapid deployment with good performance across a wide model zoo. SGLang is worth considering for structured-output-heavy or agentic workloads. llama.cpp / Ollama are appropriate for local development on consumer hardware.

---

## Hardware Realities: Brendan's Setup

On the RTX 4000 Ada (20 GB):
- **7B models at BF16**: ~14 GB for weights, leaving ~6 GB for KV cache. Usable for modest batch sizes and context lengths up to a few thousand tokens.
- **13B models at BF16**: ~26 GB — does not fit. At INT4 (AWQ/GPTQ): ~7 GB for weights, plenty of headroom.
- **30B models at INT4**: ~15–16 GB for weights, leaving 4–5 GB for KV cache. Feasible for short contexts at batch size 1–2.
- **70B models**: require INT4 and exceed 20 GB by a wide margin; not feasible without multi-GPU or CPU offloading.

On the RTX 3080 (10 GB):
- **7B at INT4**: ~4 GB weights, ~6 GB KV cache budget. Usable.
- **13B at INT4**: ~7 GB weights, ~3 GB for KV cache — very tight context budget.

KV cache quantisation (INT8) approximately halves the KV cache footprint, meaningfully extending usable context length on both machines.

---

## Likely Exam Angles

- **KV cache memory calculation**: given model architecture parameters, compute KV cache size for a given sequence length and batch size. Know that GQA/MQA reduces it.
- **PagedAttention**: explain fragmentation in pre-PagedAttention systems and how paged blocks solve it; know the vLLM origin.
- **Continuous batching vs static batching**: explain why static batching wastes GPU cycles and how in-flight batching solves this.
- **FlashAttention**: explain the IO-optimal tiling approach; know v1 (2022, arXiv:2205.14135) and v2 (2023, arXiv:2307.08691) contributions; know it addresses prefill, not decode memory.
- **Quantisation trade-offs**: distinguish weight-only (AWQ/GPTQ) from weight-and-activation (SmoothQuant/FP8); calibration vs QAT; accuracy implications.
- **Speculative decoding**: explain the draft-then-verify mechanism; know Medusa and EAGLE as variants; identify where speedup comes from and when it fails.

---

## Further Reading

- Kwon et al. (2023) — PagedAttention / vLLM: <https://arxiv.org/abs/2309.06180>
- Dao et al. (2022) — FlashAttention v1: <https://arxiv.org/abs/2205.14135>
- Dao (2023) — FlashAttention v2: <https://arxiv.org/abs/2307.08691>
- Lin et al. (2023) — AWQ: <https://arxiv.org/abs/2306.00978>
- Frantar et al. (2022) — GPTQ: <https://arxiv.org/abs/2210.17323>
- Xiao et al. (2023) — SmoothQuant: <https://arxiv.org/abs/2211.10438>
- Cai et al. (2024) — Medusa: <https://arxiv.org/abs/2401.10774>
- TensorRT-LLM developer page: <https://developer.nvidia.com/tensorrt-llm>
- vLLM docs: <https://docs.vllm.ai/>
- SGLang paper: <https://arxiv.org/abs/2312.07104>
- [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM) — TensorRT-LLM depth
- [LLM\_Hub\_Local\_LLM\_Hosting](https://github.com/BrendanJamesLynskey/LLM_Hub_Local_LLM_Hosting) — vLLM, Ollama, TGI, llama.cpp

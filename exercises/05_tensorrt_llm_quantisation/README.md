# Exercise 05 — TensorRT-LLM Quantisation Walkthrough

Step-by-step walkthrough for building a quantised TensorRT-LLM engine from a small open-weight LLM, running it, and comparing latency against a plain Hugging Face baseline.

**This is a documented walkthrough, not a one-shot runnable demo.** The build process pulls a multi-GB container image and a multi-GB model checkpoint. The full setup (container pull + model download + engine build) can take 30–90 minutes depending on storage and GPU speed. Read the "Why this is a walkthrough" section before starting.

---

## Why this is a walkthrough not a one-shot

1. **Container image size.** The TensorRT-LLM container (`nvcr.io/nvidia/tensorrt_llm:1.2.1-...`) is 20–30 GB compressed on NGC. The first `docker compose pull` will take significant time on a home broadband connection.

2. **Model download.** The target model (Llama-3.2-1B) is approximately 2.5 GB from Hugging Face. A Hugging Face token is required if the repo has a gated licence.

3. **Engine build time.** `trtllm-build` compiles CUDA kernels specifically for the target GPU architecture. On the RTX 4000 Ada, a 1B-parameter FP8 engine build takes approximately 10–20 minutes (rough estimate, not measured). On the RTX 3080, an INT8 build takes a similar duration.

4. **Driver / container coupling.** TensorRT-LLM containers are tightly coupled to a specific CUDA Toolkit and driver version. If your local driver is older than the container requires, the container will fail to start with a CUDA version mismatch. Check the compatibility note below before pulling.

5. **Disk space.** Allow at least 60 GB free: container layers (~30 GB), model checkpoint (~3 GB unquantised), quantised checkpoint (~1.5 GB), compiled engine (~1 GB).

The `scripts/` directory contains the actual commands, heavily annotated. The README explains the conceptual flow and the rationale for each step. The combination is the teaching artefact — run the scripts as a hands-on lab, or read the README for exam-focused conceptual understanding.

---

## Hardware targets and precision choice

| Card | VRAM | Architecture | sm | FP8 tensor cores | Recommended precision |
|---|---|---|---|---|---|
| RTX 4000 Ada | 20 GB | Ada Lovelace | sm_89 | Likely no (AD104) — verify per the per-SKU list below | **INT8 (W8A8 SmoothQuant) or INT4 AWQ** |
| RTX 3080 | 10 GB | Ampere | sm_86 | No | **INT8 (W8A8 SmoothQuant) or INT4 AWQ** |

**FP8 enablement on Ada is per-SKU, not per-architecture.** Per [`NVIDIA_GPU_30_Ada_Low_Level`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_30_Ada_Low_Level), 4th-gen tensor cores have FP8 enabled only on **L40S, L40, and RTX 6000 Ada** (AD102-based datacenter/top-workstation cards); FP8 is fused off on the rest of the Ada line, including the consumer RTX 40 cards and the mid-range Ada workstation cards. The RTX 4000 Ada uses the AD104 die — the same family as RTX 4070 Ti — and is not on the FP8-enabled list. Treat FP8 on RTX 4000 Ada as not available; if you find a NVIDIA driver or container note that contradicts this, prefer the more recent NVIDIA source. INT8 SmoothQuant and INT4 AWQ are the realistic precisions on both cards in this exercise. The FP8 commands below are kept as reference for users with an FP8-capable Ada SKU (L40S/L40/RTX 6000 Ada) or any Hopper card (H100/H200/GH200).

Attempting to build an FP8 engine on the RTX 3080 will produce either a build error or silent fallback to a less optimal kernel — do not target FP8 on Ampere. On the RTX 4000 Ada, an FP8 build is expected to fail at kernel selection time given the fused-off tensor cores; verify with a small calibration run before relying on it. Both cards have enough headroom for a 1B-parameter INT8 engine with room for a calibration dataset and KV cache.

For a full discussion of quantisation formats and their memory implications, see [notes/08_inference_optimisation.md](../../notes/08_inference_optimisation.md).

---

## VRAM budget for the target model

Model: **Meta Llama-3.2-1B** (or Qwen2.5-1.5B as an alternative; commands are identical, substitute the HF model ID)

| Precision | Approx. weight footprint | KV cache budget at bs=4, ctx=2048 | Fits on RTX 3080? | Fits on RTX 4000 Ada? |
|---|---|---|---|---|
| BF16 (baseline) | ~2 GB | ~0.5 GB | Yes | Yes |
| FP8 (W8A8) | ~1 GB | ~0.25 GB (FP8 KV cache) | Kernel not available | Yes |
| INT8 SmoothQuant | ~1 GB | ~0.25 GB (INT8 KV cache) | Yes | Yes |
| INT4 AWQ | ~0.5 GB | ~0.5 GB (BF16 KV cache) | Yes | Yes |

These are approximate figures for a 1B-parameter model; actual numbers depend on architecture details (MLP hidden size, number of KV heads). Cross-reference the KV cache formula in [notes/08_inference_optimisation.md](../../notes/08_inference_optimisation.md).

---

## Driver and container compatibility

> **Verify against your local CUDA driver before pulling.**

The container tag used here is `nvcr.io/nvidia/trt_llm_backend:24.12-trtllm-v0.16.0`. This bundles:

- CUDA 12.6
- TensorRT-LLM v0.16.0
- Python 3.10

CUDA 12.6 requires driver **≥ 560** on Linux. Check your driver with `nvidia-smi`. The driver version appears in the top-right of the `nvidia-smi` output. If your driver is older, update it first:

```bash
sudo ubuntu-drivers install
# or for a specific version:
sudo apt install nvidia-driver-570
```

The alternative container tag for TensorRT-LLM 1.2.1 (the latest stable as of April 2026) is `nvcr.io/nvidia/tensorrt_llm:1.2.1-py3`, which requires CUDA 12.8 and driver ≥ 570. Use whichever matches your installed driver; the `trtllm-build` commands in this walkthrough are identical across both tags.

---

## Step 0 — Set up the environment

```bash
# Clone or navigate to this exercise directory
cd exercises/05_tensorrt_llm_quantisation

# Create a host-side venv for the baseline HF comparison script
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull the TensorRT-LLM container (large — allow time)
docker compose pull
```

---

## Step 1 — Download the model checkpoint

Llama-3.2-1B is available on Hugging Face as `meta-llama/Llama-3.2-1B`. It is a gated repository; you need a Hugging Face account and must accept the Meta licence.

```bash
# Authenticate with Hugging Face (one-time)
pip install huggingface_hub
huggingface-cli login
# Paste your HF token when prompted

# Download the model (approximately 2.5 GB)
huggingface-cli download meta-llama/Llama-3.2-1B \
    --local-dir ./models/Llama-3.2-1B \
    --local-dir-use-symlinks False
```

If you prefer a fully open (non-gated) alternative, substitute `Qwen/Qwen2.5-1.5B-Instruct`:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir ./models/Qwen2.5-1.5B \
    --local-dir-use-symlinks False
```

All subsequent commands assume the Llama path. Adjust `MODEL_DIR` in the scripts if using Qwen.

---

## Step 2 — Launch the TensorRT-LLM container (interactive shell)

```bash
docker compose run --rm trtllm bash
```

You are now inside the container. The `./models` and `./engines` directories on the host are mounted at `/workspace/models` and `/workspace/engines` respectively (see `docker-compose.yml`).

Verify the GPU is visible:

```bash
nvidia-smi
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

---

## Step 3 — Quantise the checkpoint

All commands below are run **inside the container**. The annotated versions with commentary are in `scripts/build_engine.sh`; the bare commands are reproduced here for reference.

### 3a. FP8 (RTX 4000 Ada only — sm_89)

FP8 is a weight-and-activation quantisation format. It runs the matrix multiplications at 8-bit float precision, exploiting the Ada FP8 tensor cores for throughput. A small calibration dataset is required to determine per-tensor scaling factors.

```bash
cd /workspace

python /usr/local/lib/python3.10/dist-packages/tensorrt_llm/examples/quantization/quantize.py \
    --model_dir /workspace/models/Llama-3.2-1B \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --output_dir /workspace/models/Llama-3.2-1B-fp8-checkpoint \
    --calib_size 128 \
    --dtype bfloat16
```

`--calib_size 128` means 128 calibration samples are used to compute the FP8 scaling factors. Larger calibration sets improve accuracy slightly at the cost of more calibration time. 128–512 is typical for PTQ.

### 3b. INT8 SmoothQuant (RTX 3080 — sm_86, and also valid on RTX 4000 Ada)

SmoothQuant migrates quantisation difficulty from activations to weights by applying a per-channel scaling factor before quantisation. The result is a weight-and-activation INT8 model that runs on NVIDIA INT8 tensor cores (available on Ampere and later).

```bash
python /usr/local/lib/python3.10/dist-packages/tensorrt_llm/examples/quantization/quantize.py \
    --model_dir /workspace/models/Llama-3.2-1B \
    --qformat int8_sq \
    --kv_cache_dtype int8 \
    --output_dir /workspace/models/Llama-3.2-1B-int8sq-checkpoint \
    --calib_size 128 \
    --dtype bfloat16
```

### 3c. INT4 AWQ weight-only (RTX 3080, most memory-efficient)

AWQ (Activation-aware Weight Quantisation) quantises weights to INT4 but keeps activations at BF16. The matrix multiplications run in BF16 (dequantising weights on the fly), so this saves memory bandwidth rather than compute throughput. Best for memory-constrained settings where throughput is secondary to fitting the model.

```bash
python /usr/local/lib/python3.10/dist-packages/tensorrt_llm/examples/quantization/quantize.py \
    --model_dir /workspace/models/Llama-3.2-1B \
    --qformat int4_awq \
    --output_dir /workspace/models/Llama-3.2-1B-int4awq-checkpoint \
    --calib_size 128 \
    --dtype bfloat16
```

---

## Step 4 — Build the TensorRT-LLM engine

`trtllm-build` compiles the quantised checkpoint into a GPU-specific optimised engine. This step produces `.engine` files that are binary artefacts tied to the exact GPU architecture. An engine built for sm_89 (Ada) will not run on sm_86 (Ampere) and vice versa.

The full annotated build script is `scripts/build_engine.sh`. The commands below are the key invocations.

### 4a. FP8 engine (RTX 4000 Ada — sm_89)

```bash
trtllm-build \
    --checkpoint_dir /workspace/models/Llama-3.2-1B-fp8-checkpoint \
    --output_dir /workspace/engines/Llama-3.2-1B-fp8 \
    --gemm_plugin fp8 \
    --use_fp8_context_fmha enable \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_seq_len 3072 \
    --workers 1
```

Key flags explained:

| Flag | Purpose |
|---|---|
| `--gemm_plugin fp8` | Selects the FP8 GEMM plugin for matrix multiplications. Enables FP8 tensor core usage. |
| `--use_fp8_context_fmha enable` | FP8 FlashMHA for the context (prefill) phase. Requires sm_89+. |
| `--max_batch_size 8` | Maximum concurrent requests in a batch. |
| `--max_input_len 2048` | Maximum prompt length. Longer prompts are rejected at runtime. |
| `--max_seq_len 3072` | Maximum total sequence length (prompt + generation). |
| `--workers 1` | Number of parallel builder workers. Increase on multi-GPU if building a tensor-parallel engine. |

### 4b. INT8 engine (RTX 3080 — sm_86)

```bash
trtllm-build \
    --checkpoint_dir /workspace/models/Llama-3.2-1B-int8sq-checkpoint \
    --output_dir /workspace/engines/Llama-3.2-1B-int8sq \
    --gemm_plugin auto \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_seq_len 3072 \
    --workers 1
```

`--gemm_plugin auto` lets TensorRT-LLM select the highest-precision GEMM available on the current architecture. For an INT8 SmoothQuant checkpoint on Ampere, this selects INT8 tensor cores.

---

## Step 5 — Run inference inside the container

Run `scripts/run_inference.py` inside the container:

```bash
python /workspace/scripts/run_inference.py \
    --engine_dir /workspace/engines/Llama-3.2-1B-fp8 \
    --prompts "Explain what a KV cache is in one paragraph."
```

Expected output (timing is illustrative — rough estimate, not measured):

```
Prompt: Explain what a KV cache is in one paragraph.
Generated: A key-value (KV) cache stores the attention keys and values …
Latency (TTFT): ~45 ms   |   Throughput: ~180 tok/s  [rough estimate, not measured]
```

---

## Step 6 — Baseline comparison (host, outside container)

Run `scripts/baseline_hf.py` on the host (in the `.venv`) with the same prompts:

```bash
source .venv/bin/activate
python scripts/baseline_hf.py \
    --model_dir ./models/Llama-3.2-1B \
    --prompts "Explain what a KV cache is in one paragraph."
```

This runs the same model at BF16 via plain Hugging Face `transformers.pipeline`. Compare the latency and generation quality. Quality should be nearly identical for FP8; INT4 AWQ may show minor differences on complex reasoning prompts.

---

## Conceptual map of the pipeline

```
HF checkpoint (BF16 safetensors)
         │
         ▼  quantize.py (NVIDIA Model Optimizer / modelopt)
Quantised checkpoint (FP8/INT8/INT4 + scaling factors)
         │
         ▼  trtllm-build
Compiled .engine files (sm_89 or sm_86 specific)
         │
         ├──▶  TRT-LLM C++ runtime (run_inference.py)
         │         in-flight batching, paged KV cache
         │
         └──▶  Triton TRT-LLM backend  (see Exercise 04)
                    HTTP :8000 / gRPC :8001
```

For the full TRT-LLM component breakdown, see [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM).

---

## Key quantisation concepts for the exam

**Weight-only vs weight-and-activation:**
INT4 AWQ is weight-only — dequantise to BF16 before the matmul, so compute runs at BF16. This reduces memory bandwidth (loading fewer bytes per weight) but the ALU still does BF16 work. FP8 and INT8 SmoothQuant are weight-and-activation — the matmul itself runs at reduced precision, exploiting FP8/INT8 tensor cores for higher compute throughput.

**Calibration:**
Post-training quantisation (PTQ) with FP8 or INT8 requires a calibration dataset to compute per-tensor scaling factors. The calibration set should be representative of the deployment distribution; 128–512 samples is typically sufficient for scaling factor convergence. Quantisation-aware training (QAT) fine-tunes with simulated quantisation, producing better accuracy at the cost of a training run — not covered in this demo.

**KV cache quantisation:**
The `--kv_cache_dtype fp8` and `--kv_cache_dtype int8` flags in `quantize.py` quantise the KV cache independently from the weights. This halves the KV cache footprint (from BF16 to 8-bit), extending achievable context lengths or batch sizes within the same VRAM budget. See [notes/08_inference_optimisation.md](../../notes/08_inference_optimisation.md) for the KV cache memory formula.

**Engine portability:**
A compiled `.engine` is not portable across GPU architectures (sm_86 ≠ sm_89) or across TensorRT major versions. Always rebuild the engine when upgrading the container or moving to a different GPU. This is a practical constraint in multi-GPU deployments — NIM abstracts this away by shipping pre-built engines per hardware profile.

---

## Official references

- TensorRT-LLM quantisation examples: `github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization`
- TensorRT-LLM Llama examples: `github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama`
- TensorRT-LLM Qwen examples: `github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/qwen`
- LLM Python API quickstart: `github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llm-api`
- NVIDIA Model Optimizer (modelopt) docs: `https://nvidia.github.io/TensorRT-Model-Optimizer/`

---

## Common issues

**"CUDA driver version is insufficient for CUDA runtime version":**
The container requires a newer driver than is installed. Run `nvidia-smi` to check your driver version, then upgrade before pulling the container.

**`trtllm-build` command not found:**
You are not inside the container. Run `docker compose run --rm trtllm bash` first.

**Out of memory during engine build:**
Reduce `--max_batch_size` or `--max_seq_len`. The build process allocates scratch buffers proportional to these values.

**Engine loads but generates garbled output:**
Most likely a tokeniser mismatch — ensure the HF model directory used for quantisation is the same one you pass to the inference script. The engine and tokeniser are not bundled together; they must be kept in sync.

**Calibration takes very long:**
Calibration runs a forward pass over the calibration set. 128 samples at batch=1 for a 1B model takes 1–5 minutes. If it hangs, check GPU utilisation with `nvidia-smi`; a stall may indicate a PCIe bandwidth bottleneck loading the model weights.

---

## Cross-references

- [notes/08_inference_optimisation.md](../../notes/08_inference_optimisation.md) — KV cache formula, quantisation trade-offs, in-flight batching, FP8 vs INT8 vs INT4.
- [notes/10_nvidia_software_stack.md](../../notes/10_nvidia_software_stack.md) — TensorRT-LLM component boundaries, Triton backend relationship, NIM packaging.
- [Exercise 04](../04_triton_serving_demo/README.md) — Triton serving layer, which can host TRT-LLM engines once built.
- [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM) — detailed TRT-LLM coverage including benchmarks and architecture.

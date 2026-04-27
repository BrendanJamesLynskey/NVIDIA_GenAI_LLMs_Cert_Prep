#!/usr/bin/env bash
# NOTE: written but not hardware-verified — smoke-test before use
#
# build_engine.sh
# ---------------
# Annotated script to quantise a Llama-3.2-1B checkpoint and build a
# TensorRT-LLM engine.  Run this INSIDE the TRT-LLM container:
#
#   docker compose run --rm trtllm bash
#   bash /workspace/scripts/build_engine.sh [fp8|int8sq|int4awq]
#
# The first argument selects the quantisation format:
#   fp8     — FP8 weight+activation (RTX 4000 Ada, sm_89 only)
#   int8sq  — INT8 SmoothQuant weight+activation (RTX 3080 sm_86 or Ada)
#   int4awq — INT4 AWQ weight-only (any NVIDIA GPU)
#
# All paths are inside the container mount.
# Host directory layout (see docker-compose.yml volumes):
#   ./models   →  /workspace/models
#   ./engines  →  /workspace/engines
#   ./scripts  →  /workspace/scripts

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuration — edit these if using a different model or path
MODEL_DIR="/workspace/models/Llama-3.2-1B"     # HF checkpoint directory
QUANT_FORMAT="${1:-fp8}"                         # default fp8; override via arg
CALIB_SAMPLES=128                                # calibration set size for PTQ
DTYPE="bfloat16"                                 # base dtype for weight loading
MAX_BATCH_SIZE=8                                 # max concurrent sequences
MAX_INPUT_LEN=2048                               # max prompt tokens
MAX_SEQ_LEN=3072                                 # max prompt + generation tokens

# Derived paths
QUANT_CHECKPOINT_DIR="/workspace/models/Llama-3.2-1B-${QUANT_FORMAT}-checkpoint"
ENGINE_DIR="/workspace/engines/Llama-3.2-1B-${QUANT_FORMAT}"

# Location of quantize.py inside the TRT-LLM container
QUANTIZE_SCRIPT="/usr/local/lib/python3.10/dist-packages/tensorrt_llm/examples/quantization/quantize.py"

# --------------------------------------------------------------------------- #
echo "==================================================================="
echo " TensorRT-LLM Engine Build"
echo "   Model:       ${MODEL_DIR}"
echo "   Quant format: ${QUANT_FORMAT}"
echo "   Output:      ${ENGINE_DIR}"
echo "==================================================================="
echo ""

# --------------------------------------------------------------------------- #
# Step 1 — Verify the model checkpoint exists
echo "--- Step 1: Verify model checkpoint ---"
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: Model directory not found: ${MODEL_DIR}"
  echo "Run the huggingface-cli download step from the README on the host first."
  exit 1
fi
echo "Model checkpoint found at ${MODEL_DIR}"
echo ""

# --------------------------------------------------------------------------- #
# Step 2 — Quantise the checkpoint
#
# quantize.py wraps NVIDIA Model Optimizer (nvidia-modelopt).  It:
#  1. Loads the HF checkpoint into a PyTorch model.
#  2. Runs a forward pass over the calibration set to collect activation
#     statistics (skipped for weight-only formats like int4_awq).
#  3. Computes per-tensor / per-channel scaling factors.
#  4. Saves a TRT-LLM-compatible quantised checkpoint with the scale tensors
#     embedded alongside the quantised weights.
#
# The --kv_cache_dtype flag quantises the KV cache independently from the
# weights.  fp8 KV cache halves the KV cache VRAM cost vs BF16, enabling
# longer contexts or larger batch sizes at the same memory budget.
echo "--- Step 2: Quantise checkpoint (format: ${QUANT_FORMAT}) ---"

case "${QUANT_FORMAT}" in
  fp8)
    # FP8 (W8A8): weight-and-activation, 8-bit float.
    # Ada Lovelace (sm_89) and Hopper (sm_90) only — do NOT run on RTX 3080.
    # --qformat fp8        : select FP8 as the quantisation algorithm
    # --kv_cache_dtype fp8 : also quantise KV cache tensors to FP8
    # --calib_size         : number of calibration samples for scale-factor estimation
    echo "NOTE: FP8 requires sm_89 (RTX 4000 Ada) or sm_90 (H100). Will error on RTX 3080."
    python "${QUANTIZE_SCRIPT}" \
      --model_dir "${MODEL_DIR}" \
      --qformat fp8 \
      --kv_cache_dtype fp8 \
      --output_dir "${QUANT_CHECKPOINT_DIR}" \
      --calib_size "${CALIB_SAMPLES}" \
      --dtype "${DTYPE}"
    ;;

  int8sq)
    # INT8 SmoothQuant (W8A8): weight-and-activation INT8.
    # Uses SmoothQuant to migrate quantisation difficulty from activations to
    # weights via a per-channel scale.  Runs on Ampere INT8 tensor cores.
    # --qformat int8_sq    : SmoothQuant algorithm
    # --kv_cache_dtype int8: quantise KV cache to INT8 (~50% VRAM saving vs BF16)
    python "${QUANTIZE_SCRIPT}" \
      --model_dir "${MODEL_DIR}" \
      --qformat int8_sq \
      --kv_cache_dtype int8 \
      --output_dir "${QUANT_CHECKPOINT_DIR}" \
      --calib_size "${CALIB_SAMPLES}" \
      --dtype "${DTYPE}"
    ;;

  int4awq)
    # INT4 AWQ (W4A16): weight-only, 4-bit.
    # Dequantises weights to BF16 before the matmul — saves memory bandwidth
    # but the compute runs at BF16.  No activation quantisation, so no
    # calibration is strictly needed, but --calib_size is used for the AWQ
    # scale search (activation-awareness).
    # --qformat int4_awq   : AWQ algorithm
    # No --kv_cache_dtype  : KV cache remains at BF16 (default)
    python "${QUANTIZE_SCRIPT}" \
      --model_dir "${MODEL_DIR}" \
      --qformat int4_awq \
      --output_dir "${QUANT_CHECKPOINT_DIR}" \
      --calib_size "${CALIB_SAMPLES}" \
      --dtype "${DTYPE}"
    ;;

  *)
    echo "ERROR: Unknown quant format '${QUANT_FORMAT}'. Choose: fp8 | int8sq | int4awq"
    exit 1
    ;;
esac

echo "Quantised checkpoint written to: ${QUANT_CHECKPOINT_DIR}"
echo ""

# --------------------------------------------------------------------------- #
# Step 3 — Build the TensorRT-LLM engine
#
# trtllm-build compiles the quantised checkpoint into a GPU-specific engine.
# The engine contains:
#   - Fused CUDA kernels selected for the target architecture
#   - Pre-computed weight tensors (no HF model loading at runtime)
#   - Engine metadata (input/output shapes, batch limits)
#
# Important: the compiled engine is tied to:
#   - The exact GPU architecture (sm_89 != sm_86)
#   - The TensorRT version inside the container
#   - The max_batch_size and max_seq_len values specified here
#   Rebuild the engine whenever any of these change.
echo "--- Step 3: Build TensorRT-LLM engine ---"

case "${QUANT_FORMAT}" in
  fp8)
    # --gemm_plugin fp8            : use FP8 GEMM kernels for matrix multiplications
    # --use_fp8_context_fmha enable: FP8 FlashMHA during the prefill (context) phase
    #                                Requires sm_89+; gives extra throughput on Ada.
    trtllm-build \
      --checkpoint_dir "${QUANT_CHECKPOINT_DIR}" \
      --output_dir "${ENGINE_DIR}" \
      --gemm_plugin fp8 \
      --use_fp8_context_fmha enable \
      --max_batch_size "${MAX_BATCH_SIZE}" \
      --max_input_len "${MAX_INPUT_LEN}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --workers 1
    ;;

  int8sq)
    # --gemm_plugin auto: TRT-LLM selects the best GEMM plugin for the
    #   checkpoint's precision and the current GPU.  For an INT8 SmoothQuant
    #   checkpoint on Ampere/Ada, this picks INT8 tensor-core GEMM.
    trtllm-build \
      --checkpoint_dir "${QUANT_CHECKPOINT_DIR}" \
      --output_dir "${ENGINE_DIR}" \
      --gemm_plugin auto \
      --max_batch_size "${MAX_BATCH_SIZE}" \
      --max_input_len "${MAX_INPUT_LEN}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --workers 1
    ;;

  int4awq)
    # INT4 AWQ weight-only: weights are stored at INT4 and dequantised to
    # BF16 before the GEMM, so --gemm_plugin auto will select BF16 GEMM.
    # The savings come from reduced memory bandwidth during weight loading
    # (4 bits loaded vs 16 bits), not from lower-precision compute.
    trtllm-build \
      --checkpoint_dir "${QUANT_CHECKPOINT_DIR}" \
      --output_dir "${ENGINE_DIR}" \
      --gemm_plugin auto \
      --max_batch_size "${MAX_BATCH_SIZE}" \
      --max_input_len "${MAX_INPUT_LEN}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --workers 1
    ;;
esac

echo ""
echo "Engine written to: ${ENGINE_DIR}"
echo "Contents:"
ls -lh "${ENGINE_DIR}"
echo ""
echo "Build complete. Run inference with:"
echo "  python /workspace/scripts/run_inference.py --engine_dir ${ENGINE_DIR}"

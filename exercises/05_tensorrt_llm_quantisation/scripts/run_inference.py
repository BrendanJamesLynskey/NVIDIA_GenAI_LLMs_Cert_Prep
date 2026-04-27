# NOTE: written but not hardware-verified — smoke-test before use
"""
run_inference.py
----------------
Run inference against a compiled TensorRT-LLM engine and print generated
text plus latency.  Run this INSIDE the TRT-LLM container:

    docker compose run --rm trtllm bash
    python /workspace/scripts/run_inference.py \
        --engine_dir /workspace/engines/Llama-3.2-1B-fp8

Uses the TensorRT-LLM Python high-level LLM API (tensorrt_llm.LLM), which
handles tokenisation, in-flight batching, and paged KV cache internally.

The LLM API is documented at:
    https://nvidia.github.io/TensorRT-LLM/llm-api/
    github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llm-api
"""

import argparse
import time
from pathlib import Path

from tensorrt_llm import LLM, SamplingParams  # available inside TRT-LLM container


DEFAULT_PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "What is the difference between FP8 and INT8 quantisation?",
    "In three sentences, describe how TensorRT-LLM's in-flight batching works.",
]


def run(engine_dir: str, prompts: list[str], max_tokens: int) -> None:
    engine_path = Path(engine_dir)
    if not engine_path.exists():
        raise SystemExit(f"Engine directory not found: {engine_dir}\nRun build_engine.sh first.")

    print(f"Loading TensorRT-LLM engine from: {engine_dir}")
    print("(First load compiles CUDA graphs — may take 30–60 s)")

    # LLM() accepts either a compiled engine directory (produced by trtllm-build)
    # or a Hugging Face model ID / local HF directory (it will build on the fly).
    # Here we pass the pre-built engine for reproducibility and speed.
    llm = LLM(model=engine_dir)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,   # deterministic (greedy) for benchmarking
        top_p=1.0,
    )

    print(f"\nRunning {len(prompts)} prompt(s) …\n")
    print("-" * 70)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_elapsed = time.perf_counter() - t0

    total_tokens = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        n_tokens = len(output.outputs[0].token_ids)
        total_tokens += n_tokens
        print(f"Prompt:    {output.prompt!r}")
        print(f"Generated: {generated_text!r}")
        print(f"Tokens:    {n_tokens}")
        print()

    print("-" * 70)
    print(f"Total prompts:         {len(prompts)}")
    print(f"Total generated tokens: {total_tokens}")
    print(f"Elapsed time:           {total_elapsed * 1000:.1f} ms")
    if total_tokens > 0 and total_elapsed > 0:
        tok_per_s = total_tokens / total_elapsed
        print(f"Throughput:             {tok_per_s:.1f} tok/s  [rough estimate, not measured]")
    print()
    print("NOTE: Latency figures above are wall-clock for the full generate() call.")
    print("They include tokenisation overhead and the first-request CUDA graph warm-up.")
    print("For proper benchmarking, run at least 3 warm-up iterations and report the mean.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRT-LLM inference demo")
    parser.add_argument(
        "--engine_dir",
        default="/workspace/engines/Llama-3.2-1B-fp8",
        help="Path to the trtllm-build output directory",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="One or more prompt strings",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per prompt",
    )
    args = parser.parse_args()
    run(args.engine_dir, args.prompts, args.max_tokens)

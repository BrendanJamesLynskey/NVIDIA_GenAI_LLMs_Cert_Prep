# NOTE: written but not hardware-verified — smoke-test before use
"""
baseline_hf.py
--------------
Run the same prompts through plain Hugging Face transformers (BF16) for an
apples-to-apples latency and quality comparison against the TRT-LLM engine.

Run this on the HOST (not inside the TRT-LLM container) using the host venv:

    source .venv/bin/activate
    python scripts/baseline_hf.py \
        --model_dir ./models/Llama-3.2-1B \
        --prompts "Explain what a KV cache is in one paragraph."

Requirements: transformers, torch, accelerate (see requirements.txt)

The purpose of this script is to:
  1. Provide a quality reference — verify the quantised engine output is
     not meaningfully degraded compared to BF16.
  2. Provide a latency baseline — illustrate the speedup from TRT-LLM
     vs plain PyTorch inference on the same hardware.

Latency figures from this script are labelled "rough estimate, not measured"
because they are single-run wall-clock times, not a proper benchmark.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "What is the difference between FP8 and INT8 quantisation?",
    "In three sentences, describe how TensorRT-LLM's in-flight batching works.",
]


def run(model_dir: str, prompts: list[str], max_new_tokens: int) -> None:
    print(f"Loading model from: {model_dir}")
    print("Precision: BF16 (baseline — no quantisation)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA device found. Running on CPU — latency will be much higher.")

    tokeniser = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",    # auto-places layers across available devices
    )
    model.eval()

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokeniser,
        device_map="auto",
    )

    print(f"\nRunning {len(prompts)} prompt(s) …\n")
    print("-" * 70)

    total_tokens = 0
    t0 = time.perf_counter()

    for prompt in prompts:
        t_prompt = time.perf_counter()
        result = gen(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,    # greedy, matches TRT-LLM run with temperature=0
            return_full_text=False,
        )
        elapsed_prompt_ms = (time.perf_counter() - t_prompt) * 1000

        generated_text = result[0]["generated_text"]
        # Count tokens in generated portion
        n_tokens = len(tokeniser(generated_text, add_special_tokens=False)["input_ids"])
        total_tokens += n_tokens

        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print(f"Tokens:    {n_tokens}")
        print(f"Latency:   {elapsed_prompt_ms:.1f} ms  [rough estimate, not measured]")
        print()

    total_elapsed = time.perf_counter() - t0

    print("-" * 70)
    print(f"Total prompts:          {len(prompts)}")
    print(f"Total generated tokens: {total_tokens}")
    print(f"Elapsed time:           {total_elapsed * 1000:.1f} ms")
    if total_tokens > 0 and total_elapsed > 0:
        tok_per_s = total_tokens / total_elapsed
        print(f"Throughput:             {tok_per_s:.1f} tok/s  [rough estimate, not measured]")

    print()
    print("Compare these figures to run_inference.py output to assess TRT-LLM speedup.")
    print("For a rigorous comparison, use a proper benchmarking harness with warm-up runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF baseline inference for TRT-LLM comparison")
    parser.add_argument(
        "--model_dir",
        default="./models/Llama-3.2-1B",
        help="Path to the Hugging Face model directory",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="One or more prompt strings",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per prompt",
    )
    args = parser.parse_args()
    run(args.model_dir, args.prompts, args.max_new_tokens)

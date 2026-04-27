# NOTE: written but not hardware-verified — smoke-test before use
"""
finetune.py — Minimal LoRA fine-tune of Qwen2.5-0.5B-Instruct on a small
instruction-following dataset.

Model choice: Qwen2.5-0.5B-Instruct (~1 GB in bf16) fits comfortably on
both the RTX 3080 (10 GB) and RTX 4000 Ada (20 GB) with room to spare for
activations and optimiser states.

The script:
  1. Loads Qwen2.5-0.5B-Instruct in bfloat16.
  2. Wraps it with a LoRA adapter (r=8, alpha=16, target_modules q_proj/v_proj).
  3. Trains for a configurable number of steps on dataset.jsonl.
  4. Saves the adapter weights to ./lora_adapter/.
  5. Prints training loss every LOG_EVERY steps.

Usage:
    python finetune.py                     # default: 50 steps
    python finetune.py --steps 100         # longer run
    python finetune.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 50

The saved adapter can be loaded by infer.py.

Notes on target_modules:
    Qwen2.5 uses the standard causal LM attention naming from Hugging Face
    transformers: q_proj, k_proj, v_proj, o_proj for attention, and
    gate_proj, up_proj, down_proj for the MLP (SwiGLU).  We target q_proj
    and v_proj by default (the original LoRA paper recommendation), which
    gives ~0.8 M trainable parameters for the 0.5B model at r=8.

    To verify module names for a different base model, run:
        python -c "from transformers import AutoModelForCausalLM; \\
                   m = AutoModelForCausalLM.from_pretrained('<model>'); \\
                   print([n for n,_ in m.named_modules()])"
"""

import argparse
import json
import pathlib
import sys
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR     = pathlib.Path(__file__).parent / "lora_adapter"
DATASET_PATH    = pathlib.Path(__file__).parent / "dataset.jsonl"
MAX_SEQ_LEN     = 512
BATCH_SIZE      = 1           # gradient accumulation used to simulate larger batches
GRAD_ACCUM      = 4           # effective batch size = BATCH_SIZE * GRAD_ACCUM
LEARNING_RATE   = 3e-4
LOG_EVERY       = 10          # print loss every N optimiser steps

# LoRA hyperparameters — see README.md for explanation of each.
LORA_R          = 8           # adapter rank
LORA_ALPHA      = 16          # scaling factor alpha; effective scale = alpha / r = 2.0
LORA_DROPOUT    = 0.05        # dropout applied to the low-rank matrices during training
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # attention projections to adapt


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(path: pathlib.Path) -> List[Dict[str, str]]:
    """Load instruction/response pairs from a JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_prompt(record: Dict[str, str], tokenizer) -> str:
    """
    Format an instruction/response pair using the model's chat template.

    Qwen2.5-Instruct uses the ChatML template:
        <|im_start|>system\n...<|im_end|>\n
        <|im_start|>user\n{instruction}<|im_end|>\n
        <|im_start|>assistant\n{response}<|im_end|>

    We apply the tokeniser's built-in apply_chat_template if available,
    otherwise fall back to a plain format.
    """
    messages = [
        {"role": "user",      "content": record["instruction"]},
        {"role": "assistant", "content": record["response"]},
    ]
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    # Fallback format.
    return (
        f"User: {record['instruction']}\n"
        f"Assistant: {record['response']}\n"
    )


def tokenise_example(
    text: str,
    tokenizer,
    max_length: int = MAX_SEQ_LEN,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise a single formatted text string.

    Returns a dict with 'input_ids', 'attention_mask', and 'labels'.
    Labels are the same as input_ids (next-token prediction); padding tokens
    are set to -100 so they are ignored by the cross-entropy loss.
    """
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    input_ids      = enc['input_ids'].squeeze(0)      # (T,)
    attention_mask = enc['attention_mask'].squeeze(0)  # (T,)

    # Mask padding tokens in labels.
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'labels':         labels,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_lora_model(model_name: str, device: torch.device):
    """Load base model and attach LoRA adapter."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",        # do not train biases — keeps adapter small
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model


def train(
    model_name: str = DEFAULT_MODEL,
    num_steps: int  = 50,
    device: torch.device = None,
) -> None:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data.
    records = load_dataset(DATASET_PATH)
    print(f"Dataset: {len(records)} examples")

    tokenizer, model = build_lora_model(model_name, device)
    model.train()

    optimiser = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Pre-tokenise all examples.
    examples = [
        tokenise_example(format_prompt(r, tokenizer), tokenizer)
        for r in records
    ]

    # Training loop — cycles through examples if num_steps > len(examples).
    n = len(examples)
    micro_step  = 0   # counts forward passes
    opt_step    = 0   # counts optimiser updates
    accum_loss  = 0.0
    optimiser.zero_grad()

    print(f"\nTraining for {num_steps} optimiser steps "
          f"(gradient accumulation = {GRAD_ACCUM})...")

    while opt_step < num_steps:
        example = examples[micro_step % n]

        input_ids      = example['input_ids'].unsqueeze(0).to(device)
        attention_mask = example['attention_mask'].unsqueeze(0).to(device)
        labels         = example['labels'].unsqueeze(0).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        accum_loss += loss.item()

        micro_step += 1

        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimiser.step()
            optimiser.zero_grad()
            opt_step += 1

            if opt_step % LOG_EVERY == 0 or opt_step == 1:
                print(f"  step {opt_step:>4}/{num_steps}  loss={accum_loss:.4f}")

            accum_loss = 0.0

    # Save adapter.
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"\nAdapter saved to: {ADAPTER_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Minimal LoRA fine-tune.")
    parser.add_argument(
        '--model', default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        '--steps', type=int, default=50,
        help="Number of optimiser steps (default: 50)"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(model_name=args.model, num_steps=args.steps)

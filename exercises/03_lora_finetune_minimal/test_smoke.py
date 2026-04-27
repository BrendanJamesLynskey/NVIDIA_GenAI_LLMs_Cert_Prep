# NOTE: written but not hardware-verified — smoke-test before use
"""
test_smoke.py — pytest smoke test for the LoRA fine-tune pipeline.

Marks: @pytest.mark.slow — this test loads a ~1 GB model and is expected to
take 1–3 minutes on an RTX 3080 (rough estimate; not hardware-measured).

Run all tests:
    pytest test_smoke.py -v -m slow

Run only fast tests (skips this file's tests):
    pytest exercises/ -v -m "not slow"

What this test asserts:
  1. The training loop runs without error for 2 optimiser steps.
  2. The loss at step 2 is a finite positive float (not NaN, not inf).
  3. The model produces at least 1 non-empty output token (generation does not crash).
  4. The adapter saves to disk without error.
"""

import json
import pathlib
import shutil
import tempfile

import pytest
import torch


# ---------------------------------------------------------------------------
# Skip if transformers / peft not installed.
# ---------------------------------------------------------------------------

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

MODEL_NAME      = "Qwen/Qwen2.5-0.5B-Instruct"
SMOKE_STEPS     = 2       # minimal: just enough to verify the loop runs
MAX_SEQ_LEN     = 128     # shorter sequence for speed
LORA_R          = 8
LORA_ALPHA      = 16
LORA_TARGET     = ["q_proj", "v_proj"]
DATASET_PATH    = pathlib.Path(__file__).parent / "dataset.jsonl"


# ---------------------------------------------------------------------------
# Helpers (duplicated minimally from finetune.py to keep the test self-contained)
# ---------------------------------------------------------------------------

def _load_model(device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,   # disable dropout for deterministic smoke test
        target_modules=LORA_TARGET,
        bias="none",
    )
    model = peft.get_peft_model(model, lora_config)
    return tokenizer, model


def _tokenise(text, tokenizer):
    enc = tokenizer(
        text,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    input_ids      = enc['input_ids'].squeeze(0)
    attention_mask = enc['attention_mask'].squeeze(0)
    labels         = input_ids.clone()
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_training_loop_and_generation():
    """
    Run 2 training steps and 1 generation call.

    Asserts:
      - Loss after step 2 is finite and positive.
      - Generation produces at least 1 token without raising.
      - Adapter can be saved and the saved directory contains adapter_model files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = _load_model(device)
    model.train()

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
    )

    # Load one example from the dataset.
    assert DATASET_PATH.exists(), f"dataset.jsonl not found at {DATASET_PATH}"
    with open(DATASET_PATH) as f:
        record = json.loads(f.readline())

    text = f"User: {record['instruction']}\nAssistant: {record['response']}"
    input_ids, attention_mask, labels = _tokenise(text, tokenizer)
    input_ids      = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    labels         = labels.unsqueeze(0).to(device)

    final_loss = None
    for step in range(SMOKE_STEPS):
        optimiser.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimiser.step()
        final_loss = loss.item()

    # Assert loss is finite and positive.
    assert final_loss is not None
    assert torch.isfinite(torch.tensor(final_loss)), (
        f"Loss is not finite after {SMOKE_STEPS} steps: {final_loss}"
    )
    assert final_loss > 0.0, f"Loss should be positive, got {final_loss}"

    # Assert generation runs without error.
    model.eval()
    prompt = "What is a transformer?"
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors='pt').to(device)
        gen_ids = model.generate(
            **enc,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = gen_ids[0][enc['input_ids'].shape[1]:]
    assert len(new_ids) >= 1, "Generation produced no output tokens."

    # Assert adapter saves without error.
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        saved_files = list(pathlib.Path(tmpdir).iterdir())
        assert any('adapter' in f.name for f in saved_files), (
            f"No adapter files found in {tmpdir}: {[f.name for f in saved_files]}"
        )

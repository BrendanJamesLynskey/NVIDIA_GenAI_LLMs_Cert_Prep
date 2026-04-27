# 03 — Minimal LoRA Fine-Tune

This exercise runs an end-to-end PEFT pipeline on a small open model: load Qwen2.5-0.5B-Instruct in bfloat16, wrap it with a LoRA adapter, train for a configurable number of steps on a handcrafted instruction-following dataset, save the adapter, and generate responses using the saved adapter. The goal is to make every component of the fine-tuning loop concrete — data formatting, adapter configuration, the training step, gradient accumulation, and adapter serialisation — before using higher-level wrappers such as TRL's `SFTTrainer`. Cross-reference: [notes/06\_fine\_tuning\_and\_peft.md](../../notes/06_fine_tuning_and_peft.md) and [FT\_02\_LoRA\_and\_PEFT\_Variants](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants).

---

## Hardware requirements

| Card | VRAM | Status |
|---|---|---|
| RTX 3080 (10 GB GDDR6X, Ampere sm\_86) | 10 GB | Comfortable for Qwen2.5-0.5B in bf16 (~1 GB base + activations) |
| RTX 4000 Ada (20 GB GDDR6, Ada sm\_89) | 20 GB | Comfortable; also supports QLoRA on 7B models (see scaling section) |

Minimum: any GPU with 4 GB VRAM and CUDA compute capability 7.0 or higher. CPU-only training is supported but will be slow (rough estimate: 10–20× slower than GPU, not hardware-measured).

**VRAM estimate for this exercise** (from the formula in [notes/06\_fine\_tuning\_and\_peft.md](../../notes/06_fine_tuning_and_peft.md)):

- Qwen2.5-0.5B parameters: ~500 M × 2 bytes (bf16) ≈ 1 GB for weights.
- LoRA adapter (r=8, q\_proj + v\_proj): ~0.8 M trainable parameters × 4 bytes (fp32 grads) + AdamW moments ≈ 50 MB.
- Activations at MAX\_SEQ\_LEN=512: small for a 0.5B model.
- Total: well under 4 GB. This exercise is deliberately sized for broad accessibility.

---

## LoRA hyperparameters

| Hyperparameter | Value in this exercise | What it controls |
|---|---|---|
| `r` (rank) | 8 | Rank of the low-rank matrices A and B. Controls expressiveness and trainable parameter count. Higher r = more capacity, more parameters. Typical range: 4–64. |
| `lora_alpha` | 16 | Scaling factor. The effective weight applied to the adapter update is `alpha / r` = 2.0. Setting `alpha = 2r` is a common convention that keeps the effective scale stable as r varies. |
| `target_modules` | `["q_proj", "v_proj"]` | Which linear layers receive LoRA adapters. Adapting query and value projections (the original LoRA paper recommendation) gives the best performance per trainable parameter for most instruction-tuning tasks. |
| `lora_dropout` | 0.05 | Dropout rate applied to the low-rank matrices during training. Acts as a regulariser; helps prevent overfitting on small datasets. Set to 0.0 for deterministic smoke tests. |
| `bias` | `"none"` | Whether to train bias parameters. `"none"` keeps the adapter minimal; `"all"` or `"lora_only"` trains biases too (marginal effect on small datasets). |

**Rank-alpha interaction** (from [notes/06\_fine\_tuning\_and\_peft.md](../../notes/06_fine_tuning_and_peft.md)): the scaling applied to the weight update is `alpha / r`. If you double r and hold alpha constant, the effective learning rate of the adapter halves. If you want to increase capacity (double r) without changing the effective learning rate, also double alpha.

---

## File layout

| File | Purpose |
|---|---|
| `finetune.py` | Training script: loads model, applies LoRA, trains, saves adapter |
| `infer.py` | Inference script: loads saved adapter, generates responses |
| `dataset.jsonl` | 20 handcrafted instruction/response pairs on transformer and PEFT topics |
| `test_smoke.py` | pytest smoke test (marked `@pytest.mark.slow`) |
| `requirements.txt` | `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `pytest` |

---

## Setup

```bash
cd exercises/03_lora_finetune_minimal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**First-run download.** `finetune.py` downloads Qwen2.5-0.5B-Instruct (~1 GB) from the Hugging Face Hub on first run. Subsequent runs use the local cache (`~/.cache/huggingface/`). An internet connection is required for the first run.

---

## Run

**Training (default 50 optimiser steps, rough estimate: 2–5 minutes on RTX 3080):**

```bash
python finetune.py
```

**Longer training run:**

```bash
python finetune.py --steps 200
```

**Inference (requires completed training run):**

```bash
python infer.py
```

**Smoke test (loads model — allow 1–3 minutes, rough estimate):**

```bash
pytest test_smoke.py -v -m slow
```

---

## Expected output

`finetune.py`:

```
Device: cuda
Dataset: 20 examples
trainable params: 786,432 || all params: 494,476,288 || trainable%: 0.1590
Training for 50 optimiser steps (gradient accumulation = 4)...
  step    1/50  loss=2.xxxx
  step   10/50  loss=1.xxxx
  step   20/50  loss=0.xxxx
  ...
Adapter saved to: ./lora_adapter
```

Loss should decrease over 50 steps on this small dataset. A final loss below 1.0 indicates the adapter is fitting the training examples. Overfitting on 20 examples is expected and expected — the purpose is pipeline verification, not generalisation.

`pytest test_smoke.py -v -m slow`:

```
test_smoke.py::test_training_loop_and_generation PASSED
```

---

## Scaling up: QLoRA on a 7B model on RTX 4000 Ada

The RTX 4000 Ada (20 GB) can run QLoRA on a 7B model. Pseudo-configuration:

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — optimal for normal weight distributions
    bnb_4bit_use_double_quant=True,   # quantise the quantisation constants too (~0.5 bits/param saving)
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# Enable gradient checkpointing to halve activation memory.
model.gradient_checkpointing_enable()
model.enable_input_require_grads()  # required with gradient checkpointing + PEFT
```

**VRAM estimate for 7B QLoRA on RTX 4000 Ada:**
- 4-bit base model: ~7B × 0.5 bytes ≈ 3.5–4 GB.
- LoRA adapter (r=16, all attention + MLP layers): ~50 M trainable parameters × 4 bytes × 3 (grad + 2 AdamW moments) ≈ 600 MB.
- Activations with gradient checkpointing at sequence length 2048: ~1–2 GB.
- Total: approximately 6–8 GB — comfortably within 20 GB.

Use `bitsandbytes` and `accelerate` for the 4-bit base model; both are already in `requirements.txt`. The `paged_adamw_8bit` optimiser from `bitsandbytes` spills optimiser states to CPU RAM during memory pressure spikes, preventing OOM during gradient accumulation.

See [notes/06\_fine\_tuning\_and\_peft.md](../../notes/06_fine_tuning_and_peft.md) for the full VRAM table and [FT\_02\_LoRA\_and\_PEFT\_Variants](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants) for QLoRA depth.

---

## What to study from this

- **The LoRA wrapper is a two-line change.** `LoraConfig` + `get_peft_model(model, config)` injects adapter matrices and freezes the base model. Everything else (training loop, loss computation, optimiser) is unchanged.
- **target\_modules must match the model's layer names exactly.** Qwen2.5 uses `q_proj`, `v_proj`, etc. Other model families use different names (e.g. `query_key_value` for Falcon, `c_attn` for GPT-2). The NOTE in `finetune.py` shows how to inspect module names.
- **Adapter file size vs full model size.** The saved `lora_adapter/` directory is approximately 3–6 MB for r=8 on a 0.5B model, compared to ~1 GB for the full base model. This is the key practical advantage for multi-task serving.
- **Gradient accumulation simulates larger batch sizes.** With BATCH\_SIZE=1 and GRAD\_ACCUM=4, gradients from 4 forward passes accumulate before an optimiser step, giving an effective batch size of 4 while keeping per-step GPU memory constant.
- **Chat templates matter.** Fine-tuning must use the same chat template as the base model's instruction-tuning. Qwen2.5-Instruct uses ChatML; using a different format at training time degrades the model's ability to follow instructions at inference.

---

## Further reading

- [notes/06\_fine\_tuning\_and\_peft.md](../../notes/06_fine_tuning_and_peft.md) — VRAM budgeting, LoRA/QLoRA/DoRA, tooling landscape
- [FT\_02\_LoRA\_and\_PEFT\_Variants](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants) — LoRA, QLoRA, DoRA, IA³ depth
- Hu et al. (2021), LoRA paper: <https://arxiv.org/abs/2106.09685>
- Dettmers et al. (2023), QLoRA paper: <https://arxiv.org/abs/2305.14314>
- PEFT documentation: <https://huggingface.co/docs/peft>
- Qwen2.5 model card: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>

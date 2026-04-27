# NOTE: written but not hardware-verified — smoke-test before use
"""
infer.py — Load the saved LoRA adapter and generate responses to a few prompts.

Run after finetune.py has saved an adapter to ./lora_adapter/:

    python infer.py

The script loads the base model in bfloat16, attaches the saved adapter using
PEFT's PeftModel, and runs greedy decoding on a short list of test prompts.

The adapter is kept separate from the base model (not merged), which is the
default inference mode.  Optionally, the adapter can be merged into the base
weights for zero-overhead inference; see the commented-out section at the end.
"""

import pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel


ADAPTER_DIR   = pathlib.Path(__file__).parent / "lora_adapter"
MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.0   # 0.0 = greedy decoding (deterministic)


# Test prompts — these are held-out from the training set.
TEST_PROMPTS = [
    "What is the residual stream in a transformer?",
    "Why is scaling by 1/sqrt(d_k) important in attention?",
    "What is the difference between LoRA and QLoRA?",
]


def load_model_with_adapter(adapter_dir: pathlib.Path, device: torch.device):
    """
    Load the base model from the adapter directory's config, then attach
    the adapter on top.

    PEFT saves a config.json in the adapter directory that records the base
    model name, so we do not need to hard-code it here.
    """
    import json
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found at {adapter_dir}.\n"
            "Run finetune.py first to train and save the adapter."
        )

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path", "")
    print(f"Base model  : {base_model_name}")
    print(f"Adapter dir : {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return tokenizer, model


def generate(
    prompt: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Format the prompt using the chat template and generate a response."""
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(formatted, return_tensors='pt').to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decoding
            temperature=1.0,          # ignored when do_sample=False
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the input prompt).
    new_ids = output_ids[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    tokenizer, model = load_model_with_adapter(ADAPTER_DIR, device)

    print("=" * 60)
    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        response = generate(prompt, tokenizer, model, device)
        print(response)
        print("-" * 40)

    # -----------------------------------------------------------------------
    # Optional: merge adapter into base model weights for zero-overhead inference.
    # Uncomment to use.
    # -----------------------------------------------------------------------
    # merged_dir = pathlib.Path(__file__).parent / "merged_model"
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(str(merged_dir))
    # tokenizer.save_pretrained(str(merged_dir))
    # print(f"\nMerged model saved to {merged_dir}")


if __name__ == '__main__':
    main()

# CLAUDE.md

Repo-local conventions for the NVIDIA GenAI LLMs cert-prep build. Inherits from the parent [`Claude_sandbox/CLAUDE.md`](../CLAUDE.md).

---

## Inherited rules (summary)

1. Match existing style before creating new files — formatting, naming conventions, file extensions, comment style.
2. After creating new files, update `README.md` to include a link.
3. After changes, commit and push with a clear, concise commit message.

---

## Repo-specific rules

### Cross-reference, do not duplicate

Where a topic is already covered in depth by an existing `BrendanJamesLynskey/*` repo, link to it from the relevant `notes/*.md` file and do not re-derive the content. The cert-specific value-add in this repo is: domain mapping, mock questions, exercises, and synthesis — not yet another walkthrough of LoRA or attention.

Relevant existing hubs and series to check first:

- NVIDIA stack: `LLM_Hub_NVIDIA_GPUs`, `NVIDIA_GPU_19_TensorRT_LLM`, `NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise`
- Transformer architecture: `LLM_Hub_Transformer_Architecture`
- Fine-tuning / PEFT: `LLM_Hub_Fine_Tuning`, `FT_01` – `FT_05`
- RAG: `LLM_Hub_RAG_Retrieval`, `RAG_01` – `RAG_07`
- Evaluations: `LLM_Hub_Evaluations`, `LLM_Eval_01` – `LLM_Eval_05`
- Safety / alignment: `LLM_Hub_Safety_Alignment`, `Safety_01` – `Safety_02`
- LLMOps: `LLM_Hub_LLMOps`
- Local hosting / inference: `LLM_Hub_Local_LLM_Hosting`
- CUDA: `LLM_Hub_CUDA`

### Hardware assumptions

All exercises and code are written for:

- RTX 3080 (10 GB GDDR6X) — Linux desktop, native CUDA
- RTX 4000 Ada (20 GB GDDR6) — workstation under WSL2

Do not write code that assumes H100s, DGX, or any cloud GPU. If an exercise genuinely requires more memory than is available locally, document a scaled-down alternative that runs on the hardware above, or note the constraint explicitly.

TensorRT-LLM and Triton exercises use Docker. Document the exact image tag and `docker run` invocation.

### Hardware verification

Exercise code in this repo may be written by an agent before it has been run on hardware. Mark any such file with a comment at the top:

```
# NOTE: written but not hardware-verified — smoke-test before use
```

Smoke-testing before pushing exercise updates that change runtime behaviour is Brendan's responsibility.

### Cert facts require an official source

Domain weightings, exam length, question count, price, validity period, and prerequisites must be sourced from the NVIDIA Learn certification pages:

- https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/
- https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/

If a fact cannot be verified from those pages (e.g., the pages have moved or returned an error), mark it `TODO: verify from official NVIDIA cert page` with the URL that was tried. Do not guess or invent numbers.

### Notes file naming

`notes/` files are numbered to match the NCA domain order for files 01–04, then extended topics 05–10:

```
01_ml_neural_network_fundamentals.md
02_transformer_architecture.md
03_prompt_engineering.md
04_alignment_and_trustworthy_ai.md
05_rag_systems.md
06_fine_tuning_and_peft.md
07_distributed_training.md
08_inference_optimisation.md
09_evaluation_and_metrics.md
10_nvidia_software_stack.md
```

Do not renumber existing files. Add new files with the next available number.

### Exercise naming

`exercises/` subdirectories follow the pattern `NN_short_description/`. Each must contain a `README.md` explaining the goal, hardware requirements, setup steps, and expected output.

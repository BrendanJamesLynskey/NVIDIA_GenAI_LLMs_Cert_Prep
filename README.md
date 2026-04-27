# NVIDIA GenAI LLMs Certification Prep

Study and interview-prep portfolio for NVIDIA's two generative-AI LLM credentials: NCA-GENL (Associate) and NCP-GENL (Professional). The repo exists primarily for Brendan's own exam preparation — it is public because the structure and cross-references may be useful to anyone studying for the same exams.

The value-add here is the cert-specific synthesis: syllabus mapping, exercises tied to official domain weightings, mock-question batches, and a structured study plan. Topics already covered in depth by existing repos in the portfolio are cross-referenced rather than re-derived.

---

## Certifications at a glance

| Field | NCA-GENL | NCP-GENL |
| --- | --- | --- |
| Full title | Generative AI LLMs — Associate | Generative AI LLMs — Professional |
| Exam code | NCA-GENL | NCP-GENL |
| Level | Associate | Professional |
| Duration | 60 minutes | 120 minutes |
| Questions | 50–60 multiple-choice | 60–70 multiple-choice |
| Format | Online, remotely proctored | Online, remotely proctored |
| Price | USD 125 | USD 200 |
| Validity | 2 years | 2 years |
| Prerequisites | Basic understanding of generative AI and LLMs | 2–3 years practical AI/ML experience with LLMs |

Source: [NVIDIA Learn Certification hub](https://www.nvidia.com/en-us/learn/certification/) — individual cert pages at `/generative-ai-llm-associate/` and `/generative-ai-llm-professional/`.

---

## Domain weightings

### NCA-GENL (Associate)

| Domain | Weight |
| --- | --- |
| Core Machine Learning and AI Knowledge | 30% |
| Software Development | 24% |
| Experimentation | 22% |
| Data Analysis and Visualisation | 14% |
| Trustworthy AI | 10% |

### NCP-GENL (Professional)

| Domain | Weight |
| --- | --- |
| Model Optimisation | 17% |
| GPU Acceleration and Optimisation | 14% |
| Prompt Engineering | 13% |
| Fine-Tuning | 13% |
| Data Preparation | 9% |
| Model Deployment | 9% |
| Evaluation | 7% |
| Production Monitoring and Reliability | 7% |
| LLM Architecture | 6% |
| Safety, Ethics, and Compliance | 5% |

---

## Hardware used

Exercises in this repo are written for the following hardware:

| Machine | GPU | OS / environment |
| --- | --- | --- |
| Linux desktop | RTX 3080 (10 GB GDDR6X) | Ubuntu, native CUDA |
| Workstation | RTX 4000 Ada (20 GB GDDR6) | Windows + WSL2 |

Exercises requiring TensorRT-LLM or Triton are Docker-based and have been smoke-tested locally; the code in this repo is written by an agent and has not been hardware-verified end-to-end. Verification is Brendan's responsibility before pushing exercise updates that change behaviour. No H100, DGX, or cloud-GPU access is assumed.

---

## Repository contents

| Path | Description |
| --- | --- |
| [notes/](notes/) | Per-domain topic notes — one file per NCA/NCP domain area |
| [exercises/](exercises/) | Hands-on coding exercises matched to domain weightings |
| [exercises/01_tokeniser_from_scratch/](exercises/01_tokeniser_from_scratch/) | BPE tokeniser implemented from scratch in pure Python — train, encode, decode, round-trip tests |
| [exercises/02_attention_from_scratch/](exercises/02_attention_from_scratch/) | Scaled dot-product and multi-head attention in NumPy and PyTorch, with correctness checks against `torch.nn.functional` |
| [exercises/03_lora_finetune_minimal/](exercises/03_lora_finetune_minimal/) | End-to-end LoRA fine-tune on Qwen2.5-0.5B-Instruct — adapter training, inference, and smoke test |
| [exercises/04_triton_serving_demo/](exercises/04_triton_serving_demo/) | Docker-compose Triton Inference Server demo — ONNX model, Python client, metrics walkthrough |
| [exercises/05_tensorrt_llm_quantisation/](exercises/05_tensorrt_llm_quantisation/) | TensorRT-LLM engine build walkthrough — FP8 (RTX 4000 Ada) and INT8/INT4 (RTX 3080) quantisation |
| [mock\_interviews/](mock_interviews/) | Mock question banks for NCA-GENL and NCP-GENL, plus system-design and behavioural sets |
| [cheatsheets/](cheatsheets/) | Concise one-pager cheatsheets — transformer maths, quantisation, sampling, NVIDIA stack |
| [presentations/](presentations/) | Slide decks for five key topic areas |
| [diagrams/](diagrams/) | Architecture diagrams referenced from notes |
| [SYLLABUS.md](SYLLABUS.md) | Both cert tiers mapped row-by-row to notes files, exercises, and cross-reference repos |
| [STUDY\_PLAN.md](STUDY_PLAN.md) | 12-week study plan — NCA weeks 1–6, NCP weeks 7–12, ~6 hours/week |

---

## How this fits the rest of the portfolio

The cert content does not exist in isolation. The table below maps each major domain to the existing repo that already covers the depth — this repo adds the cert-specific framing and exercises on top of that prior work.

| Topic area | Existing repo |
| --- | --- |
| NVIDIA GPU architectures, NeMo, NIM, TensorRT-LLM | [LLM\_Hub\_NVIDIA\_GPUs](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs) (index of 37 presentations) |
| TensorRT-LLM depth | [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM) |
| NeMo, NIM, AI Enterprise | [NVIDIA\_GPU\_20\_NeMo\_NIM\_AI\_Enterprise](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise) |
| Tensor cores | [NVIDIA\_GPU\_03\_Tensor\_Cores](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_03_Tensor_Cores) |
| Memory hierarchy | [NVIDIA\_GPU\_04\_Memory\_Hierarchy](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_04_Memory_Hierarchy) |
| Transformer architecture | [LLM\_Hub\_Transformer\_Architecture](https://github.com/BrendanJamesLynskey/LLM_Hub_Transformer_Architecture) |
| RAG and retrieval | [LLM\_Hub\_RAG\_Retrieval](https://github.com/BrendanJamesLynskey/LLM_Hub_RAG_Retrieval) (hub) + RAG\_01 – RAG\_07 |
| Fine-tuning and PEFT | [LLM\_Hub\_Fine\_Tuning](https://github.com/BrendanJamesLynskey/LLM_Hub_Fine_Tuning) (hub) + FT\_01 – FT\_05 |
| LLM architectures (MoE, SSM, long-context) | [LLM\_Hub\_Modern\_Architectures](https://github.com/BrendanJamesLynskey/LLM_Hub_Modern_Architectures) |
| Evaluations | [LLM\_Hub\_Evaluations](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations) (hub) + LLM\_Eval\_01 – LLM\_Eval\_05 |
| Safety and alignment | [LLM\_Hub\_Safety\_Alignment](https://github.com/BrendanJamesLynskey/LLM_Hub_Safety_Alignment) (hub) + Safety\_01 – Safety\_02 |
| LLMOps and production | [LLM\_Hub\_LLMOps](https://github.com/BrendanJamesLynskey/LLM_Hub_LLMOps) |
| Local LLM hosting (vLLM, Ollama, TGI) | [LLM\_Hub\_Local\_LLM\_Hosting](https://github.com/BrendanJamesLynskey/LLM_Hub_Local_LLM_Hosting) |
| CUDA programming | [LLM\_Hub\_CUDA](https://github.com/BrendanJamesLynskey/LLM_Hub_CUDA) |

Part of the [LLMs hub](https://github.com/BrendanJamesLynskey/LLMs) — an index of LLM-related repositories.

---

## Current status

| Cert | Status | Notes |
| --- | --- | --- |
| NCA-GENL | in-prep | targeting Q3 2026 |
| NCP-GENL | not started | follows NCA |

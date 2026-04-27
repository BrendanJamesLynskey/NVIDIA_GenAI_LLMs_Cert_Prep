# 12-Week Study Plan

Target: NCA-GENL by end of Week 6 (Q3 2026), NCP-GENL by end of Week 12.
Pace: approximately 6 hours per week.

Weeks 3, 9, and 11 are heaviest — flagged explicitly below. Week 6 (exam week) and Week 12 (exam week) are consolidation-only; no new material.

Mock questions reference `mock_interviews/nca_genl_associate.md` (weeks 1–6) and `mock_interviews/ncp_genl_professional.md` (weeks 7–12).

---

## Phase 1 — NCA-GENL (Associate)

### Week 1 — Foundations and transformer architecture

**Focus:** Core ML Knowledge (30% of NCA) — the highest-weighted domain.

| Activity | Detail |
| --- | --- |
| Reading | [notes/01\_ml\_neural\_network\_fundamentals.md](notes/01_ml_neural_network_fundamentals.md) |
| Cross-reference | [`LLM_Hub_Transformer_Architecture`](https://github.com/BrendanJamesLynskey/LLM_Hub_Transformer_Architecture) — visual decoder walkthrough |
| Exercise | [exercises/01\_tokeniser\_from\_scratch/](exercises/01_tokeniser_from_scratch/) — BPE tokeniser from scratch |
| Cheatsheet | [cheatsheets/transformer\_math\_one\_pager.md](cheatsheets/transformer_math_one_pager.md) |
| Mock questions | 10 questions from `nca_genl_associate.md` — Core ML section |

**Effort note:** Week 1 is orientation — lighter than average if you already have a working knowledge of neural networks.

---

### Week 2 — Transformer architecture depth

**Focus:** Transformer internals, positional encodings, decoder-only vs encoder-decoder, KV cache basics.

| Activity | Detail |
| --- | --- |
| Reading | [notes/02\_transformer\_architecture.md](notes/02_transformer_architecture.md) |
| Cross-reference | [`LLM_Hub_Modern_Architectures`](https://github.com/BrendanJamesLynskey/LLM_Hub_Modern_Architectures) — MoE, Mamba, long-context |
| Exercise | [exercises/02\_attention\_from\_scratch/](exercises/02_attention_from_scratch/) — scaled dot-product and multi-head attention |
| Cheatsheet | [cheatsheets/transformer\_math\_one\_pager.md](cheatsheets/transformer_math_one_pager.md) |
| Mock questions | 10 questions from `nca_genl_associate.md` — architecture section |

---

### Week 3 — Prompt engineering and software development (HEAVY)

**Focus:** Software Development (24% of NCA) and Experimentation (22%) — combined these two domains account for nearly half the exam.

| Activity | Detail |
| --- | --- |
| Reading | [notes/03\_prompt\_engineering.md](notes/03_prompt_engineering.md) |
| Cross-reference | [`LLM_Hub_Local_LLM_Hosting`](https://github.com/BrendanJamesLynskey/LLM_Hub_Local_LLM_Hosting) — Ollama, vLLM, serving patterns |
| Exercise | [exercises/04\_triton\_serving\_demo/](exercises/04_triton_serving_demo/) — Docker-based; follow the README for setup |
| Cheatsheet | [cheatsheets/nvidia\_stack\_one\_pager.md](cheatsheets/nvidia_stack_one_pager.md) |
| Mock questions | 15 questions from `nca_genl_associate.md` — software development and prompting sections |

**Effort note:** This week covers the two largest NCA domains simultaneously. Budget closer to 8 hours or split across two sub-weeks if needed.

---

### Week 4 — RAG and experimentation

**Focus:** Experimentation domain continued — RAG pipelines as a key experimental pattern.

| Activity | Detail |
| --- | --- |
| Reading | [notes/05\_rag\_systems.md](notes/05_rag_systems.md) |
| Cross-reference | [`LLM_Hub_RAG_Retrieval`](https://github.com/BrendanJamesLynskey/LLM_Hub_RAG_Retrieval) hub; [`RAG_01_Embedding_Models`](https://github.com/BrendanJamesLynskey/RAG_01_Embedding_Models) through [`RAG_03_Hybrid_Search_and_Reranking`](https://github.com/BrendanJamesLynskey/RAG_03_Hybrid_Search_and_Reranking) |
| Exercise | [exercises/03\_lora\_finetune\_minimal/](exercises/03_lora_finetune_minimal/) — first pass, associate-level understanding of fine-tuning vs RAG trade-offs |
| Cheatsheet | [cheatsheets/quantisation\_and\_kv\_cache.md](cheatsheets/quantisation_and_kv_cache.md) |
| Mock questions | 10 questions from `nca_genl_associate.md` — experimentation section |

---

### Week 5 — Data, evaluation, and trustworthy AI

**Focus:** Data Analysis and Visualisation (14%) + Trustworthy AI (10%) — the two smaller NCA domains.

| Activity | Detail |
| --- | --- |
| Reading | [notes/04\_alignment\_and\_trustworthy\_ai.md](notes/04_alignment_and_trustworthy_ai.md), [notes/09\_evaluation\_and\_metrics.md](notes/09_evaluation_and_metrics.md) |
| Cross-reference | [`LLM_Hub_Safety_Alignment`](https://github.com/BrendanJamesLynskey/LLM_Hub_Safety_Alignment); [`LLM_Hub_Evaluations`](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations) |
| Exercise | Review outputs from weeks 1–4 exercises; check evaluation metrics (perplexity, BLEU, ROUGE) |
| Cheatsheet | [cheatsheets/sampling\_and\_decoding.md](cheatsheets/sampling_and_decoding.md) |
| Mock questions | 20 questions from `nca_genl_associate.md` — data, evaluation, and trustworthy AI sections |

---

### Week 6 — NCA-GENL consolidation and exam

**Focus:** No new material. Consolidation, weak-spot review, full mock, then sit the exam.

| Activity | Detail |
| --- | --- |
| Review | All five NCA domain cheatsheets |
| Mock exam | Full 50-question mock from `nca_genl_associate.md` — timed at 60 minutes |
| Weak-spot drill | Re-read notes for any domain scoring below 70% on the mock |
| SIT EXAM | NCA-GENL — online, remotely proctored, 60 minutes, USD 125 via Certiverse |

---

## Phase 2 — NCP-GENL (Professional)

### Week 7 — Distributed training

**Focus:** GPU Acceleration (14% of NCP) — covers the hardware and parallelism foundations that underpin later weeks.

| Activity | Detail |
| --- | --- |
| Reading | [notes/07\_distributed\_training.md](notes/07_distributed_training.md) |
| Cross-reference | [`LLM_Hub_NVIDIA_GPUs`](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs) — NVLink, NVSwitch, multi-GPU topologies; [`NVIDIA_GPU_04_Memory_Hierarchy`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_04_Memory_Hierarchy) |
| Exercise | Review [exercises/02\_attention\_from\_scratch/](exercises/02_attention_from_scratch/) with multi-GPU data-parallel framing |
| Cheatsheet | [cheatsheets/nvidia\_stack\_one\_pager.md](cheatsheets/nvidia_stack_one_pager.md) |
| Mock questions | 10 questions from `ncp_genl_professional.md` — GPU acceleration section |

**Effort note:** Week 7 is a ramp — after the NCA exam break, expect to rebuild momentum. The distributed-training material is dense; allow extra time for the FSDP/ZeRO concepts.

---

### Week 8 — Inference optimisation and TensorRT-LLM (HEAVY)

**Focus:** Model Optimisation (17% of NCP) — the single largest NCP domain.

| Activity | Detail |
| --- | --- |
| Reading | [notes/08\_inference\_optimisation.md](notes/08_inference_optimisation.md) |
| Cross-reference | [`NVIDIA_GPU_19_TensorRT_LLM`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM) — engine builder, paged KV, FP8/FP4, speculative decoding; [`NVIDIA_GPU_03_Tensor_Cores`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_03_Tensor_Cores) |
| Exercise | [exercises/05\_tensorrt\_llm\_quantisation/](exercises/05_tensorrt_llm_quantisation/) — Docker-based, RTX 3080 / RTX 4000 Ada; follow the README |
| Cheatsheet | [cheatsheets/quantisation\_and\_kv\_cache.md](cheatsheets/quantisation_and_kv_cache.md) |
| Mock questions | 15 questions from `ncp_genl_professional.md` — model optimisation section |

**Effort note:** The TensorRT-LLM exercise requires pulling a large Docker image and may surface driver/container compatibility issues. Reserve time to debug the environment — this is the most technically involved exercise in the repo.

---

### Week 9 — Fine-tuning, PEFT, and LoRA depth (HEAVY)

**Focus:** Fine-Tuning (13%) + Data Preparation (9%) — combined 22% of NCP; also reinforces NCA foundations.

| Activity | Detail |
| --- | --- |
| Reading | [notes/06\_fine\_tuning\_and\_peft.md](notes/06_fine_tuning_and_peft.md) |
| Cross-reference | [`FT_02_LoRA_and_PEFT_Variants`](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants) — LoRA math, rank/alpha, QLoRA, DoRA; [`FT_03_RLHF_and_PPO`](https://github.com/BrendanJamesLynskey/FT_03_RLHF_and_PPO); [`FT_04_DPO_and_Cousins`](https://github.com/BrendanJamesLynskey/FT_04_DPO_and_Cousins); [`RAG_04_Chunking_and_Ingestion`](https://github.com/BrendanJamesLynskey/RAG_04_Chunking_and_Ingestion) for data prep |
| Exercise | [exercises/03\_lora\_finetune\_minimal/](exercises/03_lora_finetune_minimal/) — full run on RTX 4000 Ada; check VRAM usage and loss curves |
| Cheatsheet | [cheatsheets/transformer\_math\_one\_pager.md](cheatsheets/transformer_math_one_pager.md) — re-read with PEFT lens |
| Mock questions | 15 questions from `ncp_genl_professional.md` — fine-tuning and data preparation sections |

**Effort note:** This is the second heaviest week. The LoRA exercise on 13B+ models will push the RTX 4000 Ada (20 GB) to its limits with QLoRA. Plan 8 hours.

---

### Week 10 — Evaluation, safety, and deployment

**Focus:** Evaluation (7%) + Safety, Ethics, and Compliance (5%) + Model Deployment (9%) + Production Monitoring (7%) — four smaller domains totalling 28% of NCP.

| Activity | Detail |
| --- | --- |
| Reading | [notes/09\_evaluation\_and\_metrics.md](notes/09_evaluation_and_metrics.md), [notes/04\_alignment\_and\_trustworthy\_ai.md](notes/04_alignment_and_trustworthy_ai.md), [notes/10\_nvidia\_software\_stack.md](notes/10_nvidia_software_stack.md) |
| Cross-reference | [`LLM_Hub_Evaluations`](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations) hub; [`LLM_Eval_04_Production_Evals`](https://github.com/BrendanJamesLynskey/LLM_Eval_04_Production_Evals); [`Safety_02_Defences_and_Compliance`](https://github.com/BrendanJamesLynskey/Safety_02_Defences_and_Compliance); [`LLM_Hub_LLMOps`](https://github.com/BrendanJamesLynskey/LLM_Hub_LLMOps); [`NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise) |
| Exercise | [exercises/04\_triton\_serving\_demo/](exercises/04_triton_serving_demo/) — add a basic health-check and latency logging |
| Cheatsheet | [cheatsheets/nvidia\_stack\_one\_pager.md](cheatsheets/nvidia_stack_one_pager.md) |
| Mock questions | 20 questions from `ncp_genl_professional.md` — evaluation, safety, deployment, monitoring sections |

---

### Week 11 — NCP mock-exam intensive (HEAVY)

**Focus:** No new notes. Mock-heavy week — identify and close gaps before the final exam.

| Activity | Detail |
| --- | --- |
| Mock exam 1 | Full 60-question mock from `ncp_genl_professional.md` — timed at 120 minutes |
| Gap analysis | For any domain below 70%: re-read the relevant notes file and cross-reference repo |
| System design | One scenario from `mock_interviews/system_design_llm_serving.md` — written answer, then review |
| Mock exam 2 | Second full mock, re-shuffled question order — timed |
| Cheatsheet review | All cheatsheets in one session |

**Effort note:** This is deliberately the heaviest week by time. Two full timed mocks plus gap-fill reading is close to 10 hours. Do not compress it.

---

### Week 12 — NCP-GENL consolidation and exam

**Focus:** No new material. Final consolidation, then sit the exam.

| Activity | Detail |
| --- | --- |
| Review | Weak domains from Week 11 mock analysis only |
| Final cheatsheet pass | Prompt engineering, quantisation, PEFT, and NVIDIA stack one-pagers |
| Behavioural prep | [mock\_interviews/behavioural\_nvidia.md](mock_interviews/behavioural_nvidia.md) — one read-through |
| SIT EXAM | NCP-GENL — online, remotely proctored, 120 minutes, USD 200 via Certiverse |

---

## Summary timeline

| Week | Phase | Focus | Hours (approx) |
| --- | --- | --- | --- |
| 1 | NCA | ML fundamentals, tokeniser exercise | 5 |
| 2 | NCA | Transformer architecture, attention exercise | 6 |
| 3 | NCA | Prompt engineering, serving exercise | 8 |
| 4 | NCA | RAG, LoRA intro | 6 |
| 5 | NCA | Data, evaluation, trustworthy AI | 6 |
| 6 | NCA | Consolidation + **NCA-GENL exam** | 5 |
| 7 | NCP | Distributed training, GPU acceleration | 7 |
| 8 | NCP | Inference optimisation, TensorRT-LLM | 8 |
| 9 | NCP | Fine-tuning, PEFT, LoRA exercise | 8 |
| 10 | NCP | Evaluation, safety, deployment | 6 |
| 11 | NCP | Mock-exam intensive | 10 |
| 12 | NCP | Consolidation + **NCP-GENL exam** | 5 |
| **Total** | | | **~80 hours** |

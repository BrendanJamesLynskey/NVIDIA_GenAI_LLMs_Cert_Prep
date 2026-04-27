# Syllabus

Official domain breakdowns for NCA-GENL (Associate) and NCP-GENL (Professional), mapped to notes files, exercises, and existing portfolio repos.

Domain weightings sourced from the NVIDIA Learn certification pages (April 2026):
- https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/
- https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/

---

## NCA-GENL — Generative AI LLMs Associate

**Exam code:** NCA-GENL | **Duration:** 60 min | **Questions:** 50–60 | **Price:** USD 125 | **Validity:** 2 years

| Domain | Weight | Maps to (this repo) | Cross-reference (existing portfolio) |
| --- | --- | --- | --- |
| Core Machine Learning and AI Knowledge | 30% | [notes/01\_ml\_neural\_network\_fundamentals.md](notes/01_ml_neural_network_fundamentals.md), [notes/02\_transformer\_architecture.md](notes/02_transformer_architecture.md) | [`LLM_Hub_Transformer_Architecture`](https://github.com/BrendanJamesLynskey/LLM_Hub_Transformer_Architecture), [`LLM_Hub_Modern_Architectures`](https://github.com/BrendanJamesLynskey/LLM_Hub_Modern_Architectures) |
| Software Development | 24% | [notes/03\_prompt\_engineering.md](notes/03_prompt_engineering.md), [notes/10\_nvidia\_software\_stack.md](notes/10_nvidia_software_stack.md), [exercises/01\_tokeniser\_from\_scratch/](exercises/01_tokeniser_from_scratch/), [exercises/02\_attention\_from\_scratch/](exercises/02_attention_from_scratch/) | [`LLM_Hub_NVIDIA_GPUs`](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs), [`NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise), [`LLM_Hub_Local_LLM_Hosting`](https://github.com/BrendanJamesLynskey/LLM_Hub_Local_LLM_Hosting) |
| Experimentation | 22% | [notes/05\_rag\_systems.md](notes/05_rag_systems.md), [notes/06\_fine\_tuning\_and\_peft.md](notes/06_fine_tuning_and_peft.md), [exercises/03\_lora\_finetune\_minimal/](exercises/03_lora_finetune_minimal/) | [`LLM_Hub_Fine_Tuning`](https://github.com/BrendanJamesLynskey/LLM_Hub_Fine_Tuning), [`LLM_Hub_RAG_Retrieval`](https://github.com/BrendanJamesLynskey/LLM_Hub_RAG_Retrieval) |
| Data Analysis and Visualisation | 14% | [notes/09\_evaluation\_and\_metrics.md](notes/09_evaluation_and_metrics.md) | [`LLM_Hub_Evaluations`](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations) |
| Trustworthy AI | 10% | [notes/04\_alignment\_and\_trustworthy\_ai.md](notes/04_alignment_and_trustworthy_ai.md) | [`LLM_Hub_Safety_Alignment`](https://github.com/BrendanJamesLynskey/LLM_Hub_Safety_Alignment) |

### NCA-GENL topic coverage in notes

| Notes file | Topics covered |
| --- | --- |
| [01\_ml\_neural\_network\_fundamentals.md](notes/01_ml_neural_network_fundamentals.md) | Supervised/unsupervised/RL, loss functions, backprop, neural network fundamentals, training dynamics |
| [02\_transformer\_architecture.md](notes/02_transformer_architecture.md) | Attention mechanism, multi-head attention, positional encoding, layer norm, decoder-only vs encoder-decoder |
| [03\_prompt\_engineering.md](notes/03_prompt_engineering.md) | Zero-shot, few-shot, chain-of-thought, system prompts, prompt injection, structured output |
| [04\_alignment\_and\_trustworthy\_ai.md](notes/04_alignment_and_trustworthy_ai.md) | RLHF, Constitutional AI, DPO, guardrails, bias, fairness, EU AI Act, NIST AI RMF |
| [09\_evaluation\_and\_metrics.md](notes/09_evaluation_and_metrics.md) | Perplexity, BLEU, ROUGE, BERTScore, LLM-as-judge, RAGAS, benchmark suites |

---

## NCP-GENL — Generative AI LLMs Professional

**Exam code:** NCP-GENL | **Duration:** 120 min | **Questions:** 60–70 | **Price:** USD 200 | **Validity:** 2 years  
**Prerequisites:** 2–3 years practical AI/ML experience with LLMs; knowledge of transformer architectures, distributed parallelism, PEFT

| Domain | Weight | Maps to (this repo) | Cross-reference (existing portfolio) |
| --- | --- | --- | --- |
| Model Optimisation | 17% | [notes/08\_inference\_optimisation.md](notes/08_inference_optimisation.md), [exercises/05\_tensorrt\_llm\_quantisation/](exercises/05_tensorrt_llm_quantisation/) | [`NVIDIA_GPU_19_TensorRT_LLM`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM), [`NVIDIA_GPU_03_Tensor_Cores`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_03_Tensor_Cores) |
| GPU Acceleration and Optimisation | 14% | [notes/10\_nvidia\_software\_stack.md](notes/10_nvidia_software_stack.md) | [`LLM_Hub_NVIDIA_GPUs`](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs), [`NVIDIA_GPU_04_Memory_Hierarchy`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_04_Memory_Hierarchy), [`LLM_Hub_CUDA`](https://github.com/BrendanJamesLynskey/LLM_Hub_CUDA) |
| Prompt Engineering | 13% | [notes/03\_prompt\_engineering.md](notes/03_prompt_engineering.md) | [`LLM_Hub_Transformer_Architecture`](https://github.com/BrendanJamesLynskey/LLM_Hub_Transformer_Architecture) |
| Fine-Tuning | 13% | [notes/06\_fine\_tuning\_and\_peft.md](notes/06_fine_tuning_and_peft.md), [exercises/03\_lora\_finetune\_minimal/](exercises/03_lora_finetune_minimal/) | [`LLM_Hub_Fine_Tuning`](https://github.com/BrendanJamesLynskey/LLM_Hub_Fine_Tuning), [`FT_02_LoRA_and_PEFT_Variants`](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants), [`FT_03_RLHF_and_PPO`](https://github.com/BrendanJamesLynskey/FT_03_RLHF_and_PPO), [`FT_04_DPO_and_Cousins`](https://github.com/BrendanJamesLynskey/FT_04_DPO_and_Cousins) |
| Data Preparation | 9% | [notes/01\_ml\_neural\_network\_fundamentals.md](notes/01_ml_neural_network_fundamentals.md) | [`RAG_04_Chunking_and_Ingestion`](https://github.com/BrendanJamesLynskey/RAG_04_Chunking_and_Ingestion) |
| Model Deployment | 9% | [notes/10\_nvidia\_software\_stack.md](notes/10_nvidia_software_stack.md), [exercises/04\_triton\_serving\_demo/](exercises/04_triton_serving_demo/) | [`LLM_Hub_Local_LLM_Hosting`](https://github.com/BrendanJamesLynskey/LLM_Hub_Local_LLM_Hosting), [`LLM_Hub_LLMOps`](https://github.com/BrendanJamesLynskey/LLM_Hub_LLMOps), [`NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise) |
| Evaluation | 7% | [notes/09\_evaluation\_and\_metrics.md](notes/09_evaluation_and_metrics.md) | [`LLM_Hub_Evaluations`](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations), [`LLM_Eval_01_Landscape`](https://github.com/BrendanJamesLynskey/LLM_Eval_01_Landscape) – [`LLM_Eval_05_Red_Teaming`](https://github.com/BrendanJamesLynskey/LLM_Eval_05_Red_Teaming) |
| Production Monitoring and Reliability | 7% | [notes/10\_nvidia\_software\_stack.md](notes/10_nvidia_software_stack.md) | [`LLM_Hub_LLMOps`](https://github.com/BrendanJamesLynskey/LLM_Hub_LLMOps) |
| LLM Architecture | 6% | [notes/02\_transformer\_architecture.md](notes/02_transformer_architecture.md) | [`LLM_Hub_Transformer_Architecture`](https://github.com/BrendanJamesLynskey/LLM_Hub_Transformer_Architecture), [`LLM_Hub_Modern_Architectures`](https://github.com/BrendanJamesLynskey/LLM_Hub_Modern_Architectures) |
| Safety, Ethics, and Compliance | 5% | [notes/04\_alignment\_and\_trustworthy\_ai.md](notes/04_alignment_and_trustworthy_ai.md) | [`LLM_Hub_Safety_Alignment`](https://github.com/BrendanJamesLynskey/LLM_Hub_Safety_Alignment), [`Safety_01_Jailbreaks`](https://github.com/BrendanJamesLynskey/Safety_01_Jailbreaks), [`Safety_02_Defences_and_Compliance`](https://github.com/BrendanJamesLynskey/Safety_02_Defences_and_Compliance) |

### NCP-GENL additional notes files

| Notes file | Topics covered |
| --- | --- |
| [05\_rag\_systems.md](notes/05_rag_systems.md) | Embedding models, vector DBs, hybrid search, reranking, agentic RAG, GraphRAG |
| [06\_fine\_tuning\_and\_peft.md](notes/06_fine_tuning_and_peft.md) | SFT, LoRA/QLoRA/DoRA, RLHF/PPO, DPO, Constitutional AI, RLAIF |
| [07\_distributed\_training.md](notes/07_distributed_training.md) | Data/tensor/pipeline/expert parallelism, FSDP, DeepSpeed ZeRO, NCCL, gradient accumulation |
| [08\_inference\_optimisation.md](notes/08_inference_optimisation.md) | Quantisation (INT8/FP8/FP4), KV cache, paged attention, speculative decoding, continuous batching, TensorRT-LLM |
| [10\_nvidia\_software\_stack.md](notes/10_nvidia_software_stack.md) | NeMo, NIM, Triton Inference Server, TensorRT-LLM, CUDA, cuDNN, NCCL, AI Enterprise |

---

## Exercises mapped to domains

| Exercise | Relevant cert domains |
| --- | --- |
| [01\_tokeniser\_from\_scratch/](exercises/01_tokeniser_from_scratch/) | NCA: Core ML Knowledge; NCP: LLM Architecture |
| [02\_attention\_from\_scratch/](exercises/02_attention_from_scratch/) | NCA: Core ML Knowledge; NCP: LLM Architecture |
| [03\_lora\_finetune\_minimal/](exercises/03_lora_finetune_minimal/) | NCA: Experimentation; NCP: Fine-Tuning |
| [04\_triton\_serving\_demo/](exercises/04_triton_serving_demo/) | NCA: Software Development; NCP: Model Deployment |
| [05\_tensorrt\_llm\_quantisation/](exercises/05_tensorrt_llm_quantisation/) | NCP: Model Optimisation, GPU Acceleration |

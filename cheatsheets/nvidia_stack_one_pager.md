# NVIDIA Stack — One-Pager

| Component | What it is | Lifecycle stage | Sits on top of | Reach for it when… |
|---|---|---|---|---|
| **CUDA Toolkit** | Compiler (nvcc), runtime, profiling APIs | All | GPU driver | Any GPU-accelerated code |
| **cuDNN** | DNN primitives (conv, attention, norm, activation) | Train / Serve | CUDA | PyTorch/TF use it transparently |
| **cuBLAS** | Dense GEMM / GEMV | Train / Serve | CUDA | Matmul throughput tuning |
| **NCCL** | Collective comms (all-reduce, all-gather) | Train / Serve | CUDA | Multi-GPU or multi-node |
| **CUTLASS** | Composable GEMM C++ templates | Train / Serve | CUDA | Custom high-perf kernels |
| **RAPIDS** | GPU data science (cuDF, cuML, cuGraph, cuVS) | Data prep | CUDA | GPU-accelerated ETL / RAG vector search |
| **NeMo Curator** | Dataset dedup, quality filter, language ID | Data prep | RAPIDS + PyTorch | Building pre-train or fine-tune corpora |
| **NeMo Framework** | Scalable pre-train + fine-tune (Megatron Core) | Train | PyTorch + NCCL | Pre-training or multi-GPU SFT at scale |
| **NeMo RL** (ex-Aligner) | RLHF (PPO), DPO, GRPO alignment training | Post-train | NeMo Framework | Alignment / preference optimisation |
| **NeMo Guardrails** | Runtime safety via Colang policies | Serve | Any LLM endpoint | Enforcing topic/format/jailbreak policies |
| **NeMo Evaluator** | Benchmark runner; integrates with NeMo Run | Evaluate | NeMo ecosystem | Running standard eval suites on checkpoints |
| **TensorRT** | General-purpose inference engine compiler | Serve | CUDA + cuDNN | Non-LLM models (CNN, encoder, embedding) |
| **TensorRT-LLM** | LLM engine compiler + C++ runtime | Serve | CUDA + CUTLASS | Max NVIDIA throughput; FP8/FP4/INT4 |
| **Triton Inference Server** | Multi-backend model server (gRPC + REST) | Serve | TRT-LLM / ONNX / PyTorch | Serving any model; dynamic batching; ensembles |
| **NIM** | Containerised LLM microservice (NGC catalogue) | Serve | TRT-LLM + Triton | Zero-config production deployment |
| **Nsight Systems** | System-level profiler (timeline: CPU+GPU+NCCL) | Optimise | CUDA driver | First step: finding where time goes |
| **Nsight Compute** | Kernel-level profiler (roofline, occupancy, stalls) | Optimise | CUDA | Second step: why a specific kernel is slow |
| **MIG** | Hardware GPU partitioning; isolated memory+compute | Serve | A100 / H100 / H200 | Strict multi-tenant isolation (not on RTX/Ada) |
| **MPS** | Multi-process shared GPU context (no isolation) | Serve | Any CUDA GPU | Same-user workloads sharing one GPU |
| **vGPU** | Hypervisor GPU sharing for VMs | Serve | Licensed vGPU driver | VDI / virtualised data centres |
| **Base Command** | DGX cluster management and job scheduling | Train / Serve | DGX infrastructure | On-prem DGX job submission and monitoring |
| **Run.ai** | Kubernetes GPU scheduler (fractional sharing, queues) | Train / Serve | Kubernetes + AI Enterprise | Multi-tenant cluster fairness / GPU sharing |
| **DGX Cloud** | Managed cloud DGX clusters (Azure/GCP/OCI) | Train / Serve | Base Command + AI Enterprise | On-demand cluster-scale training without on-prem |
| **AI Enterprise** | Commercial subscription: NeMo + NIM + Triton + Run.ai + LTS drivers | All | Above stack | Regulated industries; production SLA; NIM at scale |

---

## Key distinctions

- **TensorRT-LLM ≠ Triton**: TRT-LLM *compiles the engine and provides the runtime*; Triton *serves it* (as one of several backends).
- **NeMo Guardrails** is runtime-only — it wraps an existing LLM endpoint; it does not train anything.
- **NIM** bundles TRT-LLM + Triton + OpenAI-compatible API; it is a deployment package, not a new runtime.
- **MIG** requires A100/H100/H200 — not available on RTX 3080 or RTX 4000 Ada.
- **Nsight Systems** first → **Nsight Compute** second (system timeline → kernel deep-dive).

---

## Decision flowchart

- Prepare pre-training data at scale → **NeMo Curator**
- Pre-train or multi-GPU fine-tune → **NeMo Framework**
- Align with RLHF / DPO → **NeMo RL**
- Add runtime safety policies → **NeMo Guardrails**
- Compile fastest possible LLM engine → **TensorRT-LLM**
- Serve multiple models / backends → **Triton Inference Server**
- Deploy with one `docker run` → **NIM**
- Accelerate pandas / scikit-learn on GPU → **RAPIDS**
- Profile system-level bottlenecks → **Nsight Systems**
- Diagnose a slow CUDA kernel → **Nsight Compute**
- Schedule GPU jobs in multi-tenant K8s → **Run.ai**
- Need LTS drivers + support SLA → **AI Enterprise**

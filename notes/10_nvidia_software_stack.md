# NVIDIA Software Stack

This note covers three NCP-GENL domains: GPU Acceleration and Optimisation (14%), Model Deployment (9%), and Production Monitoring and Reliability (7%). It is also relevant to NCA-GENL Software Development (24%). The NVIDIA stack is high-stakes for the cert — questions will probe component boundaries, the relationship between tools, and where each belongs in a production workflow.

Primary portfolio references: [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM) and [NVIDIA\_GPU\_20\_NeMo\_NIM\_AI\_Enterprise](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise). The full GPU architecture series lives at [LLM\_Hub\_NVIDIA\_GPUs](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs).

---

## The Stack at a Glance

The NVIDIA software ecosystem is layered. From silicon to application:

```
┌────────────────────────────────────────────────────────────────┐
│  Applications: NeMo, NIM, user workloads                       │
├────────────────────────────────────────────────────────────────┤
│  Serving: Triton Inference Server + TensorRT-LLM backend       │
├────────────────────────────────────────────────────────────────┤
│  Optimisation: TensorRT / TensorRT-LLM (engine compilation)    │
├────────────────────────────────────────────────────────────────┤
│  ML frameworks: PyTorch, JAX, TensorFlow                       │
├────────────────────────────────────────────────────────────────┤
│  CUDA libraries: cuDNN, cuBLAS, NCCL, CUTLASS, RAPIDS         │
├────────────────────────────────────────────────────────────────┤
│  CUDA Toolkit (13.x as of April 2026) + CUDA Runtime          │
├────────────────────────────────────────────────────────────────┤
│  GPU Driver (≥ 580 for CUDA 13.x)                              │
├────────────────────────────────────────────────────────────────┤
│  GPU Hardware (Ampere / Ada / Hopper / Blackwell)              │
└────────────────────────────────────────────────────────────────┘
```

Each layer is independently versioned. Compatibility matrices matter in practice: a CUDA 13.x toolkit requires driver ≥ 580; cuDNN versions are tied to specific CUDA major versions; TensorRT-LLM container images bundle a specific combination.

### Core CUDA Libraries

| Library | Purpose |
|---------|---------|
| **cuDNN** | Primitives for deep neural networks: convolution, attention, normalisation, activation. Used by PyTorch and TensorFlow transparently |
| **cuBLAS** | Dense linear algebra (GEMM, GEMV, batch GEMM). The workhorse for matrix multiplications in LLM layers |
| **NCCL** | Collective communications (all-reduce, all-gather, reduce-scatter, all-to-all) for multi-GPU and multi-node training and inference |
| **CUTLASS** | Composable CUDA C++ templates for GEMM and convolution; used as the foundation for custom high-performance kernels including those in TensorRT-LLM |
| **cuSPARSE** | Sparse matrix operations; less central for LLMs but relevant for sparse attention |

For GPU memory hierarchy and how these libraries use it, see [NVIDIA\_GPU\_04\_Memory\_Hierarchy](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_04_Memory_Hierarchy) and [NVIDIA\_GPU\_03\_Tensor\_Cores](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_03_Tensor_Cores).

---

## NeMo Framework

NeMo Framework is NVIDIA's scalable, cloud-native library for pretraining and fine-tuning large language models, multimodal models, and speech AI. It is built on PyTorch and leverages **Megatron Core** (the Megatron-LM training engine) for the parallelism infrastructure — tensor parallelism, pipeline parallelism, sequence parallelism, context parallelism, and expert parallelism.

The framework has evolved significantly. As of the 26.x container releases, NeMo references Megatron Core via **NeMo Megatron Bridge** for LLM/VLM training, while speech AI uses the traditional NeMo collections. The main subcomponents relevant to the cert:

### NeMo Curator

Data curation library for preparing pretraining and fine-tuning datasets at scale. Capabilities include: document deduplication (exact and fuzzy, via MinHash), quality filtering (heuristics and classifier-based), language identification, toxic content filtering, and data blending. Designed for GPU-accelerated processing of internet-scale corpora. The data preparation pipeline that feeds NeMo pretraining runs typically starts here.

### NeMo RL (formerly NeMo Aligner)

The alignment training library. Implements RLHF (PPO), DPO, REINFORCE/GRPO, and related algorithms for post-training alignment. Integrates with the Megatron-based training infrastructure for efficient large-scale alignment runs. This is NVIDIA's answer to libraries like TRL (Hugging Face); it targets multi-GPU cluster operation rather than single-GPU fine-tuning.

### NeMo Guardrails

A runtime safety and controllability framework for LLM-powered applications. It is **not** a training-time tool — it operates during inference, between the application and the LLM. Guardrails are defined in Colang (a domain-specific language) and enforce conversational policies: topic restrictions, fact-checking connections, output formatting rules, jailbreak mitigations. Guardrails can wrap any LLM endpoint.

### NeMo Evaluator

An evaluation service within the NeMo ecosystem for running benchmark suites against trained models. Integrates with NeMo Run for experiment management.

### NeMo Run

Experiment management tool: configures, launches, and tracks training and evaluation runs across local and cluster environments.

For detailed coverage of NeMo and its components, see [NVIDIA\_GPU\_20\_NeMo\_NIM\_AI\_Enterprise](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise).

---

## TensorRT and TensorRT-LLM

### TensorRT

TensorRT is NVIDIA's general-purpose inference optimisation engine. Given a neural network graph (from ONNX, PyTorch, TensorFlow), it compiles a hardware-specific *engine* that fuses operations, selects optimal kernel implementations, and applies precision calibration. Supports FP32, FP16, INT8, and FP8 on appropriate hardware. TensorRT is the foundation for optimising non-LLM models (CNNs, embedding models, encoders).

### TensorRT-LLM

TensorRT-LLM is the LLM-specific extension of TensorRT. It is an open-source Python/C++ library that exposes both a Python model-authoring API and a high-performance C++ runtime. Key features verified from official documentation:

- **Quantisation**: FP8, NVFP4 (4-bit, Ada/Hopper), INT4 AWQ, INT8 SmoothQuant.
- **In-flight batching**: iteration-level dynamic batch management; sequences enter and leave the batch as they complete.
- **Paged KV cache**: non-contiguous memory blocks for KV storage; enables larger effective batch sizes.
- **Speculative decoding**: supports draft models, Medusa (extra decoding heads), and EAGLE-3 (feature-level auto-regressive draft).
- **Parallelism**: tensor parallelism, pipeline parallelism, expert parallelism (including DeepEP-based wide expert parallelism).
- **Multi-GPU/multi-node**: scales from a single GPU to multi-node deployments.

The workflow is: author the model with the TensorRT-LLM Python API (or use pre-built model implementations) → compile an optimised engine → serve via the TRT-LLM C++ runtime or expose via Triton Inference Server.

For full TensorRT-LLM coverage, see [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM).

---

## Triton Inference Server

Triton Inference Server is NVIDIA's open-source model serving system. The key conceptual distinction: **Triton is the server; TensorRT-LLM is one of its backends**.

Triton's architecture:

- **Multi-backend**: serves TensorRT engines, ONNX Runtime, PyTorch (TorchScript/eager), OpenVINO, Python (arbitrary code), and TensorRT-LLM. A single Triton instance can simultaneously serve models from different backends.
- **Dynamic batching**: the server automatically groups incoming requests into batches up to a configured maximum, improving GPU utilisation without application-level batching logic.
- **Model ensembling**: a pipeline of models can be defined as a *model ensemble* — e.g., a preprocessing step (tokenisation), the LLM forward pass, and a postprocessing step — and executed as a single logical request. Reduces round-trip latency for multi-model inference.
- **gRPC and HTTP/REST**: dual protocol support; gRPC is preferred for low-latency production; REST for compatibility.
- **Model repository**: models are served from a filesystem directory (or object store); hot-loading and versioning are supported without server restarts.

For LLM serving, the standard NVIDIA-recommended path is: TensorRT-LLM engine → TensorRT-LLM Triton backend → Triton server. NIM packages this stack as a ready-to-deploy container.

---

## NIM — NVIDIA Inference Microservices

NIM is a set of containerised inference microservices for deploying AI models in production. Each NIM container bundles:

- An optimised inference engine (TensorRT-LLM for LLMs, or other NVIDIA-optimised backends)
- A serving runtime (Triton Inference Server or an equivalent)
- An OpenAI-compatible REST API surface (plus gRPC where applicable)
- Model weights validated and tested across NVIDIA hardware profiles
- Health checks, telemetry endpoints, and enterprise runtime configuration

The value proposition is operational: instead of manually compiling TensorRT-LLM engines, configuring Triton, and writing deployment scripts, a developer runs a single `docker run` invocation (or Helm chart in Kubernetes). The NIM catalog (hosted on NGC) covers LLMs (Llama, Mistral, Nemotron families), embedding models, rerankers, and domain-specific models.

NIM requires NVIDIA AI Enterprise licensing for production use, though developer-tier access is available with an NVIDIA developer account.

For NIM and its relationship to the broader NVIDIA stack, see [NVIDIA\_GPU\_20\_NeMo\_NIM\_AI\_Enterprise](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise).

---

## Base Command, Run.ai, and DGX Cloud

These are the orchestration and infrastructure layer, relevant to enterprise cluster operation. Brendan has no DGX or cloud GPU access — this is exam-knowledge only.

### Base Command

NVIDIA Base Command is the cluster management and job scheduling platform for DGX systems. It provides a unified control plane for submitting training jobs, managing datasets, tracking experiments, and monitoring cluster health across on-premises DGX infrastructure.

### Run.ai

Run.ai (acquired by NVIDIA) is a Kubernetes-native workload scheduler optimised for GPU workloads. It implements fractional GPU sharing, advanced queuing policies, and cluster fairness across teams. Run.ai is included in NVIDIA AI Enterprise as the scheduling layer for multi-tenant GPU clusters. It sits above the raw Kubernetes scheduler and below NeMo or user applications.

### DGX Cloud

DGX Cloud is NVIDIA's managed cloud service providing on-demand access to DGX-grade GPU clusters across major cloud providers (Microsoft Azure, Google Cloud, Oracle Cloud). It integrates Base Command for job management and provides the full AI Enterprise software stack pre-configured.

---

## NVIDIA AI Enterprise

AI Enterprise is NVIDIA's commercial software subscription that bundles:

- All NeMo Framework components (NeMo Curator, NeMo RL, NeMo Guardrails, NeMo Evaluator)
- NIM microservices (with production-grade SLA)
- Triton Inference Server (enterprise-supported build)
- NVIDIA Container Toolkit
- Run.ai for workload orchestration
- Kubernetes operators for GPU and network management
- GPU drivers on supported release branches (Production Branch with 9-month support; LTS Branch with 36-month support)
- Enterprise support tiers (Business Standard baseline; 24/7 available as add-on)

The licensing model is subscription-based. Open-source community versions of NeMo and Triton exist and are freely available; AI Enterprise adds long-term support, security patches, validated configurations, and the NIM catalog.

When AI Enterprise matters: regulated industries (healthcare, finance) requiring an LTS driver branch and formal support SLA; organisations deploying NIM at scale; any deployment where NVIDIA's indemnification and support is contractually required.

---

## RAPIDS

RAPIDS is NVIDIA's open-source GPU-accelerated data science library suite, part of the CUDA-X ecosystem. It is lighter-weight in cert coverage than the inference/training stack but appears in data preparation and MLOps contexts.

| Library | Drop-in replacement for | What it accelerates |
|---------|------------------------|---------------------|
| **cuDF** | pandas | DataFrame operations — groupby, join, sort, I/O |
| **cuML** | scikit-learn | Classical ML — regression, clustering, dimensionality reduction, UMAP, HDBSCAN |
| **cuGraph** | NetworkX | Graph analytics — PageRank, connected components, community detection |
| **cuVS** | FAISS (partially) | Vector search and approximate nearest neighbour |

RAPIDS is relevant for GPU-accelerated ETL and feature engineering pipelines upstream of model training, and for embedding search in RAG systems. NeMo Curator uses RAPIDS internally for deduplication and filtering.

---

## Nsight Profiling Tools

NVIDIA provides two primary profiling tools, often confused:

### Nsight Systems

**System-level** profiler. Captures CPU thread activity, GPU kernel launches, CUDA API calls, memory transfers, NCCL collectives, I/O, and their timing relationships in a unified timeline. Use Nsight Systems first — it answers "where is the time going?" and identifies bottlenecks at the system level (e.g., data loading bottleneck, CPU-GPU synchronisation stalls, pipeline idle periods).

### Nsight Compute

**Kernel-level** profiler. Given a specific CUDA kernel, it collects roofline model data, memory throughput, warp occupancy, instruction mix, and stall reasons. Use Nsight Compute after Nsight Systems has identified a hot kernel — it answers "why is this kernel slow?" and what architectural ceiling it is hitting.

**Basic profiling workflow**:
1. Run with Nsight Systems; inspect the timeline for idle periods, CPU-GPU imbalance, and NCCL collective bottlenecks.
2. Identify the most time-consuming CUDA kernels.
3. Profile those kernels in isolation with Nsight Compute.
4. Interpret against the roofline model — is the kernel compute-bound or memory-bandwidth-bound?

For CUDA programming and kernel optimisation, see [LLM\_Hub\_CUDA](https://github.com/BrendanJamesLynskey/LLM_Hub_CUDA).

---

## MIG, MPS, and vGPU

Three mechanisms for sharing a single physical GPU across multiple workloads:

| Mechanism | Full name | What it does | Best for |
|-----------|-----------|--------------|---------|
| **MIG** | Multi-Instance GPU | Hardware partitions the GPU into isolated slices; each has dedicated memory and compute. Supported on A100, H100, H200. Not available on RTX/Ada consumer GPUs | Cloud multi-tenancy; strict isolation between workloads |
| **MPS** | Multi-Process Service | A CUDA daemon that allows multiple CUDA processes to share a single GPU context; lower overhead than process-switching; no memory isolation | Same-user workloads needing GPU sharing without full isolation |
| **vGPU** | Virtual GPU | Hypervisor-level GPU sharing for virtual machines; requires a licensed NVIDIA vGPU driver | VDI, virtualised data centres |

For a full comparison and when to use each, see [NVIDIA\_GPU\_18\_GPU\_Sharing\_MIG\_MPS\_vGPU](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_18_GPU_Sharing_MIG_MPS_vGPU).

Note: MIG is not available on the RTX 3080 or RTX 4000 Ada. MPS can be used on both.

---

## Decision Matrix

| I want to… | Reach for… |
|------------|-----------|
| Pretrain an LLM from scratch | NeMo Framework (Megatron Bridge) + NeMo Curator for data |
| Fine-tune with LoRA / SFT | NeMo Framework; or HF TRL + Accelerate for lighter setups |
| Align with RLHF / DPO | NeMo RL |
| Add runtime safety guardrails | NeMo Guardrails |
| Compile an optimised inference engine | TensorRT-LLM |
| Serve models at production throughput | Triton Inference Server + TensorRT-LLM backend |
| Deploy with zero inference configuration | NIM (pull from NGC catalog) |
| Profile the whole system for bottlenecks | Nsight Systems |
| Optimise a specific CUDA kernel | Nsight Compute |
| Accelerate data science / ETL on GPU | RAPIDS (cuDF, cuML, cuGraph) |
| Schedule GPU jobs in a multi-tenant cluster | Run.ai (via AI Enterprise) |

---

## Likely Exam Angles

- **Component boundaries**: distinguish between TensorRT-LLM (engine builder + runtime) and Triton (the serving layer that hosts it as a backend). Candidates frequently conflate these.
- **NeMo subcomponent roles**: given a task (data curation / alignment / guardrails), identify the correct NeMo subcomponent.
- **NIM**: explain what NIM bundles and why it simplifies deployment versus assembling TRT-LLM + Triton manually.
- **MIG vs MPS vs vGPU**: given a scenario (strict multi-tenant isolation / same-user sharing / VM virtualisation), select the appropriate mechanism.
- **Nsight Systems vs Nsight Compute**: know that Systems is the starting point for any profiling investigation; Compute digs into individual kernels.
- **AI Enterprise**: identify what it adds beyond the open-source equivalents and when a licence is required.

---

## Further Reading

- [NVIDIA\_GPU\_20\_NeMo\_NIM\_AI\_Enterprise](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise) — NeMo/NIM depth
- [NVIDIA\_GPU\_19\_TensorRT\_LLM](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM) — TensorRT-LLM depth
- [LLM\_Hub\_NVIDIA\_GPUs](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs) — full NVIDIA GPU architecture series
- [NVIDIA\_GPU\_18\_GPU\_Sharing\_MIG\_MPS\_vGPU](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_18_GPU_Sharing_MIG_MPS_vGPU) — GPU sharing mechanisms
- NeMo Framework docs: <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>
- TensorRT-LLM developer page: <https://developer.nvidia.com/tensorrt-llm>
- Triton Inference Server docs: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/index.html>
- NIM introduction: <https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/>
- RAPIDS overview: <https://developer.nvidia.com/rapids>
- NVIDIA AI Enterprise product support: <https://docs.nvidia.com/ai-enterprise/latest/product-support-matrix/index.html>
- Nsight Systems user guide: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- CUDA Toolkit: <https://developer.nvidia.com/cuda-toolkit>

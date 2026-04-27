# NCP-GENL Mock Exam — Generative AI LLMs Professional

**Syllabus reference:** [../SYLLABUS.md](../SYLLABUS.md) | **Relevant notes:** [../notes/](../notes/) | **Numerical formulas:** [../cheatsheets/transformer_math_one_pager.md](../cheatsheets/transformer_math_one_pager.md), [../cheatsheets/quantisation_and_kv_cache.md](../cheatsheets/quantisation_and_kv_cache.md)

This file contains 30 scenario-heavy questions in the style of the NCP-GENL Professional exam (exam code NCP-GENL; 120 min; 60–70 questions; USD 200). Questions test real engineering trade-offs, multi-component system reasoning, and — for at least 6 questions marked **Numerical** — explicit calculation. Every question has a scenario stem, four options A–D, and a rationale in the answer key at the end.

**How to use:** Set a 120-minute timer, work through all 30 questions without consulting notes. Numerical questions require arithmetic — have a calculator ready. Aim for under 4 minutes per question on average, leaving time for review.

---

## Question distribution by domain

| Domain | Exam weight | Questions in this set | Question numbers |
|---|---|---|---|
| Model Optimisation | 17% | 5 | Q1–Q5 |
| GPU Acceleration and Optimisation | 14% | 4 | Q6–Q9 |
| Prompt Engineering | 13% | 4 | Q10–Q13 |
| Fine-Tuning | 13% | 4 | Q14–Q17 |
| Data Preparation | 9% | 3 | Q18–Q20 |
| Model Deployment | 9% | 3 | Q21–Q23 |
| Evaluation | 7% | 2 | Q24–Q25 |
| Production Monitoring and Reliability | 7% | 2 | Q26–Q27 |
| LLM Architecture | 6% | 2 | Q28–Q29 |
| Safety, Ethics, and Compliance | 5% | 1 | Q30 |

**Numerical questions:** Q1, Q2, Q6, Q14, Q28, Q29 (6 total).

---

## Model Optimisation (Q1–Q5)

## Q1

**Numerical**

You are deploying a 7B model on a single NVIDIA RTX 4000 Ada (20 GB VRAM). The model has 32 transformer layers, 32 KV heads, and a head dimension of 128. You want to serve requests at an average context length of 8192 tokens in FP16 precision.

Calculate the KV cache memory per sequence at 8192 tokens. Given that the model weights occupy approximately 14 GB in BF16, how many concurrent sequences can you support before exceeding the GPU's 20 GB limit?

A. KV cache ≈ 4 GB/sequence; with 14 GB occupied by weights, no sequences can fit.  
B. KV cache ≈ 2 GB/sequence; with 14 GB weights, 3 GB remains — 1 sequence fits, but barely.  
C. KV cache ≈ 4 GB/sequence at 8192 tokens; with 14 GB weights, 6 GB remains — 1 full sequence at 8192 tokens fits, leaving 2 GB of headroom.  
D. KV cache ≈ 1 GB/sequence; with 14 GB weights, 3 sequences fit comfortably.

---

## Q2

**Numerical**

A team is evaluating whether to quantise a 7B inference workload from FP16 weights to INT4 using AWQ on an RTX 3080 (10 GB). They want to serve batches of 4 sequences at 4096-token context, with the KV cache remaining in FP16.

The model has 32 layers, 32 KV heads, head dimension 128. At INT4, the model weights occupy approximately 3.5 GB. Calculate the KV cache requirement for 4 sequences at 4096 tokens in FP16. Does the combined load fit within 10 GB?

A. KV cache = 8 GB for 4 sequences; combined with 3.5 GB weights = 11.5 GB — does not fit.  
B. KV cache = 2 GB per sequence × 4 = 8 GB; combined with 3.5 GB weights = 11.5 GB — does not fit in 10 GB. Reduce batch size to 3 or apply KV cache quantisation.  
C. KV cache = 1 GB per sequence × 4 = 4 GB; combined with 3.5 GB weights = 7.5 GB — fits comfortably.  
D. KV cache = 0.5 GB per sequence × 4 = 2 GB; combined with 3.5 GB weights = 5.5 GB — well within 10 GB.

---

## Q3

A senior ML engineer is evaluating speculative decoding for a customer-facing chatbot. The target model is a 70B model served on an 8×H100 cluster. The proposed draft model is a 7B model. Draft tokens are accepted at a rate of around 60%. Which factor most determines whether speculative decoding will improve end-to-end throughput compared to standard autoregressive decoding?

A. The H100's tensor core throughput — speculative decoding requires more FP8 matmuls, which may saturate the tensor cores.  
B. The ratio of the draft model's per-token latency to the target model's verification latency and the acceptance rate. If the draft produces k = 5 tokens accepted at 60%, roughly 3 tokens are committed per target forward pass, providing a 3× effective throughput gain if the single-target pass is not slower than 5 individual target passes.  
C. The VRAM available for the draft model; the draft must reside entirely on the same GPUs as the target model.  
D. The batch size — speculative decoding only provides gains at batch size 1; at larger batch sizes, standard continuous batching outperforms it.

---

## Q4

You are choosing between AWQ (INT4 weight-only) and FP8 (weight-and-activation) quantisation for production serving on H100 GPUs. Your workload is throughput-bound (batch size 32, context 2K). Which choice is more appropriate, and why?

A. AWQ is more appropriate because INT4 weights load in fewer bytes from HBM, which is the primary bottleneck at batch size 32.  
B. FP8 is more appropriate. At batch size 32, the workload has sufficient arithmetic intensity to benefit from H100's FP8 tensor cores, which provide higher throughput per watt than the FP16 matmuls required after AWQ dequantisation. AWQ's benefit is primarily at small batch sizes where memory bandwidth is the bottleneck.  
C. AWQ and FP8 are equivalent in throughput at batch size 32; the only trade-off is accuracy, where FP8 is slightly worse.  
D. Neither is appropriate for production; FP32 is required for accurate batch-32 inference on H100.

---

## Q5

A team observes that their TensorRT-LLM deployment shows high GPU utilisation during prefill but the per-output-token latency (TPOT) is much slower than expected during the decode phase at batch size 1. A senior engineer says the decode phase is memory-bandwidth-bound, not compute-bound. Why is this the case?

A. Decode at batch size 1 is compute-bound because each forward pass processes only one token, requiring minimal floating-point operations.  
B. During autoregressive decode at batch size 1, each token step requires streaming the full weight matrices from HBM to compute units, performing a tiny amount of arithmetic (one token's projections), and then discarding. The arithmetic intensity — FLOPs per byte loaded — is extremely low, so the bottleneck is HBM bandwidth, not tensor core throughput. Quantisation to INT4 or INT8 reduces the bytes loaded per weight, directly improving TPOT.  
C. TPOT is slow because continuous batching is not enabled; enabling it will shift the workload from memory-bound to compute-bound.  
D. The decode phase is memory-bandwidth-bound specifically because of the KV cache, not the weight matrices — removing the KV cache by disabling caching will fix this.

---

## GPU Acceleration and Optimisation (Q6–Q9)

## Q6

**Numerical**

A training job uses 3D parallelism with TP=4, PP=8, DP=4. The team wants to verify the world size (total GPUs) and understand the communication pattern for the DP all-reduce.

What is the total world size? And for a gradient all-reduce under data parallelism, which NCCL collective is used and how many GPUs participate in each all-reduce operation?

A. World size = 128; each all-reduce spans all 128 GPUs simultaneously.  
B. World size = 64; each DP all-reduce spans 4 GPUs (the DP replicas); the TP and PP ranks within each replica do not participate.  
C. World size = 16; each all-reduce spans 16 GPUs.  
D. World size = 32; each DP all-reduce spans 32 GPUs.

---

## Q7

A team is diagnosing slow training throughput on an 8×H100 node. Nsight Systems shows that all-gather operations for FSDP parameter reconstruction are taking a disproportionate share of wall-clock time. Training is using ZeRO-3 (FSDP). Which change is most likely to reduce this overhead?

A. Switch from ZeRO-3 to ZeRO-1; ZeRO-3's parameter sharding requires an all-gather before every layer's forward pass, whereas ZeRO-1 only shards optimiser states and does not need per-layer all-gathers.  
B. Increase the micro-batch size to improve GPU utilisation and amortise the fixed all-gather cost over more work per batch.  
C. Reduce tensor parallelism degree — tensor parallelism adds all-gather overhead, not FSDP.  
D. Disable NVLink and use PCIe; PCIe bandwidth forces FSDP to reduce the frequency of all-gathers.

---

## Q8

You are selecting the GPU sharing mechanism for a multi-tenant GPU cluster. Three teams share the same A100 cluster: Team A runs safety-critical inference and must have strict memory isolation from other workloads; Team B runs batch training jobs that can share compute with others; Team C operates in virtual machines. Which assignment of sharing mechanisms is correct?

A. Team A: MPS; Team B: MIG; Team C: vGPU  
B. Team A: MIG (hardware-partitioned isolation); Team B: MPS (low-overhead process sharing for same-user workloads); Team C: vGPU (hypervisor-level GPU sharing for virtual machines)  
C. Team A: vGPU; Team B: MIG; Team C: MPS  
D. All teams: MIG — it is the most secure and should always be used in multi-tenant settings.

---

## Q9

A team is profiling a custom attention kernel using Nsight Compute. The roofline analysis shows the kernel is sitting in the memory-bandwidth-limited region — its achieved FLOPs/s is much lower than the theoretical compute ceiling but close to the HBM bandwidth ceiling. Which optimisation is most directly targeted at this bottleneck?

A. Increase the number of tensor cores allocated to the kernel by raising the thread block size.  
B. Tile the computation to use on-chip SRAM (shared memory) and keep intermediate results in registers as long as possible, reducing the number of round-trips to HBM. This is the approach used by FlashAttention.  
C. Apply INT8 quantisation to the activation tensors, which reduces the arithmetic count and brings the kernel closer to the compute ceiling.  
D. Increase the batch size — larger batches improve GPU utilisation, which indirectly raises the effective FLOPs/s.

---

## Prompt Engineering (Q10–Q13)

## Q10

A developer is building a multi-step agentic pipeline. The pipeline uses an LLM to query a company knowledge base (via search tool), execute read-only SQL queries (via database tool), and summarise results. During testing, an adversarial document in the knowledge base successfully causes the model to attempt a SQL `DROP TABLE` command. Which defence most directly addresses this threat?

A. Add an explicit prohibition in the system prompt: "Never execute destructive SQL commands."  
B. Apply grammar-constrained decoding to the SQL generation step to restrict the output to a safe subset of SQL (SELECT-only schema).  
C. Use a smaller, less capable model that is less likely to follow complex injection instructions.  
D. Restrict the database tool's permissions to read-only at the database layer, and apply an output schema validator that rejects any generated SQL containing data-modification keywords before execution. Defence-in-depth.

---

## Q11

A product team wants to use a large language model to answer customer questions about a software product, citing specific sections of the documentation. They have 200 documentation pages. The team is debating between context stuffing (inserting all documentation into the prompt) and a RAG pipeline. Which reasoning best guides this decision?

A. Context stuffing is superior because the model can attend to any part of the documentation without retrieval latency.  
B. The decision turns on context length and cost. 200 pages is typically 100 000–200 000 tokens; at this scale, context stuffing becomes expensive per query, and retrieval quality of the LLM may degrade at very long contexts. A RAG pipeline retrieves relevant chunks, reducing token cost and often improving response quality for factual look-up. The RAG approach is preferred unless the documentation is small enough to fit comfortably within the model's effective context window and per-query cost is not a constraint.  
C. RAG is always superior to context stuffing for any corpus size.  
D. Context stuffing is only suitable for encoder-only models; decoder-only models should always use RAG.

---

## Q12

A team is using self-consistency decoding to improve the reliability of a chain-of-thought reasoning system. They sample 10 CoT paths and majority-vote the final answers. They observe that all 10 paths produce the same incorrect answer. Which of the following conditions most likely explains this failure mode?

A. The temperature was too high, causing the reasoning paths to diverge and cancel each other out in the majority vote.  
B. The model's parametric knowledge is incorrect for this question — all reasoning paths start from the same wrong premise, so self-consistency amplifies the error rather than correcting it. Self-consistency improves variance reduction, not bias correction.  
C. Self-consistency requires at least 20 samples to be effective; 10 is insufficient.  
D. The majority vote aggregation was applied incorrectly; the correct aggregation is the average of final answers, not the mode.

---

## Q13

A team is deploying a customer support assistant and wants the model to decline requests outside of a defined topic scope. They implement two layers: (1) a detailed system-prompt prohibition, and (2) NeMo Guardrails with a Colang dialogue flow that catches off-topic intents. A colleague questions whether both layers are necessary. Which argument for retaining both layers is most accurate?

A. NeMo Guardrails is a training-time tool; the system prompt is the only runtime mechanism, so one of the two is redundant.  
B. The system prompt provides probabilistic soft enforcement at the LLM level but can be bypassed by adversarial prompts. NeMo Guardrails operates as an independent runtime layer that intercepts and classifies requests *before* they reach the model and *after* the model responds, providing hard enforcement. Both layers serve different functions and together reduce the attack surface.  
C. The two layers will conflict with each other — the Guardrails system will override the system prompt for legitimate requests and incorrectly block them.  
D. One layer is sufficient; adding a second introduces latency without security benefit.

---

## Fine-Tuning (Q14–Q17)

## Q14

**Numerical**

You are applying LoRA with rank r = 16 and alpha α = 32 to a single linear projection layer W₀ ∈ ℝ^{4096 × 4096} in a 7B transformer. The base model has 7 billion parameters in total.

(a) How many trainable parameters does this single LoRA adapter (matrices A and B) add?  
(b) What is the effective scaling factor applied to the low-rank update ΔW = BA?

A. (a) 131 072 parameters; (b) scaling factor = 2.0  
B. (a) 65 536 parameters; (b) scaling factor = 1.0  
C. (a) 131 072 parameters; (b) scaling factor = 1.0  
D. (a) 4 096 parameters; (b) scaling factor = 2.0

---

## Q15

A team is choosing between DPO and RLHF/PPO for a preference tuning run. They have 20 000 human preference pairs (prompt, chosen response, rejected response). Their cluster has 8 × A100s. They want to minimise training complexity and infrastructure requirements. Which recommendation is appropriate and why?

A. RLHF/PPO — it produces better alignment than DPO for large datasets and is the industry standard for production systems.  
B. DPO — it eliminates the separate reward model and the PPO actor-critic training loop, using only a simple binary cross-entropy loss over the preference pairs with a frozen reference model. It is substantially simpler to implement, more training-stable, and empirically competitive with PPO on summarisation and dialogue tasks.  
C. DPO — it eliminates both the reward model and the frozen reference model, making it the simplest possible training setup.  
D. RLHF/PPO — DPO requires online sampling during training, which requires more infrastructure than RLHF.

---

## Q16

A practitioner is fine-tuning a model using Axolotl with QLoRA. They set rank r = 64 to capture more task-specific expressiveness. A colleague warns that doubling the rank from 32 to 64 will significantly increase memory usage. Is the colleague correct, and why?

A. Yes — doubling the rank doubles the number of adapter parameters and doubles the adapter's activation memory footprint, which is significant relative to the base model.  
B. No — the LoRA adapter parameters (A ∈ ℝ^{r × k}, B ∈ ℝ^{d × r}) scale as 2 × r × d per layer. For a 4096-dimensional layer, rank 64 adds 2 × 64 × 4096 ≈ 524 K parameters, compared to ≈ 262 K for rank 32. This is modest — tens of MB — relative to the multi-GB base model loaded in 4-bit. The colleague overestimates the practical impact.  
C. Yes — QLoRA at rank 64 requires the base model to be stored in FP16 rather than INT4, doubling the base model memory.  
D. No — LoRA adapters are always stored in INT4 regardless of rank, so memory is constant.

---

## Q17

A team is evaluating whether to use SFT (supervised fine-tuning) alone, or SFT followed by DPO, for a code assistant. In human evaluations, SFT-only outputs are rated as technically correct 78% of the time but are described as "unhelpful" in tone and format. DPO training on preference pairs (preferred: concise and well-formatted; rejected: verbose and poorly structured) raises the helpfulness rating to 91%, with no statistically significant change in technical correctness. What does this result demonstrate about DPO's role in the training pipeline?

A. DPO improves factual accuracy — the technical correctness improvement from 78% to 91% is attributable to DPO fixing hallucinations.  
B. DPO tunes *behaviour and style* (tone, format, conciseness) relative to a reference SFT model, without necessarily improving or degrading the underlying task capability. The SFT stage installs task knowledge; DPO aligns output style to human preference. The results are consistent with this interpretation.  
C. SFT is unnecessary if DPO is applied; DPO alone from a base model achieves the same result.  
D. DPO requires that the SFT model first achieve near-perfect accuracy; 78% is too low a baseline for DPO to function correctly.

---

## Data Preparation (Q18–Q20)

## Q18

A team is constructing a pre-training dataset using NeMo Curator. They discover that approximately 30% of documents in the corpus are near-duplicates of each other (similar but not identical text), inflating the effective dataset size and likely causing the model to memorise those documents. Which NeMo Curator capability directly addresses this?

A. NeMo Curator's language identification module — it removes duplicate languages, reducing duplication.  
B. NeMo Curator's exact and fuzzy deduplication (MinHash-based) — it identifies near-duplicate documents across the corpus and removes them, preventing over-representation of any single document.  
C. NeMo Curator's quality filtering module — it removes low-quality documents, which tend to be repetitive.  
D. NeMo Curator's data blending module — it reweights source distributions to reduce duplicates from over-represented sources.

---

## Q19

A RAG pipeline is ingesting a 500-page product manual stored as a PDF. The document contains structured tables of specifications, prose descriptions, and numbered procedure steps. The current fixed-size text chunker (512 tokens, 64-token overlap) produces poor retrieval results for specification-related queries. Which intervention is most appropriate?

A. Increase the chunk size to 2048 tokens so each chunk captures a full table.  
B. Use layout-aware document parsing (e.g., Unstructured or LlamaParse) to extract table cells, paragraphs, and procedure steps as structurally labelled elements before chunking, then chunk each element type separately. Attach metadata (section heading, page number, element type) to enable filtered retrieval.  
C. Switch from HNSW to IVF indexing; IVF handles structured data better.  
D. Replace the PDF with an HTML version; PDF parsing is inherently too imprecise for structured retrieval.

---

## Q20

A team is building a supervised fine-tuning dataset from internal customer support transcripts. They have 50 000 raw conversational turns. A data scientist proposes using all 50 000 turns to maximise dataset size. What is the most important data quality consideration that should take precedence over dataset size, and what practical step addresses it?

A. Token count — each training example must contain exactly 512 tokens. Pad or truncate all examples.  
B. Format consistency — all examples must use the same chat template. Verify and enforce this before training.  
C. Quality over quantity — low-quality, inconsistent, or incorrectly resolved support transcripts will degrade model behaviour even if present in small numbers. A filtering pass using quality heuristics or a classifier to retain high-quality examples is more valuable than maximising raw count. A curated 5 000-example subset often outperforms a noisy 50 000-example full dataset.  
D. Deduplication — all 50 000 turns must be deduplicated before training to avoid overfitting.

---

## Model Deployment (Q21–Q23)

## Q21

You are setting up Triton Inference Server to serve three different models simultaneously: a BF16 LLaMA-3-8B running via TensorRT-LLM, an ONNX-format embedding model, and a custom Python pre-processing step. Is this a supported Triton configuration and what is its architecture?

A. No — Triton can only serve one model type per instance; separate servers are needed for LLM and ONNX models.  
B. Yes — Triton is multi-backend: it can simultaneously serve TensorRT-LLM engines (via the TRT-LLM backend), ONNX Runtime models, and Python model scripts within a single server instance. A model ensemble can chain the Python pre-processor, the ONNX embedder, and the LLM backend into a single logical request.  
C. Yes — but only if all models use the same precision; mixing BF16 and FP32 backends requires separate server instances.  
D. No — the TensorRT-LLM backend is a separate product that cannot co-exist with the ONNX Runtime backend in the same Triton instance.

---

## Q22

A team wants to deploy multiple LoRA fine-tuned variants of the same 7B base model (one per customer, ~50 customers) on a single GPU server. Each LoRA adapter has rank 16. They ask whether it is better to merge each adapter into a separate full model or to use multi-LoRA serving. Which approach is correct for this scale?

A. Merge each adapter — merging is simpler to implement and 50 merged models can be loaded efficiently on a multi-GPU server.  
B. Multi-LoRA serving (S-LoRA or vLLM's multi-LoRA support) — keep the base model weights resident on GPU and manage the 50 small adapters (rank-16 LoRA on 7B ≈ a few hundred MB each) as a separate pool. This avoids loading 50 × 14 GB of merged models and allows dynamic adapter swapping per request.  
C. Merge each adapter — multi-LoRA serving requires each adapter to be on a different GPU, which is impractical at scale.  
D. Neither — 50 customers requires 50 separate model instances; there is no way to share a base model across customers without degrading isolation.

---

## Q23

A platform team is evaluating NIM versus manually assembling TensorRT-LLM and Triton for production LLM serving. Their use case is a standard Llama-3-70B deployment on 4×H100 nodes with an OpenAI-compatible API. Which is the more appropriate choice and what is the key operational trade-off?

A. Manual assembly is always preferable — NIM is a black-box that hides important configuration details.  
B. NIM bundles a pre-validated TensorRT-LLM engine, Triton backend, OpenAI-compatible API, health checks, and telemetry into a single container, avoiding the need to manage TRT-LLM compilation, Triton configuration, and API compatibility separately. For a standard model deployment without unusual customisation requirements, NIM is the lower-complexity choice. The trade-off is reduced ability to customise the TRT-LLM compilation flags or serve model variants not in the NIM catalog.  
C. NIM requires NVIDIA AI Enterprise licensing even for development; manual assembly is free and equally capable.  
D. Triton Inference Server is not included in NIM; a manual Triton setup is always needed alongside NIM.

---

## Evaluation (Q24–Q25)

## Q24

A team has trained two models: Model A (SFT only) and Model B (SFT + DPO). They run both on an internal benchmark suite (MMLU, GSM8K, IFEval) and find that Model B scores marginally higher on IFEval but the same on MMLU and GSM8K. They conclude Model B is better. A senior evaluator raises a concern about benchmark contamination. What is the most specific concern they are likely raising?

A. MMLU and GSM8K questions are not appropriate for evaluating instruction-following models.  
B. The DPO preference data used to train Model B may contain examples drawn from or similar to the IFEval benchmark, inflating Model B's IFEval score without representing genuine instruction-following improvement. The marginal improvement on a benchmark where contamination is plausible is weak evidence.  
C. Running static benchmarks is always invalid for comparing fine-tuned models; only human evaluation counts.  
D. GSM8K is known to have very high contamination in base model pre-training data, making it unsuitable for comparison of any models trained on internet-scale corpora.

---

## Q25

A product team is measuring the quality of a RAG chatbot. Their RAGAS evaluation shows faithfulness = 0.95, context precision = 0.88, but context recall = 0.52. A team member proposes increasing the number of retrieved chunks (top-k) from 5 to 15 as the primary remedy. Is this the right intervention, and what is the key risk?

A. Yes — increasing top-k is always the correct remedy for low context recall; there is no meaningful risk.  
B. Yes, but with caution. Low context recall (0.52) indicates that the retrieved chunks are missing information needed to answer questions. Increasing top-k will likely improve recall. The key risk is context bloat: passing 15 chunks to the LLM increases token cost and can introduce irrelevant chunks that reduce faithfulness or cause the model to produce incoherent answers due to the "lost in the middle" attention failure mode. Combining increased top-k with a reranking step maintains precision.  
C. No — low context recall is a retrieval training problem; top-k has no effect on recall.  
D. No — context recall is irrelevant for production RAG; only faithfulness matters.

---

## Production Monitoring and Reliability (Q26–Q27)

## Q26

A team has deployed a fine-tuned LLM for financial document summarisation. Three weeks after deployment, users report that summaries of recent regulatory documents contain terminology inconsistent with the documents. The model was not retrained. Monitoring shows that input embedding distributions have shifted significantly compared to the baseline recorded at deployment. What has most likely occurred and what is the appropriate response?

A. The model has been jailbroken by users; revoke all user API keys and redeploy.  
B. Distribution shift — the input data (regulatory documents) has evolved since training, and the embedding distribution shift confirms this. The model's parametric knowledge may be inconsistent with new terminology. The appropriate response is to update the RAG knowledge base (if RAG is in use) or retrain/fine-tune on recent documents, and to establish ongoing drift monitoring with automated alerts.  
C. The model has suffered catastrophic forgetting due to continuous online learning; disable online learning.  
D. The summarisation prompt has changed; roll back to the original prompt.

---

## Q27

A team is designing a CI/CD pipeline for LLM deployments. They want to gate each new model version (new fine-tune or system prompt change) from being deployed to production unless it passes a regression test. Which evaluation practice most directly supports this gate?

A. Run the full MMLU benchmark before each deployment; any score drop of more than 1% blocks release.  
B. Maintain a versioned golden set — a curated collection of inputs with known-good outputs (human-verified) — and require all regression metrics (accuracy on the golden set, refusal rate, toxicity classifier score) to stay within defined thresholds before each deployment. The golden set evolves alongside the application.  
C. Human evaluation of 100 random samples before each deployment is the only valid gate.  
D. Monitor production metrics (latency, error rate) after deployment as a post-hoc gate; roll back if metrics degrade within 24 hours.

---

## LLM Architecture (Q28–Q29)

## Q28

**Numerical**

A decoder-only transformer model has the following configuration:
- n_layers = 32
- d_model = 4096
- n_heads = 32
- d_ff = 11 008 (SwiGLU, 3-matrix FFN)
- Vocabulary size V = 32 000

Using the approximate per-layer formula (attention: 4 × d_model², FFN: 3 × d_model × d_ff), estimate:

(a) The parameter count per transformer layer.  
(b) The total parameter count (including embedding matrix, ignoring weight tying).

A. (a) ~202 M per layer; (b) ~32 × 202 M + 131 M ≈ 6.6 B total  
B. (a) ~134 M per layer; (b) ~4.4 B total  
C. (a) ~268 M per layer; (b) ~8.7 B total  
D. (a) ~67 M per layer; (b) ~2.3 B total

---

## Q29

**Numerical**

A model uses Grouped-Query Attention (GQA) with n_query_heads = 32 and n_kv_heads = 8. The model has 32 layers, head_dim = 128. You are serving at sequence length 4096 in FP16.

(a) What is the KV cache memory per sequence?  
(b) Compared to Multi-Head Attention (MHA) with 32 KV heads, by what factor does GQA reduce the KV cache?

A. (a) 0.5 GB per sequence; (b) 4× reduction  
B. (a) 0.25 GB per sequence; (b) 4× reduction  
C. (a) 0.5 GB per sequence; (b) 4× reduction — but only applies to the key cache; the value cache is unchanged.  
D. (a) 1 GB per sequence; (b) 2× reduction

---

## Safety, Ethics, and Compliance (Q30)

## Q30

A large enterprise is deploying a general-purpose LLM API for internal use across business units including HR, legal, and finance. The model will process employee performance reviews, legal contracts, and financial forecasts. Under the NIST AI Risk Management Framework (AI RMF 1.0), the CISO asks which RMF function covers establishing organisational accountability structures, risk tolerance policies, and governance ownership for this deployment before any technical work begins. The team also needs to know which function covers ongoing quantification and tracking of identified risks once the system is live.

A. MAP covers governance and accountability; MANAGE covers quantification.  
B. GOVERN covers governance, accountability structures, risk tolerance policies, and ownership — it is the organisational readiness function applied before and throughout deployment. MEASURE covers ongoing quantification, analysis, and tracking of identified risks once the system is live.  
C. GOVERN covers governance; MAP covers quantification.  
D. MANAGE covers governance; MEASURE covers risk identification.

---

## Answer key

**Format:** Question number — Correct answer — Rationale.

---

1. **C** — KV cache = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes = 2 × 32 × 32 × 128 × 8192 × 2 = 4 294 967 296 bytes ≈ 4 GB per sequence. With 14 GB consumed by weights, 6 GB remains. 4 GB < 6 GB, so 1 sequence fits with ~2 GB headroom. A second sequence would require 8 GB of remaining capacity (2 × 4 GB) but only 6 GB is free — it does not fit. **Distractor B** uses the 4 096-token figure rather than 8 192, halving the correct result.

2. **B** — KV cache per sequence = 2 × 32 × 32 × 128 × 4096 × 2 = 2 GB (same formula at 4 096 tokens). For 4 sequences: 4 × 2 = 8 GB. With 3.5 GB INT4 weights: 3.5 + 8 = 11.5 GB, which exceeds 10 GB. The fix is to reduce batch size to 3 (3 × 2 + 3.5 = 9.5 GB) or to quantise the KV cache to INT8 (~1 GB/sequence), giving 3.5 + 4 = 7.5 GB. **Distractor C** uses 1 GB/sequence — half the correct value — which would correspond to INT8 KV quantisation, not FP16 as stated.

3. **B** — Speculative decoding's benefit depends on the acceptance rate and the ratio of speculative overhead to per-target-step savings. At 60% acceptance with k = 5 draft tokens, roughly 3 tokens are committed per target verification pass; if the verification pass is not much slower than 5 individual target decodes (and for a 70B model, one forward pass costs roughly the same regardless of whether it processes 1 or k+1 tokens), the effective speedup is ~3×. **Distractor D** is incorrect — speculative decoding benefits are actually *lower* at large batch sizes, where the 70B target is already compute-bound; the gains are most pronounced at small batch sizes.

4. **B** — At batch size 32, the workload accumulates sufficient arithmetic intensity (many rows being multiplied per weight-load from HBM) to benefit from H100's native FP8 tensor cores, which provide roughly 2× the throughput of FP16 matmuls. AWQ keeps compute in FP16 (it dequantises before matmul), so its benefit is limited to reducing memory bandwidth pressure — most valuable at batch size 1. **Distractor A** has the analysis backwards: AWQ's advantage is memory-bandwidth relief, which matters most at small batch sizes, not large ones.

5. **B** — During decode at batch size 1, each step processes a single token position. The matmul is a matrix-vector product (one token × weight matrix), performing d_model² FLOPs but loading d_model² × bytes_per_weight from HBM. The arithmetic intensity is ~1 FLOP/byte for FP16 — well below the A100/H100's compute-to-bandwidth ratio, placing the operation firmly in the memory-bandwidth-limited regime. INT4 quantisation reduces bytes loaded by 4×, directly improving TPOT. **Distractor D** is wrong: the weight matrices are the dominant bandwidth consumer during decode; the KV cache is also loaded but the weight streaming is the primary bottleneck.

6. **B** — World size = TP × PP × DP = 4 × 8 × 4 = 128. Under data parallelism, each DP replica holds the same parameters and processes a different micro-batch; at the end of the backward pass, gradients are synchronised across the 4 DP replicas using an all-reduce. The all-reduce spans the 4 DP replicas — not all 128 GPUs. TP and PP ranks within each DP replica are not part of the DP gradient sync collective. **Distractor A** is wrong: the all-reduce spans only the DP replicas (4 GPUs), not the full world size.

7. **A** — ZeRO-3/FSDP shards all parameters; before each layer's forward pass, an all-gather reconstructs the full layer parameters. This introduces an all-gather every layer, and the overhead can dominate on fast compute hardware (H100) where the compute per layer is fast but the communication cannot fully overlap. Switching to ZeRO-1 keeps parameters unsharded (only optimiser states are sharded), eliminating per-layer all-gathers entirely. **Distractor B** (increasing micro-batch size) would help amortise cost but does not eliminate the structural overhead of per-layer all-gathers.

8. **B** — MIG provides hardware-enforced memory and compute isolation and is the correct choice for safety-critical workloads requiring strict isolation (MIG is available on A100). MPS allows multiple CUDA processes to share a single GPU context efficiently but without memory isolation — appropriate for same-user workloads. vGPU provides GPU virtualisation for virtual machines. **Distractor D** is wrong: MIG is not always preferred; it reduces per-instance throughput and is not supported on consumer/prosumer GPUs.

9. **B** — A memory-bandwidth-limited kernel is bottlenecked on HBM read/write throughput. The remedy is to tile the computation to fit intermediate results in on-chip SRAM, minimising HBM round-trips. This is exactly the approach FlashAttention takes for the attention score matrix — it tiles Q, K, V blocks into SRAM and never materialises the full L×L score matrix in HBM. **Distractor C** (INT8 quantisation) reduces bytes loaded but is an approximation; tiling is the structural fix that eliminates unnecessary HBM traffic.

10. **D** — The most robust defence is minimising capability exposure at the tool layer: restrict the SQL tool to read-only permissions at the database level, so that even if injection succeeds, the tool cannot execute destructive commands. Adding an output schema validator that rejects non-SELECT SQL before execution provides a second layer. **Distractor A** (system prompt prohibition) is soft enforcement only and does not prevent injection. **Distractor B** (grammar-constrained decoding) is strong for output structure but must be combined with tool-layer restrictions for defence-in-depth.

11. **B** — Context stuffing is technically possible for 200 pages but becomes expensive (100 000–200 000 tokens per query) and may degrade LLM factual precision at very long contexts. RAG reduces per-query cost by retrieving only relevant chunks and is the standard architectural pattern for large documentation corpora. The caveat is that for very small, stable corpora (a few pages) context stuffing may be simpler and effective. **Distractor C** is wrong — for genuinely small corpora, context stuffing is a valid simpler alternative.

12. **B** — Self-consistency reduces *variance* (inconsistency across samples from stochastic sampling). If all 10 paths agree on an incorrect answer, the model's prior distribution is biased toward the wrong answer — self-consistency cannot correct a bias in the model's knowledge, only variance in its sampling. **Distractor A** is backwards: high temperature would cause paths to *diverge*, not converge on the same wrong answer.

13. **B** — The system prompt is soft enforcement (tokens that can be partially overridden). NeMo Guardrails operates as an independent runtime layer, classifying intent before the LLM is called and validating output after. The two layers are complementary: the system prompt shapes model behaviour for normal requests; Guardrails provides a hard enforcement boundary that is independent of the LLM's behaviour. **Distractor A** is factually wrong — NeMo Guardrails is a runtime (inference-time) framework, not a training-time tool.

14. **A** — (a) Matrix A ∈ ℝ^{r × k} = ℝ^{16 × 4096} has 65 536 parameters. Matrix B ∈ ℝ^{d × r} = ℝ^{4096 × 16} has 65 536 parameters. Total = 131 072. (b) The LoRA scaling factor is α/r = 32/16 = 2.0. The effective update is (α/r) × BAx = 2 × BAx. **Distractor C** gets the parameter count right but incorrectly computes the scaling as 1.0, confusing α (32) with r (16) in the ratio.

15. **B** — DPO is the correct recommendation for teams that want minimum infrastructure and training complexity. It requires no reward model, no PPO actor-critic loop, no online sampling — just a frozen reference model and a binary cross-entropy loss on preference pairs. It is empirically competitive with PPO on alignment quality. **Distractor C** is the most dangerous distractor: DPO does *not* eliminate the reference model — it uses a frozen reference model to compute log-probability ratios that form the implicit reward.

16. **B** — For a 4096×4096 projection with rank 32: A = 32×4096 ≈ 131 K params, B = 4096×32 ≈ 131 K params; total ≈ 262 K. At rank 64: 524 K total — an increase of ~262 K parameters, which is a few MB even in FP32. This is negligible relative to the 7B base model's 3.5 GB footprint in 4-bit. The colleague's concern is technically correct but practically irrelevant at these scales. **Distractor C** is wrong: QLoRA always keeps the base model in 4-bit regardless of adapter rank.

17. **B** — The technical correctness metric (78% to 91% — noting the jump here is comparing SFT to SFT+DPO and the technical correctness was "no statistically significant change") reflects that SFT installed the task knowledge. DPO moved the helpfulness rating by adjusting tone, format, and conciseness. This is consistent with DPO's design: it optimises for human preferences expressed in the preference pairs, which in this case were about style, not factual correctness. **Distractor A** misattributes the helpfulness improvement (a style metric) to accuracy improvement.

18. **B** — NeMo Curator's deduplication capability (exact and fuzzy deduplication via MinHash) is specifically designed to remove near-duplicate documents. MinHash identifies documents with high Jaccard similarity of their token n-gram sets, enabling fuzzy matching. **Distractor C** (quality filtering) removes low-quality documents but is not targeted at duplication specifically.

19. **B** — Layout-aware parsing with metadata attachment is the correct solution for PDFs with mixed structured and unstructured content. Extracting tables as table objects, prose as paragraphs, and procedures as step sequences — then attaching section metadata — produces coherent, filterable chunks. **Distractor A** (larger chunks) would capture tables but mix table content with unrelated prose, worsening retrieval precision.

20. **C** — Dataset quality dominates quantity for SFT. Low-quality transcripts (incorrectly resolved issues, inconsistent formats, ambiguous instructions) teach the model bad behaviours that are hard to unlearn. A high-quality 5 000-example subset with verified resolutions consistently outperforms a noisy 50 000-example set. **Distractor B** (format consistency) is important but is a prerequisite, not the primary trade-off consideration — the core question is quality vs quantity.

21. **B** — Triton Inference Server is explicitly multi-backend: it can simultaneously serve TensorRT-LLM engines, ONNX Runtime models, and Python model scripts within a single instance, and can chain them as a model ensemble. This is a core design feature documented in the Triton architecture. **Distractor A** is incorrect and reflects a common misconception.

22. **B** — Multi-LoRA serving keeps a single 14 GB base model resident on GPU and manages the small adapters (rank-16 LoRA on 7B ≈ ~200–400 MB per adapter) as a pool. 50 such adapters require only ~10–20 GB of adapter storage total. Merging 50 separate 14 GB models would require 700 GB of storage and loading a different 14 GB model for each customer. **Distractor A** is impractical at this scale.

23. **B** — For a standard supported model on NVIDIA hardware with no custom compilation requirements, NIM is the lower-complexity path: it eliminates TRT-LLM engine compilation, Triton configuration, and API compatibility work. The legitimate trade-off is reduced customisation flexibility. **Distractor C** is partially true (NIM requires AI Enterprise for production SLA) but developer-tier access is available; "manual assembly is free" is correct but does not negate NIM's operational advantages.

24. **B** — Benchmark contamination is most acute when the fine-tuning data (specifically the DPO preference pairs) may overlap with or closely resemble the benchmark's test examples. A marginal IFEval improvement — especially when neither MMLU nor GSM8K changed — is weak evidence if the DPO data sources are not audited for IFEval overlap. **Distractor D** is also true (GSM8K contamination in base model training is documented) but the question specifically asks about the concern the evaluator raises in the context of *this comparison*, which is the DPO-IFEval overlap.

25. **B** — Low context recall means the first-stage retrieval is failing to return chunks that contain necessary information. Increasing top-k directly addresses this by widening the retrieval net. The risk is real: a "lost in the middle" attention failure — where the LLM underweights relevant information in the middle of a long context — and increased cost per query. Adding a cross-encoder reranker after expanding top-k maintains precision while improving recall. **Distractor A** ignores the real risk.

26. **B** — The embedding distribution shift of input documents is a diagnostic indicator of covariate shift (the input distribution has changed). In this context, regulatory documents have evolved terminology post-training. The appropriate response is to update the retrieval index (if RAG is used) or schedule retraining, and to alert on future distribution shifts. **Distractor A** (jailbreak) is ruled out by the embedding distribution shift evidence, which indicates input content change, not adversarial attack.

27. **B** — A versioned golden set with defined regression thresholds is the industry-standard approach for gating LLM deployments. It provides fast, deterministic, reproducible results that can be run in CI without human involvement. **Distractor D** (post-hoc production monitoring) is valuable but is not a deployment gate — it catches failures after users have already been affected.

28. **A** — Attention per layer: 4 × d_model² = 4 × 4096² = 67 108 864 ≈ 67 M. FFN per layer (SwiGLU, 3 matrices): 3 × d_model × d_ff = 3 × 4096 × 11 008 ≈ 135 M. Per layer total ≈ 202 M. 32 layers: 32 × 202 M ≈ 6 464 M ≈ 6.46 B. Embedding: 32 000 × 4096 = 131 M. Grand total ≈ 6.46 B + 0.13 B ≈ 6.6 B. This rounds to "7B" in practice when unembedding and layer norm parameters are included. **Distractor B** uses a smaller per-layer estimate inconsistent with the formula given.

29. **A** — KV cache = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes = 2 × 32 × 8 × 128 × 4096 × 2 = 536 870 912 bytes ≈ 0.5 GB per sequence. MHA equivalent (32 KV heads): 2 × 32 × 32 × 128 × 4096 × 2 ≈ 2 GB. Ratio: 2 GB / 0.5 GB = 4× reduction. GQA with 8 KV heads against 32 query heads is a 4× reduction (32 / 8 = 4). **Distractor B** gives the same reduction factor (4×) but halves the KV cache size to 0.25 GB, which is incorrect; the calculation gives 0.5 GB.

30. **B** — GOVERN is the NIST AI RMF function that covers organisational culture, policies, accountability structures, and risk tolerance — it is specifically described as the foundation that must be in place before and throughout the other functions. MEASURE covers quantification, analysis, and tracking of identified risks over time (ongoing, once the system is live). **Distractor A** confuses MAP with GOVERN: MAP is about identifying and categorising risks in context (who is affected, what can go wrong), not about organisational accountability structures.

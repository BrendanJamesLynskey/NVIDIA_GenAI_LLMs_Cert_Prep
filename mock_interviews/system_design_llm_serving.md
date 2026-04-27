# LLM Serving System Design — Mock Interview Scenarios

Three open-ended scenarios with worked model answers. The goal is not to list
components but to reason through trade-offs explicitly, showing numerical
grounding where it informs the decision. Each scenario follows the same
structure: brief, clarifying questions, design, trade-offs, and a rubric
distinguishing a competent answer from a strong one.

---

## Scenario 1: Enterprise RAG over a Knowledge Base

### Brief

A mid-size enterprise wants to build a question-answering assistant over its
internal knowledge base. The corpus is 10–50 GB of mixed-modality documents:
PDFs (technical specifications, policy documents), HTML pages (internal wiki),
and Confluence pages (project documentation). The corpus changes daily as new
documents are published and existing ones are revised. The system must support
multiple business units, each of which has strict access controls — a finance
analyst must not retrieve HR documents even if both live in the same index.
Target answer latency is p95 < 3 s end-to-end.

The system is expected to handle several hundred concurrent users during
business hours, answer factual questions with source citations, and degrade
gracefully when no relevant information exists rather than hallucinating.

The 3 s budget must cover ingestion-time embedding of new documents, query-time
retrieval, optional reranking, and generation. In practice, the generation step
against a hosted LLM typically consumes 1–2 s of that budget, leaving 1–1.5 s
for everything upstream of the generation call.

### Clarifying questions you'd ask

- **ACL granularity**: are access controls at document level, section level, or
  user-group level? Do ACLs change frequently (e.g., project access revoked when
  a team member leaves)?
- **Query distribution**: are queries mostly factual lookups within a single
  document, or multi-hop analytical questions requiring synthesis across many
  documents? The answer determines whether graph-augmented retrieval is worth
  the construction cost.
- **Update cadence details**: "daily-changing" could mean 10 new documents per
  day or 10,000. The ingestion pipeline architecture differs substantially — a
  nightly batch job vs a near-real-time event-driven pipeline.
- **Language and modality scope**: is the corpus English-only, or multilingual?
  Are tables and figures within PDFs meaningful content that must be indexed, or
  can they be skipped?
- **Latency budget allocation**: is the 3 s p95 an end-to-end wall-clock SLO
  including network RTTs to the LLM API, or purely the retrieval-plus-generation
  compute time?
- **Evaluation baseline**: is there an existing labelled QA test set for
  retrieval recall measurement, or will one need to be constructed? Without this,
  iterating on retrieval quality is guesswork.

### Design

#### Ingestion pipeline

The primary complexity in this scenario is document diversity. A naive text
extraction strategy — treating every document as a flat string — will corrupt
table content and lose structural context from PDFs and HTML.

Use a layout-aware extraction layer (Unstructured or LlamaParse) rather than
a simple PDF text extractor. These tools emit annotated elements: paragraph,
table, heading, code block. This matters for chunking: a table cell should
stay with its header row; a heading should travel with the section it
introduces rather than being separated by a chunk boundary.

Chunking strategy: semantic chunking at natural paragraph and section
boundaries, targeting approximately 400–600 tokens per chunk with a 50–100
token overlap. Fixed-size token splitting is simpler but regularly breaks
mid-sentence in policy-dense prose, degrading retrieval recall. The overlap
is necessary because relevant information often straddles chunk boundaries.
For tables: extract them as self-contained chunks with the table header
prepended, even if that pushes the chunk past the target size — a reranker
will handle the redundancy.

Metadata attached at ingest time: document ID, source system (Confluence/PDF
storage/wiki), page or section number, last-modified timestamp, owning
business unit, and ACL tags. ACL tags are the mechanism for multi-tenant
filtering downstream.

Embedding: a general-purpose bi-encoder such as `text-embedding-3-large` (1536
dimensions) or an equivalent open-weights model (e.g.,
`BAAI/bge-large-en-v1.5`, 1024 dimensions). The choice between these is not
primarily about model quality — both are strong — but about operational
considerations: a hosted embedding API avoids GPU provisioning for the ingestion
service but introduces per-call cost and latency; a locally hosted model
adds GPU infrastructure but has predictable cost and no egress latency. At
50 GB with average chunk size 500 tokens, approximate chunk count is in the
range of 2–5 million chunks. At 1024 dimensions in FP32 that is 8–20 GB of
vector storage — well within the capacity of a mid-size vector database
deployment.

Also generate sparse embeddings (BM25 via the vector database's built-in
sparse support, or a SPLADE model) for every chunk. This doubles storage but
is essential for hybrid retrieval.

#### Vector index and ACL filtering

Store embeddings and metadata in a vector database that supports filtered
ANN search — Qdrant or Weaviate are strong choices here. The critical
requirement is pre-filtering on ACL tags before the ANN search executes, not
post-filtering after retrieving k results. Post-filtering is incorrect for
access control: it can expose document existence by counting filtered-out
results, and it degrades recall when ACL filters are selective. Qdrant's
`must` filter in the query payload executes the ACL filter inside the HNSW
traversal.

Index configuration: HNSW with M=24, ef\_construction=200. Why M=24 rather
than the default M=16? At 2–5 million chunks, recall at ef\_search=64
flattens past M=24 on recall/latency curves (back-of-envelope from HNSW
benchmarks on ANN-Benchmarks); going to M=32 adds ~30% memory overhead for
<1% recall gain at this scale. For HNSW vs IVF detail see
[`RAG_02_Vector_Databases`](https://github.com/BrendanJamesLynskey/RAG_02_Vector_Databases).

#### Retrieval: hybrid BM25 + dense with RRF fusion

Neither dense nor sparse retrieval alone is optimal for an enterprise corpus
that mixes prose, tables, and technical abbreviations. Dense retrieval
handles semantic similarity and paraphrase well but will miss a query for a
specific product code or regulation number if it is an out-of-vocabulary term.
Sparse (BM25/SPLADE) is precise on exact-match but cannot handle the
"what is our policy on X?" query style where the query and document use
different vocabulary.

Run both retrievers in parallel and merge with Reciprocal Rank Fusion:

```
RRF(d) = Σ_r 1 / (60 + r(d))
```

The k=60 smoothing constant is robust to score scale differences between
BM25 and cosine similarity; it is not necessary to normalise scores. Retrieve
top-50 from each arm; merge to a candidate set of up to 100 documents.

#### Reranking

The first-stage retrieval candidate set of 50–100 chunks is passed to a
cross-encoder reranker. A cross-encoder processes the (query, chunk) pair
jointly — unlike the bi-encoder which encodes them independently — producing
a calibrated relevance score. The precision improvement justifies the latency
cost only if the reranker is fast enough to fit in the budget.

Latency check: scoring 100 candidates with a `cross-encoder/ms-marco-MiniLM-L-6-v2`
class model (6-layer MiniLM, ~22 M parameters) takes roughly 80–120 ms on a
single A10G GPU. That fits within the 1–1.5 s pre-generation budget at p95.
A larger cross-encoder (12 layers) would improve precision further but at
approximately double the latency — marginal given the bi-encoder recall is
already strong.

Pass top-5 reranked chunks to the LLM context. Beyond 5 chunks the quality
improvement is typically marginal and every additional chunk increases
generation latency and cost.

#### Generation

Use a hosted instruction-tuned LLM (e.g., GPT-4o-mini or a Nemotron model via
NIM) with the retrieved chunks injected into the context via a system prompt.
The system prompt instructs the model to answer only from the provided context
and to cite the source (document ID + section) for each factual claim.

For latency, the generation step dominates. A 3,000-token context window with
300-token output at GPT-4o-mini throughput is approximately 1–1.5 s. This
leaves 1–1.5 s for the entire upstream retrieval stack, which is achievable
with the hybrid + rerank design above.

#### Update pipeline

Daily-changing documents require a differential ingestion trigger. The
simplest robust approach: a daily job compares document IDs and last-modified
timestamps against the index metadata; new and modified documents are re-embedded
and upserted; deleted documents are removed by document ID. For truly
near-real-time updates (e.g., Confluence pages edited throughout the day),
a webhook-triggered event pipeline is more appropriate. The choice depends on
the answer to the clarifying question about update cadence.

#### Evaluation and monitoring

Evaluation at launch: construct a set of 200–500 test questions with known
relevant documents (drawn from a sample of the corpus, ideally annotated by
domain experts). Measure retrieval recall@5 on this set — the fraction of
questions where at least one relevant document appears in the top-5 retrieved
results. This is the leading indicator of system quality; generation quality
is secondary to retrieval quality.

For ongoing monitoring, deploy RAGAS to score production responses on
faithfulness (are claims grounded in the retrieved context?), answer relevancy,
context precision, and context recall. Alert on degradation — a sudden drop in
faithfulness is often a symptom of a corpus update introducing off-distribution
documents. Log retrieval latency at p50, p95, and p99 separately from
generation latency; degradation in retrieval often signals index configuration
or hardware problems; generation latency spikes are typically provider-side.

For deeper coverage of evaluation metrics and the RAGAS framework, see
[`RAG_07_Production_RAG`](https://github.com/BrendanJamesLynskey/RAG_07_Production_RAG)
and [`LLM_Hub_Evaluations`](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations).

### Trade-offs called out

- **Layout-aware extraction vs simple text split**: layout-aware parsing
  (Unstructured/LlamaParse) adds ingestion complexity and cost but is necessary
  for PDF tables and HTML structure. A plain text split would corrupt a
  significant fraction of the corpus.
- **HNSW vs IVF for the vector index**: HNSW at M=24 gives better recall at low
  ef\_search (faster queries) but is more memory-intensive than IVF. At 2–5
  million chunks, HNSW memory is manageable; above 50 million chunks, IVF with
  product quantisation would become the better trade-off.
- **Hybrid retrieval cost vs single-modality**: running two retrievers and RRF
  adds latency and infrastructure complexity versus a dense-only system. The
  recall improvement on exact-match queries in an enterprise corpus justifies
  this — dense-only systems regularly miss product codes, regulation references,
  and acronyms.
- **Cross-encoder reranking vs no reranking**: reranking adds 80–120 ms of
  latency and a GPU dependency. The precision improvement is worth it when the
  corpus is large and the first-stage retrieval is imprecise. For a small corpus
  (<100 k chunks) with high-quality embeddings, reranking may be
  disproportionately expensive.
- **Pre-filter vs post-filter for ACLs**: pre-filtering inside the ANN traversal
  is correct for access control but degrades recall when ACL filter selectivity
  is high (i.e., a user has access to only a small fraction of the corpus).
  Namespace separation (one index shard per business unit) avoids this
  degradation at the cost of operational complexity and prevents cross-unit
  retrieval.
- **Hosted LLM vs self-hosted for generation**: a hosted LLM (GPT-4o-mini,
  Claude, NIM endpoint) avoids GPU provisioning but introduces external latency
  and data-residency concerns for an enterprise. A self-hosted NIM container
  running Nemotron or Llama-3-Instruct on A10G GPUs adds infrastructure but
  keeps data on-premises and provides predictable latency.

### What "good" vs "better" looks like

A good answer describes the component pipeline correctly: extract, chunk,
embed, index, retrieve, rerank, generate, evaluate. It names reasonable tool
choices at each stage and acknowledges that dense retrieval alone is
insufficient.

A better answer reasons through the ACL problem carefully — recognising that
post-filtering is architecturally wrong for access control and explaining why
pre-filtering or namespace sharding is required. It allocates the 3 s latency
budget explicitly: ~80–120 ms for reranking, ~1–1.5 s for generation, ~50–100
ms for retrieval, leaving a clear picture of where headroom exists and what
breaks first as load increases. It notes that retrieval recall@k on a labelled
test set is the correct leading metric — not RAGAS faithfulness, which measures
a downstream property and cannot distinguish a retrieval failure from a
generation failure. It also recognises that the daily-update requirement has
different solutions depending on update volume, and asks the right clarifying
question to distinguish them.

---

## Scenario 2: 70B Model Serving with Tight Latency SLOs on a Fixed GPU Budget

### Brief

You are given a fixed GPU budget of 8× H100 80 GB SXM nodes (one node of
8 GPUs interconnected via NVLink). You must serve a Llama-3 70B parameter
instruction-tuned model with the following SLOs: p95 time-to-first-token (TTFT)
≤ 200 ms, p95 time-per-output-token (TPOT) ≤ 30 ms, and throughput maximised
subject to those constraints. The incoming request mix has input lengths
distributed between 200 and 8,000 tokens and output lengths of 200–500 tokens.

You have no option to use a smaller model — the 70B is a product requirement.
All trade-off decisions must be justified against the TTFT and TPOT SLOs.

The Llama-3 70B architecture parameters relevant to this analysis:
`n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8, head_dim=128`.

### Clarifying questions you'd ask

- **Traffic pattern**: is the load bursty (e.g., a daytime peak at 10× overnight
  baseline) or relatively flat? Bursty traffic changes the batch size
  distribution and may warrant autoscaling headroom rather than always-on
  fixed-batch configuration.
- **Output length distribution**: is the 200–500 token output range relatively
  uniform, or do a small number of requests generate much longer outputs?
  Speculative decoding acceptance rates depend on output predictability.
- **Accuracy floor**: is the model used for an application where 1–2% quality
  degradation from quantisation is acceptable, or is this a high-stakes domain
  (medical, legal) where near-lossless precision is required?
- **Request isolation requirements**: are requests from different tenants or
  users, and is there any requirement for prompt confidentiality (i.e., can
  system prompts be cached and shared across requests)?
- **Streaming vs batch responses**: does the client consume tokens as they are
  generated (streaming), or wait for the full response? This affects how TPOT
  latency is experienced and whether speculative decoding token acceptance rate
  matters for perceived quality.
- **Availability and failover**: is this a single-node deployment or must it
  be redundant? If the 8× H100 node is the full budget, there is no failover
  without rebalancing.

### Design

#### KV cache and memory budget

Before choosing frameworks or quantisation, establish the memory budget. This
constrains every other decision.

**Llama-3 70B weight memory:**

```
Weights at FP16:  70 × 10⁹ × 2 bytes = 140 GB
Weights at FP8:   70 × 10⁹ × 1 byte  =  70 GB
Weights at INT4:  70 × 10⁹ × 0.5 B   =  35 GB
```

Across 8× H100 80 GB with tensor parallelism degree 8:

```
Per-GPU weight memory (FP8):  70 GB / 8  = 8.75 GB
Per-GPU weight memory (INT4): 35 GB / 8  = 4.375 GB
Per-GPU remaining for KV (FP8): 80 - 8.75 ≈ 71 GB
Per-GPU remaining for KV (INT4): 80 - 4.375 ≈ 75 GB
```

**KV cache per sequence (FP16):**

```
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_element
         = 2 × 80 × 8 × 128 × seq_len × 2

At seq_len = 8000:  2 × 80 × 8 × 128 × 8000 × 2  =  2,621,440,000 bytes ≈ 2.44 GB
At seq_len = 200:   2 × 80 × 8 × 128 × 200  × 2  ≈  61 MB
```

With tensor parallelism TP=8, the KV heads are distributed across GPUs. With
n\_kv\_heads=8 and TP=8, each GPU holds exactly 1 KV head. Per-GPU KV
footprint per sequence:

```
At seq_len = 8000 (FP16): 2 × 80 × 1 × 128 × 8000 × 2 ≈ 328 MB
At seq_len = 8000 (FP8):  ≈ 164 MB
```

With 71 GB per GPU available for KV (after FP8 weights):

```
Max concurrent sequences at 8000 tokens (FP8 KV): 71 GB / 0.164 GB ≈ 433
```

This is a comfortable budget. Even at maximum input length, the cluster can
hold hundreds of concurrent long-context sequences in the KV cache, which is
sufficient for the target throughput with continuous batching.

**KV cache quantisation to FP8**: with the H100's FP8 tensor cores and
TensorRT-LLM's native FP8 KV path, the quality impact is near-lossless
(back-of-envelope from TensorRT-LLM benchmarks on Llama-family models). INT8
KV achieves the same 2× memory reduction with better hardware support on
non-H100 GPUs but is less relevant here. Using FP8 KV throughout is the
right call.

#### Parallelism strategy

With 8× H100 on a single node connected by NVLink (900 GB/s bisection bandwidth
total), pure tensor parallelism (TP=8) is the primary strategy. Each
all-reduce in a transformer layer traverses NVLink, which is fast relative to
the computation. Pipeline parallelism (PP) would add pipeline bubbles at every
forward pass and increase TTFT — the opposite of what the 200 ms TTFT target
demands. Sequence parallelism beyond what is fused into the TP communication
is unnecessary at these sequence lengths.

**Why TP=8 over PP=2×TP=4**: pipeline parallelism amortises best over large
batch sizes with long generation sequences (the fill is small relative to the
forward pass volume). For TTFT-sensitive workloads where every layer must
complete before the first token is returned, PP introduces a pipeline depth
penalty proportional to the number of pipeline stages. At these SLOs, TP=8
on NVLink is strictly better.

Expert parallelism (EP) is irrelevant — Llama-3 70B is a dense model.

#### Framework selection: TensorRT-LLM over vLLM

For maximising throughput on NVIDIA hardware against strict SLOs, TensorRT-LLM
is the right choice over vLLM. The reason is not PagedAttention — both
implement it — but the degree of NVIDIA-specific kernel optimisation.
TensorRT-LLM uses FP8 tensor core paths on H100 natively, applies CUTLASS-based
fused attention kernels, and exposes the in-flight batching C++ runtime which
operates at lower overhead than vLLM's Python scheduling layer. At the
throughput levels this hardware can sustain (hundreds of requests per second),
vLLM's Python overhead becomes a ceiling; TensorRT-LLM's C++ runtime does not.

vLLM is the better choice for: broad model support without an engine compilation
step; teams without NVIDIA-specialist knowledge; non-H100 targets. For this
scenario, TensorRT-LLM's throughput advantage on H100 FP8 is decisive.

Serve via Triton Inference Server with the TensorRT-LLM backend. NIM packages
this combination as a single container with validated performance profiles for
the major model families, which reduces time-to-deployment and guarantees that
the engine configuration matches the hardware.

#### Quantisation: FP8 (W8A8)

FP8 rather than INT4 weight-only for this hardware. The H100 has native FP8
tensor cores; running FP8 matmuls (not just FP8 weight storage) means the
tensor core throughput benefit applies to both the compute and the memory
bandwidth. INT4 weight-only (AWQ/GPTQ) would compress weights further (35 GB
vs 70 GB) but the matmuls dequantise to FP16 before execution, so there is no
tensor-core benefit beyond bandwidth reduction. With 71 GB per GPU available
for KV after FP8 weights, there is no memory pressure that would push the
decision toward INT4.

Accuracy consideration: FP8 on H100 is near-lossless on standard benchmarks.
If the application is high-stakes, validate on a task-specific eval set before
committing; the difference from FP16 is typically within 0.5% on standard
benchmarks (back-of-envelope from NVIDIA blog comparisons).

#### Batching strategy: continuous batching with PagedAttention

Static batching is ruled out — any fixed batch will idle GPU cycles whenever
short-output requests finish before the longest in the batch. Continuous
(in-flight) batching releases a slot as soon as a sequence completes and
immediately schedules a new request, keeping GPU utilisation near peak.
TensorRT-LLM's C++ runtime implements this natively.

PagedAttention (or TensorRT-LLM's paged KV equivalent) is necessary because
input lengths range from 200 to 8,000 tokens — a 40× range. Contiguous KV
allocation at maximum sequence length would waste 97.5% of allocated KV memory
for a 200-token request, collapsing effective batch size. With paged blocks,
KV memory scales with actual token count.

**Chunked prefill** (splitting long prefills across multiple iterations) is
worth enabling. A 8,000-token prefill at 8 sequences in a batch is
a very large compute step that delays the decode phase for other in-flight
requests. Chunked prefill limits the compute step per iteration, smoothing
TTFT for requests behind a long prefill in the queue. TensorRT-LLM supports
this via the `max_num_tokens` parameter.

#### TTFT estimate

H100 peak FP8 throughput: ~3,958 TFLOPS (NVIDIA H100 datasheet, Tensor Core,
sparsity-off). With TP=8, total cluster FP8 throughput: ≈ 31,664 TFLOPS.

FLOPs for prefill of 8,000-token sequence:

```
FLOPs ≈ 2 × n_params × seq_len = 2 × 70 × 10⁹ × 8,000 ≈ 1.12 × 10¹⁵
```

Compute-limited prefill time:

```
t_prefill = 1.12 × 10¹⁵ / 3.1664 × 10¹⁶ ≈ 35 ms  (back-of-envelope; does not
account for memory-fetch or framework overhead)
```

Adding realistic overhead (attention memory access, framework scheduling,
activation recomputation if enabled): estimate 80–120 ms for an 8,000-token
prefill on this cluster. That is within the 200 ms TTFT SLO with meaningful
headroom. For a 200-token input the prefill FLOPs are 40× smaller, giving a
compute-limited estimate of <5 ms; latency is dominated by framework overhead
at this length.

#### TPOT estimate

During decode, the forward pass generates one token. The compute per token:

```
FLOPs per token ≈ 2 × 70 × 10⁹ = 140 GFLOP
```

At batch size B with FP8 execution on TP=8:

```
t_compute = (B × 140 × 10⁹) / 3.1664 × 10¹⁶  seconds
```

Memory-bandwidth bound: each decode step must stream all model weights from
HBM. H100 SXM HBM bandwidth: 3.35 TB/s per GPU. With TP=8, each GPU holds
8.75 GB of weights (FP8). Time to stream:

```
t_memory = 8.75 × 10⁹ / 3.35 × 10¹² ≈ 2.6 ms per GPU per step
```

At batch size 1, decode is memory-bandwidth-bound at ~2.6 ms per token — well
within the 30 ms TPOT SLO. The compute bound does not hit until:

```
B_crossover = (3.1664 × 10¹⁶ × 2.6 × 10⁻³) / (8 × 140 × 10⁹) ≈ 73,000
```

In practice, output token latency is dominated by memory bandwidth at all
realistic batch sizes for this hardware configuration — meaning the 30 ms
TPOT target is easily met and the cluster throughput is constrained by how
many requests can be kept in flight simultaneously, not by per-token compute.
The effective throughput ceiling is therefore set by the KV cache budget and
the continuous batching scheduler, not the GPU compute.

#### Speculative decoding

With TPOT headroom well within 30 ms at realistic batch sizes, speculative
decoding is worth evaluating. Using EAGLE-3 (TensorRT-LLM's native speculative
decoder with auto-regressive feature-level drafting) with a small draft model
can yield 2–3× token generation speedup on predictable outputs, which would
allow either tighter TPOT SLOs or proportionally higher throughput at the same
latency. However: speculative decoding acceptance rates are highly
input-dependent. For an instruction-following workload with diverse outputs,
acceptance rates may be lower than the 2–3× benchmark figure. Measure on the
actual output distribution before committing the draft model infrastructure.

#### KV cache prefix caching

If requests share a common system prompt (common for instruction-following
deployments), enable prefix caching (RadixAttention-style in SGLang, or the
equivalent prefix-block reuse in TensorRT-LLM). Shared prefix KV blocks are
computed once and reused across requests, reducing effective TTFT for any
request that begins with the cached prefix. With a 500-token system prompt, the
KV for those 500 tokens is reused for every request sharing that prompt —
saving approximately 500/8000 ≈ 6% of the prefill compute for maximum-length
inputs, and proportionally more for short inputs.

For TensorRT-LLM and Triton integration detail, see
[`NVIDIA_GPU_19_TensorRT_LLM`](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_19_TensorRT_LLM).

### Trade-offs called out

- **TensorRT-LLM over vLLM**: TensorRT-LLM provides FP8 tensor-core paths and a
  lower-overhead C++ runtime that matters at high request rates. The cost is
  model support breadth and an engine compilation step; vLLM is simpler to
  operate but leaves throughput on the table at H100 scale.
- **FP8 over INT4**: INT4 halves weight memory again (35 GB total, 4.4 GB per
  GPU) but the compute runs in FP16 — no tensor-core gain beyond bandwidth.
  With FP8 weight memory already fitting comfortably (8.75 GB per GPU), the
  INT4 advantage is marginal and the quality trade-off is larger.
- **TP=8 over PP×TP mixed**: pipeline parallelism adds TTFT penalty proportional
  to pipeline depth. On a single NVLink-connected node, TP=8 uses the full
  interconnect bandwidth without pipeline bubbles.
- **Chunked prefill trade-off**: chunked prefill smooths TTFT for requests behind
  long prefills but reduces raw prefill throughput per token by adding iteration
  overhead. The right chunk size is workload-dependent; a good starting point is
  4,096 tokens per chunk.
- **Speculative decoding gain vs complexity**: EAGLE-3 can deliver 2–3× token
  speedup but requires a draft model, a verification step, and careful acceptance
  rate monitoring. If the output distribution is unpredictable, the overhead from
  rejected speculative tokens adds latency rather than removing it.
- **Prefix caching vs KV memory consumption**: cached prefixes consume KV blocks
  permanently until evicted. For deployments with many distinct system prompts,
  the cache hit rate drops and the memory consumption is pure waste. Best for
  deployments with one or a small number of common prompts.

### What "good" vs "better" looks like

A good answer recognises that 70B on 8× H100 requires tensor parallelism,
selects FP8 quantisation, chooses continuous batching, and names TensorRT-LLM
or vLLM. It correctly identifies that TPOT is memory-bandwidth-bound and TTFT
is compute-bound.

A better answer derives the KV cache budget explicitly (showing the 2 × 80 ×
8 × 128 formula), concludes that the per-GPU KV budget is comfortable even at
8,000-token inputs, and uses this to confirm that throughput is not KV-limited
at the target SLOs. It distinguishes TP from PP and explains why PP's pipeline
latency penalty makes it wrong for a tight TTFT SLO on a single NVLink node.
It computes the memory-bandwidth-limited decode time (~2.6 ms per step per GPU)
and shows that the 30 ms TPOT target has a factor-of-ten headroom, then
re-interprets this as meaning the system is throughput-limited by the
KV scheduler rather than by compute. It addresses speculative decoding with
appropriate caution about acceptance rates on diverse outputs, rather than
asserting "add EAGLE-3 for 2× speedup" without condition.

---

## Scenario 3: Domain-Specific Assistant Fine-Tune with Limited Compute and Small Labelled Dataset

### Brief

You need to build a domain-specific assistant for regulatory compliance writing.
The domain is narrow but technically dense: drafting and interpreting regulatory
filings, compliance reports, and policy documentation. You have 5,000
instruction/response pairs annotated by compliance experts. The responses are
long (typically 300–800 tokens), precise in terminology, and follow document
conventions that general-purpose LLMs do not reliably reproduce.

Compute budget: a single RTX 4000 Ada (20 GB GDDR6) on a local workstation
(Brendan's actual hardware), or alternatively one cloud GPU instance for 24
hours. The model will be served locally after training. There is no budget for
H100s or multi-GPU cluster runs.

The task is to choose between the following approaches, justify the choice, and
describe an evaluation plan:

1. Full SFT (supervised fine-tuning on all parameters)
2. LoRA fine-tuning (low-rank adaptation on a BF16 base model)
3. QLoRA fine-tuning (LoRA on a 4-bit quantised base model)
4. RAG-only (no fine-tuning; retrieval-augmented generation over a corpus)
5. Hybrid (QLoRA fine-tune plus RAG retrieval at inference time)

### Clarifying questions you'd ask

- **Nature of the knowledge gap**: is the primary problem that the base model
  lacks domain vocabulary and document format conventions (a style/format problem),
  or that it lacks factual knowledge of specific regulations (a knowledge problem)?
  Style problems are best addressed by fine-tuning; knowledge problems are better
  addressed by RAG. Most domain-specific problems are a mix of both.
- **Availability of a reference corpus**: is there an existing corpus of regulatory
  documents (published regulations, past filings, policy manuals) that could serve
  as a RAG knowledge base? If yes, RAG becomes a stronger option; if the knowledge
  lives entirely in the 5,000 labelled pairs, RAG has nothing to retrieve from.
- **Update cadence**: do regulations change frequently? If the compliance domain
  updates substantially each year, a pure fine-tune will require periodic retraining;
  a RAG component can absorb new regulation documents without retraining.
- **Inference environment**: will the model be served via a local API only for the
  compliance team, or is there a latency requirement? A QLoRA-trained 7B model
  served locally on the RTX 4000 Ada will have different latency characteristics
  than a larger model served via API.
- **Evaluation criteria**: are there existing human-graded compliance outputs that
  could serve as a gold standard, or will evaluation require expert annotation?
  Without a principled evaluation set, it is impossible to compare approaches
  objectively.
- **Acceptable base model size**: must the model be self-hosted (constraining
  size to what fits on 20 GB), or can the fine-tuned adapter be applied to a
  hosted base model for inference?

### Design

#### Approach comparison

**Full SFT** is infeasible on this hardware. A 7B model requires approximately
112 GB VRAM for parameters, gradients, and AdamW optimiser states with BF16
training:

```
Full SFT VRAM ≈ 16 bytes/parameter × 7 × 10⁹ = 112 GB
```

This exceeds the RTX 4000 Ada's 20 GB by a factor of ~5.6. Even with gradient
checkpointing (which halves activation memory), the optimiser states alone are
56 GB. Full SFT at 7B scale on this hardware is not viable without significant
scaling down to a 1B-class model, which is unlikely to have the reasoning
capacity for complex compliance writing.

**LoRA (BF16 base)** is also infeasible for a 7B base model on this hardware.
The base model weights in BF16 consume approximately 14 GB, leaving only 6 GB
for activations, gradients, and the LoRA adapter parameters. At a sequence
length of 512 tokens and batch size 1, activation memory for a 7B model
approaches 4–6 GB; this is at the edge of feasibility and makes training
unstable. LoRA on a 13B base is firmly out. LoRA is viable for smaller models
(1B–3B at BF16) on this hardware, but those models underperform for complex
generation tasks.

**QLoRA** loads the base model in 4-bit NF4 quantisation, reducing the 7B
base footprint to approximately 3.5–4 GB, and trains LoRA adapters in BF16
on top. Total VRAM with a rank-16 adapter across all attention matrices and
MLP layers, batch size 1, sequence length 512, with gradient checkpointing:
approximately 10–14 GB on the RTX 4000 Ada. This fits. The paged-AdamW
optimiser (from the original QLoRA paper) handles memory spikes from gradient
accumulation by paging optimiser states to CPU RAM.

For the 5,000-sample dataset, 3–5 training epochs at batch size 1 with gradient
accumulation of 8 steps takes approximately 4–8 hours on the RTX 4000 Ada with
Unsloth (which provides kernel-level optimisations for QLoRA, claiming 2–5×
speedup over baseline TRL). This is well within the 24-hour cloud budget or
feasible as an overnight local run.

**RAG-only** addresses the knowledge problem but not the style problem. If the
compliance domain has a specific document structure (section headers, citation
formats, regulatory cross-reference conventions) that a general-purpose 7B
model does not reproduce reliably, RAG alone will not fix this — it adds
factual context but generation style is controlled by the base model. RAG-only
is the right choice if the primary failure mode is factual incorrectness, not
poor formatting or style.

**Hybrid (QLoRA + RAG)** is the recommended approach, provided a reference
corpus exists. The fine-tune addresses the style problem — the model learns
the document conventions and terminology from the 5,000 labelled pairs. The
RAG component addresses the knowledge problem — it retrieves relevant
regulation text and past filing examples at inference time, preventing
hallucination of specific regulation details. The hybrid does not require the
fine-tune to memorise every regulation in the training set; it only needs to
learn the format and reasoning style.

#### Recommended approach: QLoRA with RAG augmentation at inference

Base model: Llama-3-8B-Instruct (8B, slightly larger than the 7B approximation
above, but fits at 4-bit NF4: ~4–4.5 GB weights, ~11–14 GB total during
training with gradient checkpointing at rank 16).

Training configuration:
- Framework: Unsloth wrapping TRL's `SFTTrainer` — Unsloth's attention and MLP
  kernel optimisations reduce VRAM and training time on Ada. Axolotl is a
  reasonable alternative with more configuration flexibility.
- LoRA rank: r=16, alpha=32 (the convention of alpha=2r gives a scaling factor
  of 2.0, which avoids the need to retune the learning rate when rank changes).
  Adapting all linear layers (Wq, Wk, Wv, Wo, and the two MLP projections)
  rather than only attention gives meaningfully better style learning for
  generation tasks; the additional trainable parameters are small relative to
  the 4-bit base.
- Epochs: 3, with early stopping on validation loss (hold out 500 samples).
- Sequence length: 1024 tokens (covers most instruction/response pairs in the
  dataset without padding waste; longer sequences increase activation memory
  substantially on Ada).
- Gradient accumulation: 8 steps at batch size 1, giving an effective batch of
  8 — a reasonable compromise between stability and throughput on 20 GB.

For the RAG component, a small Chroma or Qdrant local index over the regulatory
corpus (published regulations, past filings), embedded with `bge-small-en-v1.5`
(33 M parameters, fast on GPU). At inference time, retrieve top-3 relevant
passages and prepend them to the context. The fine-tuned model is already
calibrated to the document style; the RAG passages supply the specific facts.

Inference: the fine-tuned QLoRA adapter can be merged into the base weights
post-training (`W = W0 + BA`), eliminating any inference overhead from the
separate adapter matrices. The merged model served via llama.cpp (GGUF INT4
or INT8) on the RTX 4000 Ada provides comfortable single-user serving latency.
For multi-user serving, vLLM with the merged INT4 model is preferable.

#### Why QLoRA over the alternatives (summary)

- Full SFT: infeasible at 7B+ on 20 GB, and 1B models are too small for the task.
- LoRA (BF16): infeasible at 7B on 20 GB due to base model weight footprint.
- RAG-only: cannot solve the style/convention learning problem that domain
  labelled pairs are designed to address.
- Pure QLoRA (no RAG): viable and simpler, but knowledge hallucination on specific
  regulation references remains a risk. Adding a small RAG index costs little
  and meaningfully reduces this failure mode.

#### Evaluation plan

**Offline evaluation before deployment:**

1. Hold out 500 labelled pairs from the 5,000 as a test set. Never used for
   training or adapter rank/hyperparameter selection. Use 500 for validation
   (early stopping) and 4,000 for training.

2. Generate responses on the test set from: (a) the base model without
   fine-tuning, (b) the QLoRA-tuned model, (c) the QLoRA + RAG hybrid.
   Compare on the following metrics:

   - **ROUGE-L** and **BERTScore F1**: automatic proxies for lexical and semantic
     similarity to the gold responses. Useful for detecting gross regression
     relative to the base model, but insufficient as primary metrics for
     compliance writing — two compliant documents can use entirely different
     phrasings.
   - **Faithfulness (RAGAS)**: for the RAG condition, does the generated response
     contain only claims supported by retrieved context? This is the primary
     hallucination metric for the hybrid.
   - **Regulatory citation accuracy**: a domain-specific metric. Extract all
     regulatory references from generated responses (regex or NER) and verify
     they exist in the regulation corpus. Hallucinated citation identifiers are
     a critical failure mode in compliance writing.
   - **Format compliance**: automated checks for document structure conventions
     (required section headers present, correct numbering style). These are
     cheap to run and catch obvious format regressions.

3. **Expert review panel** (most important, hardest to scale): have 2–3
   compliance domain experts evaluate 50–100 generated responses on a 5-point
   rubric: regulatory accuracy, document format, completeness, and overall
   fitness for use. This is the ground truth evaluation; all automated metrics
   are proxies for it.

**Iterative improvement signals:**
- If ROUGE-L on the test set is not materially above the base model baseline,
  the fine-tune is not learning from the training data — investigate data quality
  or training configuration.
- If faithfulness degrades in the hybrid vs fine-tune-only condition, the RAG
  context is confusing the model — examine the retrieval quality and consider
  reranking or a tighter context budget.
- If expert reviewers flag hallucinated citations, the RAG index is missing
  coverage — expand the corpus.

For evaluation depth, see [`LLM_Hub_Evaluations`](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations).
For fine-tuning tooling detail on this hardware, see
[`FT_02_LoRA_and_PEFT_Variants`](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants).

### Trade-offs called out

- **QLoRA vs LoRA**: QLoRA's 4-bit base makes 7B training feasible on 20 GB;
  LoRA with a BF16 base does not fit. The cost is approximately 0.5–1 point on
  generation benchmarks compared to the BF16 base — acceptable given the
  alternative is infeasibility. On a 24-hour cloud A100 (40 GB), LoRA with a
  BF16 7B base would be viable; the choice changes with hardware.
- **Rank r=16 vs higher rank**: r=64 would give more adapter expressiveness and
  might produce better style adaptation on a complex generation task, but
  increases VRAM and is likely unnecessary for a 5,000-sample dataset — the
  fine-tune is data-limited before it is capacity-limited. A strong answer
  mentions this asymmetry.
- **Hybrid vs RAG-only**: RAG-only is simpler to deploy and maintain (no
  fine-tune artefact to version), and updates automatically as the regulation
  corpus is refreshed. The hybrid is better when the base model's output format
  is a known failure mode — which it will be for any narrow-domain document
  convention.
- **5,000 samples is small for SFT**: data quality therefore matters more than
  quantity. Removing low-quality or inconsistent examples from the training set
  may improve results more than training for additional epochs.
- **Unsloth vs Axolotl vs TRL directly**: Unsloth is the fastest option on
  Ada with its custom attention kernels, but it has a narrower model support
  matrix. Axolotl is more flexible and well-maintained; TRL is most
  transparent for anyone who needs to debug the training loop.
- **Merging the adapter post-training**: merging eliminates inference overhead
  and simplifies serving. The trade-off is that the merged checkpoint is a full
  model copy; keeping the adapter separate allows swapping adapters on a shared
  base. At this scale (one adapter, local serving), merging is the right default.

### What "good" vs "better" looks like

A good answer recognises that full SFT and BF16 LoRA are not feasible on 20 GB
for a 7B model, selects QLoRA as the viable fine-tuning method, and recommends
the hybrid over pure RAG because of the style-learning benefit. It mentions
ROUGE-L and RAGAS faithfulness as evaluation metrics.

A better answer provides the arithmetic ruling out full SFT (16 bytes/param ×
7B ≈ 112 GB) and BF16 LoRA (14 GB base leaves inadequate headroom), rather
than asserting infeasibility by assertion. It explains the alpha=2r convention
and why it avoids learning rate retuning. It separates the style problem
(addressable by fine-tuning) from the knowledge problem (addressable by RAG)
and argues that the hybrid addresses both, with citations to which failure mode
each component mitigates. It proposes regulatory citation accuracy as a
domain-specific evaluation metric — not just ROUGE-L and faithfulness — and
acknowledges that expert review is the ground truth evaluation that all
automated metrics approximate.

---

*For deeper treatment of any component above, see the linked portfolio repos.
The value of this file is synthesis and trade-off reasoning under constraints;
the repos carry the technical depth.*

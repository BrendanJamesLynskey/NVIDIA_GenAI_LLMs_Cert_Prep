# Distributed Training

Distributed training sits at the intersection of two NCP-GENL domains — GPU Acceleration and Optimisation (14%) and, partially, Model Optimisation (17%). The core exam requirement is understanding *why* each parallelism strategy exists, *what* it shards or partitions, *what* the communication cost is, and *how* the strategies compose. Practical implementation at scale is out of reach on Brendan's hardware, but the conceptual model is entirely testable.

Cross-reference: [LLM\_Hub\_NVIDIA\_GPUs](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs) for hardware topology; [NVIDIA\_GPU\_20\_NeMo\_NIM\_AI\_Enterprise](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise) for NeMo's parallelism implementation via Megatron Core.

---

## Why Distributed Training Is Necessary

Training FLOPs scale roughly as `6 × N × D` (where N is parameter count and D is the number of training tokens), a relationship sometimes called the Chinchilla scaling law framework. Single-GPU memory, however, grows only as the silicon roadmap allows. The consequence is straightforward: a 70B-parameter model in BF16 alone occupies around 140 GB of weights — far beyond any single consumer or prosumer GPU. Add optimiser states (Adam stores first and second moments, tripling the weight footprint), gradients, and activations, and the gap widens further.

Even setting memory aside, compute throughput per GPU has a ceiling. Training on a single A100 at, say, 300 TFLOP/s would take months for large models at Chinchilla-optimal token counts. Distributing across hundreds of GPUs reduces wall-clock time to days.

The three fundamental axes of parallelism — data, tensor, and pipeline — address different bottlenecks. Expert and sequence parallelism are specialisations on top of those three.

---

## Data Parallelism and ZeRO

Data parallelism (DP) is the simplest form: each GPU holds a full model copy and processes a different micro-batch of data. At the end of each forward-backward pass, gradients are synchronised across all ranks via an all-reduce collective. PyTorch's `DistributedDataParallel` (DDP) is the canonical implementation; gradients are overlapped with the backward pass via bucketed all-reduces.

The memory problem with vanilla DP is that each GPU stores the *entire* model state: weights, gradients, and optimiser states. For Adam, the optimiser state alone is roughly 12 bytes per parameter (fp32 parameters + fp32 first moment + fp32 second moment), so a 7B model requires around 84 GB of optimiser state per replica.

**ZeRO (Zero Redundancy Optimizer)**, from DeepSpeed, attacks this redundancy in three stages:

| Stage | What is sharded | Memory reduction (rough) |
|-------|----------------|--------------------------|
| ZeRO-1 | Optimiser states only | ~4× |
| ZeRO-2 | Optimiser states + gradients | ~8× |
| ZeRO-3 | Optimiser states + gradients + parameters | ~N× (proportional to world size) |

PyTorch's **Fully Sharded Data Parallel (FSDP)** is the upstream equivalent of ZeRO-3: parameters are sharded across ranks and all-gathered before each layer's forward pass, then immediately discarded after. Gradients are reduced-scattered rather than all-reduced, halving the communication volume relative to DDP. The trade-off is increased communication frequency — every layer triggers an all-gather — which can dominate at low batch sizes or high inter-GPU latency.

ZeRO-3/FSDP is the right choice when a model fits in aggregate GPU memory across a node but not on a single card, and when the interconnect is fast enough (NVLink within a node is ideal).

---

## Tensor Parallelism

Tensor parallelism (TP), formalised in the Megatron-LM papers, partitions individual layer *tensors* across GPUs. The key insight is that matrix multiplications can be split either column-wise or row-wise, with a single synchronisation point at the split boundary.

For a feed-forward block `Y = GELU(XA)B`:
- **Column parallel**: split A by columns across N GPUs; each GPU computes a slice of the GELU output independently.
- **Row parallel**: split B by rows; each GPU holds the matching slice, and outputs are summed with an all-reduce.

For multi-head attention, query/key/value projections are split column-parallel (each GPU owns a subset of heads), and the output projection is row-parallel.

The communication cost per layer is two collective operations (an all-reduce or equivalent reduce-scatter + all-gather). For large enough hidden dimensions and fast enough interconnect (NVLink bandwidth in the hundreds of GB/s range within a node), this overhead is acceptable. Across nodes over InfiniBand (typically tens of GB/s), TP is usually confined to within a single node.

Sequence parallelism (SP) is an extension of TP: the dropout and layer-norm operations that lie *between* the TP-parallelised layers have their sequence dimension sharded across the same TP ranks. This avoids replicating activations at those points and reduces peak activation memory. NeMo's Megatron Bridge exposes SP as a flag alongside TP.

---

## Pipeline Parallelism

Pipeline parallelism (PP) assigns consecutive groups of layers to different GPUs — GPU 0 runs layers 0–7, GPU 1 runs layers 8–15, and so on. Each GPU only stores the parameters for its assigned stage.

**Naive pipeline parallelism** runs one micro-batch at a time: GPU 0 finishes the forward pass and hands activations to GPU 1, which hands to GPU 2, and so on; on the backward pass the gradient flows back in reverse. At any moment only one GPU is active — the *pipeline bubble* is `(p-1)/p` of the total time, where p is the number of stages. For p = 4, 75% of time is wasted.

**GPipe** and the 1F1B (one-forward-one-backward) schedule reduce the bubble. In 1F1B, each GPU interleaves forward passes for new micro-batches with backward passes for old ones once the pipeline is primed. The bubble fraction drops to approximately `(p-1) / (m + p - 1)` where m is the number of micro-batches per batch. At large m the bubble becomes negligible.

**Interleaved 1F1B** (used in Megatron-LM) assigns multiple non-contiguous layer chunks to each GPU — GPU 0 owns layers {0,1} and {16,17}, for example. This further reduces the bubble at the cost of additional communication.

The activation memory cost of pipeline parallelism is significant: activations from earlier stages must be retained until the backward pass arrives, which can require storing many micro-batch activations simultaneously. Gradient checkpointing (recomputing activations during backward rather than storing them) is typically used alongside PP to manage this.

---

## Expert Parallelism and Mixture-of-Experts

In Mixture-of-Experts (MoE) architectures, only a subset of the FFN experts are activated per token. Expert parallelism distributes experts across GPUs — each GPU owns a subset of experts and only computes for tokens routed to its experts. The communication pattern involves an all-to-all collective: tokens are dispatched to the GPUs holding their assigned experts, processed, then gathered back.

Expert parallelism interacts with the router load-balancing problem: if routing collapses onto a subset of experts, some GPUs become bottlenecks while others idle. Auxiliary load-balancing losses or token-dropping strategies are used to maintain even utilisation.

For MoE architecture details and cert-relevant depth, see [`LLM_Hub_Modern_Architectures`](https://github.com/BrendanJamesLynskey/LLM_Hub_Modern_Architectures), specifically the `Arch_01_MoE` material.

---

## Context / Sequence Parallelism for Long-Context Training

When sequence lengths exceed tens of thousands of tokens, attention itself becomes a memory and compute bottleneck even within a single layer. Two approaches distribute the sequence dimension across GPUs:

- **DeepSpeed Ulysses**: all-to-all transposes the sequence and head dimensions before and after attention, so each GPU runs full attention over all tokens for a subset of heads.
- **Ring attention**: each GPU holds a contiguous chunk of the sequence; attention is computed in a ring topology where key-value chunks are passed around. Communication overlaps with compute via careful pipelining.

Both approaches allow effectively infinite sequence lengths at the cost of additional all-to-all communication per attention layer.

---

## 3D Parallelism

Large-scale training typically combines all three axes:

```
World size = TP × PP × DP
```

A typical configuration might use TP=4 within a node (exploiting NVLink bandwidth), PP=8 across nodes (lower communication volume — only activations between stages), and DP across the remaining replicas. Each axis addresses a different constraint:

- **TP** shards within-layer compute and reduces activation memory per layer, but requires fast interconnect.
- **PP** reduces per-device parameter count and optimizer memory, tolerates slower inter-node links, but introduces pipeline bubble.
- **DP** scales throughput linearly with the number of replicas, shards optimizer state under ZeRO.

Choosing the right decomposition requires profiling: too much TP increases communication overhead; too little PP means each device holds too many parameters.

---

## NCCL Collectives

NVIDIA's Collective Communications Library (NCCL) underpins the inter-GPU communication for all of the above:

| Operation | Description | Primary use |
|-----------|-------------|-------------|
| All-reduce | Sum (or other reduction) across all ranks; result on all ranks | DDP gradient sync |
| Reduce-scatter | Reduce across ranks, scatter result shards | ZeRO-2/3 gradient sharding |
| All-gather | Gather shards from all ranks onto all ranks | ZeRO-3 parameter reconstruction |
| All-to-all | Personalised exchange — each rank sends a different chunk to each other rank | Expert parallelism |

Both **bandwidth** and **latency** matter. All-reduce is bandwidth-bound for large tensors (ring-based implementations achieve near-peak bandwidth). For small tensors — e.g., layer norms between pipeline stages — latency dominates. NCCL selects algorithms based on message size automatically.

**NVLink** (within a node) provides much higher bandwidth and lower latency than PCIe; NVLink 4.0 in Hopper nodes is on the order of 900 GB/s bidirectional. **InfiniBand** is the standard inter-node fabric in GPU clusters, with HDR/NDR bandwidths in the hundreds of Gb/s — sufficient for pipeline stage boundaries and DP all-reduces, but not for the high-frequency all-gathers of aggressive TP.

For GPU interconnect topology in detail, see [`LLM_Hub_NVIDIA_GPUs`](https://github.com/BrendanJamesLynskey/LLM_Hub_NVIDIA_GPUs).

---

## Hardware Realities: Brendan's Setup

On the RTX 3080 (10 GB) and RTX 4000 Ada (20 GB), distributed training is not practical — these are single-GPU machines with no high-bandwidth interconnect between them, and PCIe-based multi-GPU training for LLMs is communication-bandwidth-limited to the point of being counterproductive.

The practical scope is:
- Single-GPU training (gradient accumulation to simulate large batch sizes).
- `accelerate` from Hugging Face provides a consistent API that would generalise to multi-GPU or multi-node if hardware access changes; writing training loops with `accelerate` is good practice even on a single card.
- ZeRO-1 via `deepspeed` or FSDP with `offload_to_cpu=True` can help fit larger models in single-GPU memory by offloading optimiser states to CPU RAM.

---

## Likely Exam Angles

- **ZeRO stages**: given a model size and GPU count, calculate which ZeRO stage is needed to fit it in memory. Know what each stage shards.
- **Communication pattern**: identify the correct NCCL collective for a given parallelism operation (all-reduce for DDP, reduce-scatter + all-gather for FSDP, all-to-all for expert parallelism).
- **Pipeline bubble fraction**: given stages p and micro-batches m, compute approximate bubble fraction under 1F1B.
- **TP within node, PP across node**: explain why tensor parallelism is typically constrained to a single node while pipeline parallelism spans nodes.
- **3D parallelism composition**: given a world size, select a TP × PP × DP decomposition and justify it.
- **Sequence parallelism vs context parallelism**: distinguish the two — SP extends TP to non-attention operations, CP distributes the sequence for attention itself at long context.

---

## Further Reading

- Shoeybi et al. (2019) — Megatron-LM: <https://arxiv.org/abs/1909.08053>
- Rajbhandari et al. (2020) — ZeRO: <https://arxiv.org/abs/1910.02054>
- Huang et al. (2019) — GPipe: <https://arxiv.org/abs/1811.06965>
- Narayanan et al. (2021) — Efficient Large-Scale Language Model Training (Megatron 3D): <https://arxiv.org/abs/2104.04473>
- PyTorch FSDP tutorial: <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>
- DeepSpeed ZeRO docs: <https://www.deepspeed.ai/training/>
- NeMo Framework parallelisms: <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html>
- Fang et al. (2024) — DeepEP (expert parallelism): <https://arxiv.org/abs/2412.18928>
- [NVIDIA\_GPU\_20\_NeMo\_NIM\_AI\_Enterprise](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise) — NeMo's parallelism implementation

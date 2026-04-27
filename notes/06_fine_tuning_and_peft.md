# Fine-Tuning and PEFT

Fine-tuning accounts for 13% of the NCP-GENL Professional exam and is a component of the 22% Experimentation domain on the NCA-GENL Associate exam. Questions are concrete: VRAM budgets, rank/alpha interaction, which DPO eliminates, what LoRA actually does to the weight matrices. The portfolio repos below carry the full technical treatment; this file provides the cert-calibrated synthesis with hardware constraints specific to the RTX 3080 (10 GB) and RTX 4000 Ada (20 GB) used in this portfolio.

---

## The Post-Training Stack

Pretraining produces a model capable of predicting text; the post-training pipeline turns it into one that follows instructions, stays on-topic, and behaves safely:

1. **Supervised Fine-Tuning (SFT)** — fine-tune on curated instruction/response pairs using standard cross-entropy loss. The model learns the format and style of helpful responses.
2. **Preference tuning** — use human (or AI) comparisons to push the model towards preferred outputs. The two dominant approaches are RLHF (via PPO, with an explicit reward model) and DPO (a closed-form alternative that eliminates the reward model and RL loop).
3. **Constitutional AI / RLAIF** — optionally replace human preference labellers with an AI model guided by a written constitution.

Each stage builds on the previous. SFT-only models follow instructions adequately; adding preference tuning substantially improves helpfulness and safety. Alignment detail is in [notes/04\_alignment\_and\_trustworthy\_ai.md](04_alignment_and_trustworthy_ai.md); RLHF/PPO depth is in [FT\_03\_RLHF\_and\_PPO](https://github.com/BrendanJamesLynskey/FT_03_RLHF_and_PPO); DPO depth is in [FT\_04\_DPO\_and\_Cousins](https://github.com/BrendanJamesLynskey/FT_04_DPO_and_Cousins).

---

## SFT Pipeline

**Instruction datasets.** SFT requires (prompt, completion) pairs in a consistent chat template — typically ChatML or Llama-3's `<|begin_of_text|>` format. Dataset quality dominates quantity: a few thousand high-quality examples outperform millions of low-quality ones. Common sources include Alpaca, OpenHermes, SlimOrca, and purpose-built domain datasets.

**Chat templates.** Modern models use structured templates to delineate system, user, and assistant turns. The tokeniser applies the template; training must use the same template the model was pretrained or instruct-tuned with, or the model will not learn the correct format.

**Loss masking.** During SFT, cross-entropy loss is computed only on the assistant turns, not on the prompt. Without masking, the model wastes capacity fitting the user-input distribution it should already handle.

**Sample packing.** Multiple short examples are concatenated into a single sequence of length equal to the model's maximum context. This eliminates padding waste and can double effective training throughput. Attention masks must prevent cross-example attention.

**VRAM for SFT.** A rough estimate for full-precision (bf16) SFT:

$$\text{VRAM} \approx (M_p \times 2) + (M_p \times 2) + (M_p \times 8) + \text{activations}$$

where $M_p$ is parameters in billions × 2 bytes. The three terms are: parameters, gradients, and AdamW optimiser states (two moments × fp32). This gives approximately 16 bytes per parameter for full SFT with AdamW:

| Model size | Full SFT (AdamW, bf16) |
| --- | --- |
| 1B | ~16 GB |
| 3B | ~48 GB |
| 7B | ~112 GB |

Full technical depth: [FT\_01\_SFT\_Pipeline](https://github.com/BrendanJamesLynskey/FT_01_SFT_Pipeline).

---

## Full-Parameter Fine-Tuning vs PEFT

**Full-parameter fine-tuning** updates every weight in the model. It achieves the best task-specific performance ceiling but requires VRAM for parameters + gradients + optimiser states, and produces a full model checkpoint for each task.

**PEFT (Parameter-Efficient Fine-Tuning)** freezes most of the model and trains only a small set of additional or modified parameters. The adapter is orders of magnitude smaller than the base model, can be stored and swapped cheaply, and dramatically reduces VRAM during training.

When to use full fine-tuning: small models (1B–3B) where VRAM is not the bottleneck; tasks where the full parameter expressiveness matters; when merging is not a priority. When to use PEFT: large models; multi-task scenarios with one base model and many adapters; resource-constrained hardware.

---

## LoRA, QLoRA, DoRA, and IA³

### LoRA (Low-Rank Adaptation)

LoRA (Hu et al., 2021) freezes the pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and injects a trainable low-rank decomposition alongside it:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$. At initialisation, $A$ receives random Gaussian values and $B$ is zeroed, so $\Delta W = 0$ and training starts from the pretrained output.

**Key hyperparameters.**
- **Rank $r$**: The default in the original paper's GPT-2 experiments was $r = 4$ for query and value projections; GPT-3 experiments used $r = 1$ or $r = 8$ depending on parameter budget. In practice $r \in \{8, 16, 32, 64\}$ is typical. Higher rank increases expressiveness and parameter count quadratically.
- **Alpha $\alpha$**: A scaling factor applied to $\Delta W$ as $\frac{\alpha}{r}$. The authors set $\alpha = 32$ for GPT-2 and found it sufficient to match learning rate tuning. The convention is to set $\alpha = 2r$ or simply to not tune it beyond the initial value.
- **Target modules**: The original paper found that adapting both $W_q$ and $W_v$ gives the best performance for a fixed parameter budget. Adapting all attention matrices ($W_q, W_k, W_v, W_o$) plus MLP layers increases capacity at higher cost; frameworks such as Axolotl default to all linear layers.
- **Inference**: At inference time, the adapter can be merged: $W = W_0 + BA$, so there is no inference latency overhead from LoRA.

LoRA reduces trainable parameters by roughly $\frac{2rd}{d^2} = \frac{2r}{d}$ — for a 4096-dimensional model with $r = 16$, this is less than 1% of the original weight count. The full paper claims up to 10,000× fewer trainable parameters vs full fine-tuning of GPT-3.

### QLoRA (Quantised LoRA)

QLoRA (Dettmers et al., 2023) applies LoRA training on top of a 4-bit quantised base model, reducing VRAM further:

- **NF4 (4-bit NormalFloat)**: A data type designed to be information-theoretically optimal for weight distributions that are approximately normally distributed. More precise than INT4 for typical pretrained weights.
- **Double quantisation**: The quantisation constants (scaling factors) are themselves quantised, saving a further ~0.5 bits per parameter on average.
- **Paged optimisers**: NVIDIA's unified memory is used to page optimiser states to CPU RAM when GPU memory is under pressure, preventing OOM errors during gradient accumulation spikes.

QLoRA enables fine-tuning 65B parameter models on a single 48 GB GPU. On consumer hardware, it makes 7B–13B fine-tuning accessible on 20 GB VRAM.

### DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA (Liu et al., 2024, ICML Oral) decomposes the pretrained weight $W_0$ into magnitude ($m$) and direction ($V$) components and applies LoRA only to the directional component:

$$W = m \cdot \frac{V + \Delta V}{\|V + \Delta V\|}$$

where $\Delta V = BA$ is the low-rank update. The magnitude vector $m$ is trained as a separate parameter. This separation mirrors how pretrained models change during full fine-tuning (direction changes dominate in early training; magnitude adjusts later) and consistently outperforms LoRA at equivalent rank on LLaMA, LLaVA, and VL-BART benchmarks, with no additional inference overhead after merging.

### IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

IA³ modifies activations rather than weight matrices by element-wise rescaling: it learns a small vector of scale factors applied to the keys, values, and feed-forward activations. Trainable parameter count is an order of magnitude smaller than LoRA ($\sim 0.01\%$ of base model parameters). Best suited to scenarios with very limited data or extremely tight VRAM budgets; typically underperforms LoRA on complex tasks.

Full technical depth: [FT\_02\_LoRA\_and\_PEFT\_Variants](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants).

---

## Adapter Merging and Multi-LoRA Serving

**Merging.** Because $W = W_0 + BA$ is exact, a LoRA adapter can be fused into the base model weights after training. The merged model is indistinguishable from a fully fine-tuned model at inference and requires no changes to the serving stack. Merging multiple adapters (linear or TIES merging) produces a single model that combines capabilities from multiple fine-tuning runs.

**Multi-LoRA serving.** When a single base model must serve many tasks simultaneously — each with its own LoRA adapter — loading and unloading adapters per request is impractical. S-LoRA and Punica are systems designed to batch requests across different LoRA adapters efficiently, keeping the base model weights resident on GPU and managing the adapter weights as a separate pool. This is the standard architecture for SaaS fine-tuning platforms. Coverage: [FT\_02\_LoRA\_and\_PEFT\_Variants](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants).

---

## RLHF and DPO — High Level

RLHF (Reinforcement Learning from Human Feedback) trains a reward model on human preference pairs and then optimises the language model against that reward using PPO, with a KL penalty to the SFT reference model to prevent reward hacking.

DPO (Direct Preference Optimisation) eliminates the reward model and PPO by reparameterising the RLHF objective in closed form — the language model's own log-probabilities serve as an implicit reward. DPO trains with a simple binary cross-entropy loss over preference pairs and is substantially more stable.

This file provides context and VRAM framing; see [notes/04\_alignment\_and\_trustworthy\_ai.md](04_alignment_and_trustworthy_ai.md) for the alignment synthesis, [FT\_03\_RLHF\_and\_PPO](https://github.com/BrendanJamesLynskey/FT_03_RLHF_and_PPO) for the PPO actor-critic detail, and [FT\_04\_DPO\_and\_Cousins](https://github.com/BrendanJamesLynskey/FT_04_DPO_and_Cousins) for DPO, IPO, KTO, ORPO, and GRPO.

---

## VRAM Budgeting on Available Hardware

The hardware in this portfolio is the RTX 3080 (10 GB) and RTX 4000 Ada (20 GB). The following table gives realistic guidance; numbers assume bf16 activations, no tensor parallelism, and standard AdamW/paged-AdamW.

| Configuration | 10 GB (RTX 3080) | 20 GB (RTX 4000 Ada) |
| --- | --- | --- |
| 1B full SFT | Tight (fits with small batch) | Comfortable |
| 3B full SFT | Not feasible | Tight with gradient checkpointing |
| 7B full SFT | Not feasible | Not feasible |
| 1B QLoRA | Comfortable | Comfortable |
| 3B QLoRA | Comfortable | Comfortable |
| 7B QLoRA | Tight (~9–10 GB with r=16) | Comfortable |
| 13B QLoRA | Not feasible (>12 GB) | Tight (~18–20 GB) |
| 7B LoRA (bf16 base) | Not feasible (base alone ~14 GB) | Not feasible |
| DPO on 7B QLoRA | Tight on 10 GB (two model copies) | Feasible |

**Key rules of thumb.**
- A 7B model in bf16 occupies approximately 14 GB for parameters alone, so it does not fit on the 3080 without quantisation.
- QLoRA on a 4-bit base model brings a 7B model's base footprint to approximately 4–5 GB, making training feasible on 10 GB with a small batch size.
- DPO requires the policy model and a frozen reference model simultaneously. On QLoRA with a 7B base, two 4-bit model copies are approximately 8–10 GB, which is tight on 10 GB.
- Gradient checkpointing trades recomputation for memory and is essential for training larger models on consumer GPUs; it roughly halves activation memory.
- Flash Attention 2 reduces attention memory from $O(n^2)$ to $O(n)$ in activation storage and is always worth enabling.

---

## Tooling Landscape

| Tool | Primary use | Notes |
| --- | --- | --- |
| **TRL** (Hugging Face) | SFT, DPO, PPO, GRPO | Standard reference implementation; `SFTTrainer`, `DPOTrainer` |
| **Axolotl** | SFT and PEFT with config files | Wraps TRL; supports QLoRA, DoRA, sample packing, chat templates; well-maintained |
| **Unsloth** | Fast QLoRA / LoRA | Kernel-level optimisations for attention and MLP; claims 2–5× speedup; works on consumer GPUs |
| **torchtune** | PyTorch-native SFT and PEFT | Meta's official fine-tuning library; minimal dependencies; good for understanding internals |
| **NeMo Aligner** | Full RLHF, DPO, SteerLM on NVIDIA stack | Requires NeMo ecosystem; targets multi-GPU / Slurm deployments; not suited to single-GPU |
| **LLaMA-Factory** | SFT, LoRA, QLoRA with web UI | Broad model and method support; good for rapid experimentation |

For single-GPU work on the RTX 3080 or RTX 4000 Ada, **Axolotl** or **Unsloth** are the practical defaults. TRL is the right choice when you need to understand or modify the training loop directly. NeMo Aligner is out of scope without a multi-GPU cluster.

---

## Likely Exam Angles

- **LoRA rank and alpha interaction.** The scaling applied to $\Delta W$ is $\frac{\alpha}{r}$; if rank doubles and alpha is held constant, the effective learning rate of the adapter halves. Distractors may claim alpha is a learning rate multiplier applied independently of rank.
- **QLoRA vs LoRA memory difference.** QLoRA quantises the *base model* to 4 bits; LoRA itself is still trained in full precision. The memory saving comes from the quantised base, not from quantising the adapter. Distractors may conflate adapter quantisation with base model quantisation.
- **Inference overhead.** After merging, a LoRA adapter adds zero inference overhead. Before merging (separate adapter), there is a small overhead for the matrix multiplications $BAx$. DoRA also has zero inference overhead after merging. Distractors may claim LoRA always adds inference latency.
- **DPO eliminates what.** DPO eliminates the explicit reward model and the PPO training loop. It does *not* eliminate the reference model — a frozen reference model is still used. Examiners regularly use "eliminates the reference model" as a distractor.
- **Full SFT vs QLoRA on 7B.** Full SFT of a 7B model requires approximately 112 GB VRAM with AdamW and is not feasible on the hardware in this portfolio. QLoRA brings it within reach of the RTX 4000 Ada at 20 GB.
- **S-LoRA / multi-LoRA use case.** S-LoRA is relevant when one base model serves many different LoRA fine-tuned tasks simultaneously. It does not improve single-adapter inference speed; distractors may claim it does.

---

## Further Reading

- SFT pipeline depth: [FT\_01\_SFT\_Pipeline](https://github.com/BrendanJamesLynskey/FT_01_SFT_Pipeline)
- LoRA/QLoRA/DoRA/IA³ depth: [FT\_02\_LoRA\_and\_PEFT\_Variants](https://github.com/BrendanJamesLynskey/FT_02_LoRA_and_PEFT_Variants)
- RLHF and PPO depth: [FT\_03\_RLHF\_and\_PPO](https://github.com/BrendanJamesLynskey/FT_03_RLHF_and_PPO)
- DPO and cousins depth: [FT\_04\_DPO\_and\_Cousins](https://github.com/BrendanJamesLynskey/FT_04_DPO_and_Cousins)
- Constitutional AI and RLAIF: [FT\_05\_Constitutional\_AI\_and\_RLAIF](https://github.com/BrendanJamesLynskey/FT_05_Constitutional_AI_and_RLAIF)
- Fine-Tuning hub: [LLM\_Hub\_Fine\_Tuning](https://github.com/BrendanJamesLynskey/LLM_Hub_Fine_Tuning)
- Hu et al. (2021), LoRA paper: <https://arxiv.org/abs/2106.09685>
- Dettmers et al. (2023), QLoRA paper: <https://arxiv.org/abs/2305.14314>
- Liu et al. (2024), DoRA paper: <https://arxiv.org/abs/2402.09353>
- Rafailov et al. (2023), DPO paper: <https://arxiv.org/abs/2305.18290>
- TRL documentation: <https://huggingface.co/docs/trl>
- Axolotl: <https://github.com/axolotl-ai-cloud/axolotl>
- Unsloth: <https://github.com/unslothai/unsloth>
- torchtune: <https://github.com/pytorch/torchtune>
- NeMo Aligner: <https://github.com/NVIDIA/NeMo-Aligner>

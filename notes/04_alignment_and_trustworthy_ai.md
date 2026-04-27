# Alignment and Trustworthy AI

This domain covers 10% of the NCA-GENL Associate exam (Trustworthy AI) and 5% of the NCP-GENL Professional exam (Safety, Ethics, and Compliance). The weighting is modest but the questions are precise: examiners test specific mechanisms — reward hacking, KL penalties, DPO's implicit reward model — rather than general awareness. The depth here is deliberately cert-calibrated; the portfolio repos listed throughout carry the full technical treatment.

---

## Why Alignment Is Necessary

Pretraining optimises a language model to predict the next token in a corpus sampled from the internet. The resulting model is capable but misaligned with human intent: it can follow a harmful instruction as readily as a benign one, because the pretraining objective is indifferent to usefulness or safety.

The post-training pipeline exists to correct this. Its canonical stages are:

1. **Supervised Fine-Tuning (SFT)** — fine-tune on curated instruction/response pairs so the model learns to answer rather than complete.
2. **Preference tuning** — use human (or AI-generated) comparisons between model outputs to push behaviour towards preferred responses. RLHF via PPO is the original approach; DPO and its cousins are the dominant alternatives.
3. **Constitutional AI / RLAIF** — optionally replace human preference labellers with an AI model, guided by a written constitution.

Each stage can be applied independently, but full post-training pipelines typically stack all three.

---

## RLHF: Reward Modelling and PPO

Reinforcement Learning from Human Feedback (RLHF) introduces a reward model (RM) trained on human preference data and then uses proximal policy optimisation (PPO) to optimise the language model against that reward.

**Reward modelling.** Human annotators compare pairs of model outputs and mark a preference. The Bradley-Terry model is the standard framework: it assigns a scalar reward $r$ to each output and fits the parameters so that the probability of preferring response $y_w$ over $y_l$ given prompt $x$ is:

$$P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

The reward model is typically the SFT model with its final language-modelling head replaced by a scalar head.

**PPO actor-critic loop.** The language model (actor) generates responses; the reward model scores them; a value network (critic) estimates expected future reward; PPO updates the actor's weights via a clipped surrogate objective.

**KL penalty.** A KL-divergence term between the policy being trained and the frozen SFT reference model is added to the reward:

$$r_{\text{total}} = r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

The $\beta$ coefficient prevents the policy from drifting so far from the SFT baseline that it exploits the reward model rather than actually improving. Without the KL penalty, reward hacking — finding degenerate outputs that score highly under $r_\phi$ but are nonsensical — is common.

**Reward hacking failure modes.** The reward model is itself an imperfect proxy for human preferences. The policy can learn to generate outputs that maximise $r_\phi$ through patterns the reward model overvalues (verbose responses, sycophantic openers, repeated phrases) rather than through genuine quality. This is an instance of Goodhart's Law.

Full coverage: [FT\_03\_RLHF\_and\_PPO](https://github.com/BrendanJamesLynskey/FT_03_RLHF_and_PPO) and [LLM\_Hub\_Safety\_Alignment](https://github.com/BrendanJamesLynskey/LLM_Hub_Safety_Alignment).

---

## DPO and Preference-Based Alternatives

Direct Preference Optimisation (DPO; Rafailov et al., 2023) eliminates the separate reward model and the RL training loop by re-parameterising the RLHF objective directly in terms of the language model.

The key insight is that the optimal policy under the KL-constrained RLHF objective has a closed form, which allows the reward to be expressed as a function of the policy's own log-probabilities. The DPO loss is then a binary classification over preference pairs using only a simple cross-entropy objective — no PPO, no reward model, no sampling during training.

In practice DPO is substantially more stable to train than PPO and is the default preference-tuning method in most open-source pipelines. Its empirical performance matches or exceeds PPO on summarisation and dialogue tasks.

**Cousins.**

| Method | Key distinction |
| --- | --- |
| **IPO** (Identity Preference Optimisation) | Replaces DPO's log-ratio with a squared error to address overfitting on deterministic preference datasets |
| **KTO** (Kahneman-Tversky Optimisation) | Works on unpaired good/bad examples rather than preference pairs; more data-efficient when pairs are unavailable |
| **ORPO** (Odds Ratio Preference Optimisation) | Combines SFT and preference tuning into a single loss; removes the need for a separate SFT stage |
| **GRPO** (Group Relative Policy Optimisation) | Used in DeepSeek-R1; computes advantages relative to a group of sampled responses rather than a critic; popular for reasoning tasks |

Full coverage: [FT\_04\_DPO\_and\_Cousins](https://github.com/BrendanJamesLynskey/FT_04_DPO_and_Cousins).

---

## Constitutional AI and RLAIF

Constitutional AI (CAI; Bai et al., Anthropic 2022) reduces dependence on human preference labellers by replacing them with an AI model guided by a written set of principles — the "constitution."

**Supervised stage (CAI-SL).** The model generates an initial response to a potentially harmful prompt, then applies a critique-revision cycle: it critiques its own response against a selected constitutional principle and rewrites it. The revised responses become supervised fine-tuning data. No human labelling is required beyond writing the constitution itself.

**RLAIF stage.** Rather than asking human raters to compare response pairs, CAI uses the same AI model (or a larger "AI labeller") to assign preferences. The preference dataset is then used to train a reward model and run PPO, exactly as in RLHF — the only difference is that the feedback signal comes from an AI rather than humans.

The result is a harmless but non-evasive assistant: it engages with problematic queries by explaining its objections rather than simply refusing. Chain-of-thought reasoning is incorporated to make the model's alignment reasoning interpretable.

RLAIF is now widely used because scaling human preference labelling is expensive and inconsistent. The quality of the AI labeller is the dominant factor in outcome quality.

Full coverage: [FT\_05\_Constitutional\_AI\_and\_RLAIF](https://github.com/BrendanJamesLynskey/FT_05_Constitutional_AI_and_RLAIF).

---

## Guardrails: Input Filters, Output Filters, and Hardening

Post-training alone is insufficient in production. A separate guardrails layer intercepts requests and responses before they reach end users.

**Input filters** classify the user prompt before it reaches the model. Approaches range from keyword lists and regex (cheap, low recall) to dedicated classifier models. [Llama Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) (Meta) and [ShieldGemma](https://ai.google.dev/gemma/docs/shieldgemma) (Google) are LLM-based input/output classifiers trained on safety taxonomies; they can categorise violation type as well as flagging presence.

**Output filters** apply similar classification to model responses, often with higher latency tolerance because blocking a response is safer than blocking a request post-hoc.

**System-prompt hardening** embeds policy constraints and persona-locking instructions in the system prompt. It provides soft enforcement only — a sufficiently adversarial prompt can override system-prompt instructions — and should not be treated as the only guardrail layer.

**NeMo Guardrails** (NVIDIA) is a programmable framework that sits in front of (and behind) the model, defining topical rails, safety rails, and fact-checking rails as Colang dialogue flows. It is part of the NeMo platform and integrates with NVIDIA's AI Enterprise offering. Coverage: [NVIDIA\_GPU\_20\_NeMo\_NIM\_AI\_Enterprise](https://github.com/BrendanJamesLynskey/NVIDIA_GPU_20_NeMo_NIM_AI_Enterprise).

Full coverage of defences: [Safety\_02\_Defences\_and\_Compliance](https://github.com/BrendanJamesLynskey/Safety_02_Defences_and_Compliance).

---

## Jailbreak Taxonomy

Understanding how guardrails fail is necessary for designing robust defences. The main attack classes are:

- **Persona / roleplay attacks** — asking the model to pretend it is a different AI without restrictions ("DAN", "STAN", etc.). The model's safety training is anchored to its identity; persona shifts can partially bypass it.
- **Encoding attacks** — base64, leetspeak, ROT-13, or other obfuscated representations of a harmful query. Many filters operate on decoded text but some input classifiers miss encoded variants.
- **Automated red-teaming attacks** — GCG (Greedy Coordinate Gradient, Zou et al.) appends adversarially optimised token suffixes that cause the model to follow harmful instructions while appearing benign to human review. PAIR (Prompt Automatic Iterative Refinement) uses a second LLM to iteratively refine jailbreak prompts. AutoDAN generates syntactically fluent jailbreaks automatically. These attacks generalise across models and are the reason automated red-teaming is now standard practice before deployment.

Full coverage: [Safety\_01\_Jailbreaks](https://github.com/BrendanJamesLynskey/Safety_01_Jailbreaks).

---

## Compliance: NIST AI RMF and the EU AI Act

### NIST AI Risk Management Framework (AI RMF 1.0, 2023)

The NIST AI RMF organises risk management around four functions:

| Function | What it covers |
| --- | --- |
| **GOVERN** | Organisational culture, policies, accountability structures, and risk tolerance for AI |
| **MAP** | Identifying and categorising AI risks in context — who is affected, what can go wrong |
| **MEASURE** | Quantifying, analysing, and tracking identified risks through metrics and evaluation |
| **MANAGE** | Responding to, mitigating, and monitoring risks on an ongoing basis |

The RMF is voluntary for US organisations but is increasingly referenced in procurement contracts and sector-specific regulations. It defines "trustworthy AI" across seven properties: valid and reliable, safe, secure and resilient, explainable and interpretable, privacy-enhanced, fair with managed bias, and accountable and transparent.

### EU AI Act (in force 2024)

The EU AI Act classifies AI systems into four risk tiers:

| Tier | Examples | Requirement |
| --- | --- | --- |
| **Unacceptable risk** | Social scoring, real-time biometric surveillance in public | Prohibited |
| **High risk** | Medical devices, CV screening, critical infrastructure, law enforcement | Conformity assessment, human oversight, documentation |
| **Limited risk** | Chatbots, deepfakes | Transparency obligations (disclose AI nature) |
| **Minimal risk** | Spam filters, AI in games | No mandatory requirements |

**GPAI (General-Purpose AI) models** — models with a wide range of uses not tied to a specific application — are subject to specific obligations under the Act. GPAI models above a defined training compute threshold are designated "systemic risk" models and face additional evaluation and incident-reporting requirements. LLMs released as APIs or open weights fall under the GPAI classification.

---

## Bias and Fairness

Bias in LLMs can arise at multiple stages: in the pretraining corpus (over-representation of certain languages, demographics, viewpoints), in the instruction datasets used for SFT, and in the reward model used for preference tuning (if human annotators share systematic preferences).

What is measurable: demographic parity on classification outputs, representation rates across groups in generated text, performance differentials on benchmarks decomposed by group attribute.

What is harder to measure: subtle value misalignment, implicit framing effects, inconsistency between in-context claims and actual behaviour. The HELM, BIG-Bench, and WinoBias benchmarks address specific slices; no single benchmark covers bias comprehensively.

Fairness definitions (equal opportunity, demographic parity, individual fairness) are mathematically incompatible in most settings — satisfying one generally requires violating another. This is not a solvable problem but a design trade-off that must be made explicitly.

---

## Likely Exam Angles

- **RLHF vs DPO distinction.** A question will likely ask which component DPO eliminates: the separate reward model and the PPO training loop. Distractors: "it eliminates the KL penalty" (false — the KL constraint is implicit in DPO's derivation), or "it requires no reference model" (false — a frozen reference model is still used to compute log-probability ratios).
- **KL penalty purpose.** The KL penalty between the trained policy and the SFT reference model prevents reward hacking by anchoring the policy. Distractors may present it as preventing catastrophic forgetting (partially true but not the primary purpose) or as stabilising PPO's learning rate.
- **Constitutional AI vs RLAIF.** CAI is the full Anthropic framework (critique-revision SFT + RLAIF). RLAIF is the specific technique of using an AI model as the preference labeller. They are not synonyms. Examiners may test whether you know that RLAIF still involves a reward model and PPO.
- **EU AI Act GPAI.** Know that LLMs accessed via API are GPAI under the Act, and that a compute threshold (currently 10^25 FLOPs training) separates standard GPAI from systemic-risk GPAI.
- **NIST AI RMF functions.** GOVERN and MAP are about organisational readiness and context-setting; MEASURE and MANAGE are about quantification and response. A common distractor conflates MAP (identifying risks) with MEASURE (quantifying them).
- **NeMo Guardrails vs Llama Guard.** NeMo Guardrails is a programmable dialogue-flow framework; Llama Guard is a fine-tuned classifier model. They operate at different levels and are complementary, not alternatives.

---

## Further Reading

- RLHF / PPO depth: [FT\_03\_RLHF\_and\_PPO](https://github.com/BrendanJamesLynskey/FT_03_RLHF_and_PPO)
- DPO and cousins depth: [FT\_04\_DPO\_and\_Cousins](https://github.com/BrendanJamesLynskey/FT_04_DPO_and_Cousins)
- Constitutional AI depth: [FT\_05\_Constitutional\_AI\_and\_RLAIF](https://github.com/BrendanJamesLynskey/FT_05_Constitutional_AI_and_RLAIF)
- Jailbreak taxonomy: [Safety\_01\_Jailbreaks](https://github.com/BrendanJamesLynskey/Safety_01_Jailbreaks)
- Defences and compliance: [Safety\_02\_Defences\_and\_Compliance](https://github.com/BrendanJamesLynskey/Safety_02_Defences_and_Compliance)
- Safety/alignment hub: [LLM\_Hub\_Safety\_Alignment](https://github.com/BrendanJamesLynskey/LLM_Hub_Safety_Alignment)
- Rafailov et al. (2023), DPO paper: <https://arxiv.org/abs/2305.18290>
- Bai et al. (2022), Constitutional AI: <https://arxiv.org/abs/2212.08073>
- NIST AI RMF 1.0: <https://doi.org/10.6028/NIST.AI.100-1>
- EU AI Act text (EUR-Lex): <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689>
- Llama Guard: <https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/>
- NeMo Guardrails docs: <https://docs.nvidia.com/nemo/guardrails/>

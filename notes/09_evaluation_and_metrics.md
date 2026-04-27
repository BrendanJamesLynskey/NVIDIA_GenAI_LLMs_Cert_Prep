# Evaluation and Metrics

Evaluation maps to two cert domains: NCP-GENL Evaluation (7%) and NCA-GENL Data Analysis and Visualisation (14%). The NCA domain weight makes this more consequential for the associate exam than the domain name might suggest. Evaluation is also where teams most commonly under-invest — it is therefore high-signal territory for a practitioner exam.

An extended treatment of the evaluation landscape lives in Brendan's portfolio at [LLM\_Eval\_01\_Landscape](https://github.com/BrendanJamesLynskey/LLM_Eval_01_Landscape) through [LLM\_Eval\_05\_Red\_Teaming](https://github.com/BrendanJamesLynskey/LLM_Eval_05_Red_Teaming) and the hub [LLM\_Hub\_Evaluations](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations). This note synthesises that material to the cert syllabus.

---

## The Evaluation Pyramid

A useful mental model is a hierarchy from cheap-and-automated at the base to expensive-and-authoritative at the top:

```
          ┌─────────────────────────────┐
          │   Production telemetry      │  (slowest signal, highest validity)
          ├─────────────────────────────┤
          │   Human evaluation          │
          ├─────────────────────────────┤
          │   LLM-as-judge              │
          ├─────────────────────────────┤
          │   Static benchmarks         │
          └─────────────────────────────┘
          │   Perplexity / unit evals   │  (fastest signal, lowest external validity)
```

Each layer has a different cost, latency, noise level, and validity. A mature evaluation programme runs all five. The cert expects you to know what each layer measures, where each fails, and how to use them in combination.

This framing is developed in detail at [LLM\_Eval\_01\_Landscape](https://github.com/BrendanJamesLynskey/LLM_Eval_01_Landscape).

---

## Classical Metrics

### Perplexity

Perplexity is the exponentiated average negative log-likelihood of the model over a held-out corpus:

```
PPL = exp(- (1/N) Σ log P(token_i | context))
```

Lower perplexity means the model assigns higher probability to the ground-truth tokens. It is fast, deterministic, and hardware-consistent. It is a good signal during pretraining to detect instability or degradation.

**Where it fails**: perplexity measures fluency and calibration on a reference corpus, not task performance. A model with excellent perplexity on Wikipedia may be poor at instruction following, coding, or reasoning. It is also not comparable across tokenisers — a model with a larger vocabulary will tend to have lower perplexity on the same text for structural reasons.

### BLEU

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between a generated text and one or more reference texts, with a brevity penalty for overly short outputs. Originally designed for machine translation.

**Where it fails**: BLEU does not measure semantic correctness — a paraphrase that uses different words scores poorly. It correlates poorly with human judgement for open-ended generation. It is now largely obsolete for LLM evaluation except in specific MT benchmarks.

### ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is recall-oriented n-gram overlap, designed for summarisation. ROUGE-1 measures unigram recall; ROUGE-2 bigrams; ROUGE-L longest common subsequence.

**Where it fails**: same n-gram limitations as BLEU. Good summaries that rephrase the source score poorly; repetitive summaries that copy source sentences score well.

### BERTScore

BERTScore (Zhang et al., 2019) replaces n-gram matching with contextual embedding similarity using a pretrained BERT-family encoder. For each token in the candidate, it finds the most similar token in the reference by cosine similarity of contextual embeddings, then aggregates to precision/recall/F1.

**Where it fails**: correlation with human judgement is better than BLEU/ROUGE but still imperfect for long-form generation. The score is not interpretable in absolute terms — it requires calibration against human ratings for a given task. Sensitive to the choice of underlying BERT model.

---

## Static Benchmarks

Static benchmarks provide reproducible, comparable snapshots of capability across models. Key benchmarks the cert is likely to reference:

| Benchmark | What it tests | Format | Known issues |
|-----------|---------------|--------|--------------|
| **MMLU** (Hendrycks et al., 2021) | 57-subject multiple-choice; broad academic knowledge | MCQ | Contamination; question quality varies by subject |
| **MMLU-Pro** | Harder version of MMLU; 10-choice questions | MCQ | Partially mitigates MMLU ceiling effects |
| **HumanEval** (Chen et al., 2021) | Python function generation from docstrings; 164 problems | Code gen | Narrow; pass@k metric requires sampling |
| **GSM8K** (Cobbe et al., 2021) | Grade-school maths word problems | Free-form | Contamination well-documented post-2023 |
| **BIG-Bench / BIG-Bench Hard** | 200+ diverse tasks | Various | Heterogeneous difficulty; aggregate score misleading |
| **IFEval** (Zhou et al., 2023) | Instruction-following evaluation; verifiable constraints | Programmatic check | Narrow to instruction format; does not test content quality |
| **SWE-bench** | Real GitHub issues; model must produce patches | Code gen | Expensive to run; strong signal for software engineering |

**Benchmark contamination** is a genuine problem. Models trained on internet-scale data may have seen benchmark questions in their training corpus, inflating scores. Contamination detection is an active research area. When evaluating a fine-tuned model on benchmarks, contamination of the base model's training data must be considered separately from fine-tuning data contamination. Dynamic or held-out benchmarks (e.g., LiveBench) mitigate this at the cost of comparability over time.

---

## LLM-as-Judge

LLM-as-judge evaluation uses a capable language model (commonly GPT-4 or a dedicated evaluator model) to score outputs, either against a rubric (Likert scale) or pairwise against a reference or competing output.

**Pairwise comparison**: the judge receives two responses and selects the better one, or declares a tie. This avoids the calibration problem of absolute scores but is expensive (O(n²) for n models) and cannot recover absolute quality estimates without aggregation.

**Likert scoring**: the judge assigns a score on a scale (e.g., 1–5) against explicit criteria. Cheaper than pairwise at scale but requires careful rubric design.

**Known biases**:
- **Positional bias**: the judge tends to favour the response in the first position, especially for similar-quality outputs.
- **Verbosity bias**: longer responses are often rated higher regardless of quality.
- **Self-preference**: a GPT-4 judge will tend to rate GPT-4 outputs higher than outputs from other model families.
- **Format sensitivity**: markdown formatting, bullet points, and confident-sounding phrasing inflate scores.

**Calibration**: LLM judge scores should be calibrated against human ratings on a held-out set before being used as a primary metric. Positional bias can be partially mitigated by evaluating each pair in both orderings and averaging.

For detailed coverage of LLM-as-judge methodology and bias mitigation, see [LLM\_Eval\_02\_LLM\_as\_Judge](https://github.com/BrendanJamesLynskey/LLM_Eval_02_LLM_as_Judge).

---

## Evaluation Frameworks

| Framework | Primary use case | Hosted / local | Notes |
|-----------|-----------------|----------------|-------|
| **Inspect AI** (UK AISI) | Research-grade evals; custom tasks; agent evals | Local | Strong task abstraction; good for rigorous research evals |
| **Braintrust** | Product evals; prompt management; tracing | SaaS | Good developer UX; built-in dataset versioning |
| **Arize Phoenix** | LLMOps observability; tracing; drift detection | Local + SaaS | Strong on production monitoring and RAG evals |
| **Langfuse** | Open-source LLMOps; tracing; evals | Self-hosted + SaaS | Good Langchain/LlamaIndex integration |
| **OpenAI Evals** | Benchmark-style evals; model comparisons | Local | Originally internal OpenAI; now open-source |
| **deepeval** | Unit-test-style evals; CI integration; RAG metrics | Local | Lowest friction for adding eval to a CI pipeline |

**When to pick which**: deepeval is best for instrumenting evals into CI/CD early. Braintrust or Langfuse suit teams that want integrated tracing and eval in a product context. Inspect AI is appropriate for adversarial or research-grade evaluation requiring precise reproducibility. Arize Phoenix is the strongest option for production monitoring with a RAG component.

For a structured comparison, see [LLM\_Eval\_03\_Eval\_Frameworks](https://github.com/BrendanJamesLynskey/LLM_Eval_03_Eval_Frameworks).

---

## Production Evaluation

Production evaluation is fundamentally different from offline benchmark evaluation: the input distribution is real and unknown, outputs go to real users, and feedback is delayed and sparse.

Key practices:

- **Online sampling**: log a random sample of production inputs and outputs for periodic human review or LLM-judge scoring. Sampling rate is a cost-quality trade-off.
- **Golden-set versioning**: maintain a curated set of inputs with known-good outputs. Run this set on every model deployment as a regression check. Version the golden set alongside the model; as the application evolves, so should the golden set.
- **Drift detection**: track distribution statistics over time — input embedding distributions, output length distributions, toxicity scores, refusal rates. Sudden shifts indicate distribution shift in the user population or model degradation.
- **Regression CI**: before any model update (new base model, new fine-tune, new system prompt), gate the deployment on a regression suite that must pass. Treat eval as a first-class software testing discipline.

For production monitoring patterns, see [LLM\_Eval\_04\_Production\_Evals](https://github.com/BrendanJamesLynskey/LLM_Eval_04_Production_Evals).

---

## RAG Evaluation

RAG systems have failure modes distinct from standalone LLM generation: retrieval can fail (wrong documents retrieved), and the generator can fail to use retrieved context correctly (hallucination on top of correct context).

**RAGAS** (Es et al., 2023) is the most widely adopted RAG eval framework. Its primary metrics:

| Metric | What it measures | How computed |
|--------|-----------------|--------------|
| **Faithfulness** | Does the answer contain only claims that can be inferred from the retrieved context? | LLM-judge: each claim in the answer is checked against the context |
| **Answer relevancy** | Does the answer address the question asked? | Embedding similarity between generated reverse-questions and original question |
| **Context precision** | Are the top-ranked retrieved chunks relevant to answering the question? | LLM-judge: relevance of each chunk |
| **Context recall** | Were the chunks needed to answer the question actually retrieved? | LLM-judge against ground-truth answer |

Faithfulness is the most important metric for production RAG — a high-faithfulness answer that stays grounded in context is preferable to a confident-sounding but hallucinated answer.

For RAG evaluation in depth, see [RAG\_07\_Production\_RAG](https://github.com/BrendanJamesLynskey/RAG_07_Production_RAG) and [LLM\_Hub\_RAG\_Retrieval](https://github.com/BrendanJamesLynskey/LLM_Hub_RAG_Retrieval).

---

## Red-Teaming and Safety Evaluation

Safety evaluation assesses model behaviour under adversarial or out-of-distribution inputs. Benchmarks:

- **HarmBench** (Mazeika et al., 2024): standardised evaluation framework for LLM safety; covers chemical, cyber, and harmful-content attack categories; includes both attack methods and defences.
- **AdvBench** (Zou et al., 2023): 500 harmful behaviours and strings; used to measure attack success rate (ASR) for jailbreak methods.
- **JailbreakBench** (Chao et al., 2024): open-source leaderboard for jailbreak attacks and defences; reproducible attack scripts; tracks ASR across multiple models.
- **MT-Bench**: multi-turn instruction-following benchmark; not safety-specific but useful for detecting degradation under multi-turn attack scenarios.

Red-teaming is distinct from safety benchmarking: human red-teamers craft novel attacks outside any fixed benchmark. Automated red-teaming (using an attacker LLM to generate adversarial prompts) provides higher coverage at lower cost.

For safety evaluation methodology, see [LLM\_Eval\_05\_Red\_Teaming](https://github.com/BrendanJamesLynskey/LLM_Eval_05_Red_Teaming).

---

## Cost-Aware Evaluation

Evaluation is itself an expensive workload, and this is frequently overlooked. A full GPT-4-judge evaluation over a production dataset of 100k samples costs real money and real time. Practical cost management:

- Use perplexity or deterministic metrics for rapid iteration (cheap, fast).
- Use static benchmarks for model-selection checkpoints (moderate cost, high comparability).
- Use LLM-as-judge for the final selection and for qualitative analysis of failure modes (expensive, high validity).
- Reserve human evaluation for gold-set construction and calibration (most expensive, used sparingly).
- Cache judge outputs — deterministic inputs produce deterministic judge outputs; do not re-evaluate unchanged items.
- Profile evaluation infrastructure the same way you profile training — slow evaluation pipelines block iteration cycles.

---

## Likely Exam Angles

- **Metric selection**: given a task (summarisation, code generation, instruction following, RAG), identify the appropriate metric(s) and explain the choice.
- **Eval pyramid**: describe the trade-offs between each layer; explain why perplexity alone is insufficient for production decisions.
- **LLM-as-judge biases**: name at least three biases and explain mitigation strategies.
- **RAGAS metrics**: define faithfulness and context precision; explain why faithfulness is the primary anti-hallucination metric.
- **Benchmark contamination**: explain the problem and why it is more acute for fine-tuned models than base models.
- **Production eval practices**: describe the role of golden sets and regression CI in a deployment pipeline.

---

## Further Reading

- [LLM\_Hub\_Evaluations](https://github.com/BrendanJamesLynskey/LLM_Hub_Evaluations) — evaluation hub
- [LLM\_Eval\_01\_Landscape](https://github.com/BrendanJamesLynskey/LLM_Eval_01_Landscape) — eval pyramid and benchmark landscape
- [LLM\_Eval\_02\_LLM\_as\_Judge](https://github.com/BrendanJamesLynskey/LLM_Eval_02_LLM_as_Judge) — LLM-as-judge methodology
- [LLM\_Eval\_03\_Eval\_Frameworks](https://github.com/BrendanJamesLynskey/LLM_Eval_03_Eval_Frameworks) — framework comparison
- [LLM\_Eval\_04\_Production\_Evals](https://github.com/BrendanJamesLynskey/LLM_Eval_04_Production_Evals) — production monitoring
- [LLM\_Eval\_05\_Red\_Teaming](https://github.com/BrendanJamesLynskey/LLM_Eval_05_Red_Teaming) — red-teaming and safety evals
- Es et al. (2023) — RAGAS: <https://arxiv.org/abs/2309.15217>
- Mazeika et al. (2024) — HarmBench: <https://arxiv.org/abs/2402.04249>
- Chao et al. (2024) — JailbreakBench: <https://arxiv.org/abs/2404.01318>
- Zhang et al. (2019) — BERTScore: <https://arxiv.org/abs/1904.09675>
- Hendrycks et al. (2021) — MMLU: <https://arxiv.org/abs/2009.03300>
- Inspect AI framework: <https://inspect.ai-safety-institute.org.uk/>
- RAGAS documentation: <https://docs.ragas.io/>

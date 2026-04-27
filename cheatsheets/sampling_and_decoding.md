# Sampling and Decoding

## Decoding strategies

| Strategy | How it works | Use | Avoid |
|---|---|---|---|
| **Greedy** | Pick highest-probability token at each step | Structured output; T=0 determinism | Repetitive loops; open-ended generation |
| **Beam search** | Maintain k hypotheses; pick highest joint probability | Translation; structured / constrained output | Open-ended / creative text (stiff, repetitive) |
| **Temperature sampling** | Scale logits by 1/T before softmax; sample | General generation with T tuning | T=0 loses diversity; T>1.5 incoherent |
| **Top-k** | Zero out all tokens outside top-k; sample remainder | Prevents low-prob nonsense | Fixed k can be too broad or too narrow |
| **Top-p (nucleus)** | Keep smallest set with cumulative prob ≥ p; sample | Adaptive vocabulary truncation; default p=0.9–0.95 | Very low p → near-greedy; very high p → noisy |
| **Min-p** | Keep tokens with prob ≥ min_p × max_token_prob | Scales threshold with confidence dynamically | Less standard; fewer framework defaults |
| **Typical-p** | Sample tokens with information content near expected entropy | Coherent yet diverse text | Less widely supported |
| **Mirostat** | PID-style controller targeting a fixed perplexity τ | Consistent complexity throughout long output | Complex to tune; limited framework support |

---

## Temperature: effect on distribution

```
logit_scaled_i = logit_i / T       # applied before softmax
```

| T value | Effect | Typical use |
|---|---|---|
| 0 | Greedy (argmax) — fully deterministic | JSON extraction, code, fact recall |
| 0.1–0.3 | Sharp distribution; near-deterministic | Structured output with some variation |
| 0.7 | Conventional default; balanced | Chat, summarisation |
| 1.0 | Raw model distribution; unmodified | Sampling from training distribution |
| >1.5 | Flat distribution; high randomness | Data augmentation, brainstorming (often incoherent) |

---

## Beam vs sampling quick-reference

| Task | Preferred strategy |
|---|---|
| Machine translation | Beam (length-normalised) |
| Code generation (exact) | Greedy / low-T sampling |
| Structured output (JSON, SQL) | Greedy + constrained decoding |
| Open-ended / creative | Temperature + top-p |
| Self-consistency reasoning | Temperature >0 to get diverse paths |

---

## Speculative decoding

**Mechanism:** small *draft model* generates k tokens speculatively → *target model* verifies all k in one forward pass → accept prefix of matching tokens; reject and resample at first mismatch.

```
Speedup condition:  draft acceptance rate high  AND  system is memory-bandwidth-bound
No benefit when:    compute-bound (large batch) OR  draft acceptance rate low
```

**Variants:**
- **Medusa** — extra decoding heads on the target model; no separate draft model.
- **EAGLE / EAGLE-3** — auto-regressive draft at feature (hidden-state) level; higher acceptance rates; native in TensorRT-LLM.

---

## Constrained decoding

Grammar-constrained decoding masks invalid tokens at every step via a finite-state machine derived from a schema or grammar. Libraries: **outlines**, **lm-format-enforcer**, llama.cpp grammars, SGLang.

| Mode | Guarantee | Notes |
|---|---|---|
| JSON mode (API) | Syntactically valid JSON | Schema conformance not guaranteed |
| Grammar-constrained | Syntactically valid + schema-conformant | ~ms overhead per token to advance FSM |
| Prompt-only formatting | None | Requires post-parse validation and retry logic |

Neither grammar-constrained decoding nor JSON mode guarantees *semantic* correctness — only structural validity.

---

## If you only remember three things

1. **T=0 = greedy** (deterministic); top-p and top-k are applied *after* temperature scaling, not instead of it.
2. **Speculative decoding** is lossless (same output distribution as target-only) and only helps when the system is memory-bandwidth-bound.
3. **Grammar-constrained decoding** eliminates parse failures; it does not fix hallucinations or wrong values.

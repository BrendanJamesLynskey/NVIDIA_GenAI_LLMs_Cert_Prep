# Behavioural — NVIDIA

Twelve behavioural questions in the style NVIDIA's hiring panels actually use, with STAR scaffolds drawn from the public portfolio in the same GitHub organisation. The scaffolds are templates, not scripted answers — swap in your own professional incidents where you have something stronger, but keep the structure and the same level of concrete detail.

NVIDIA's behavioural lens leans on five things in particular:

1. **Technical depth in your strongest area** — they will pull on threads to see how far down they go.
2. **Ownership and impact** — driving outcomes, not just executing tasks.
3. **Cross-functional collaboration on the hardware/software boundary** — this matters more here than at most companies.
4. **Comfort with ambiguity** — research-flavoured problems, evolving stacks, vague specs.
5. **Curiosity and continuous learning** — the stack changes every quarter and they expect you to keep up.

Use the [LLM hub](https://github.com/BrendanJamesLynskey/LLMs) and the per-topic repos to anchor your answers in real artefacts.

---

## Q1 — Going beyond the formal scope

> "Tell me about a time you took on a problem that wasn't strictly in your remit, because you saw it needed doing."

**Why this question.** NVIDIA wants ownership, not job-description compliance. They are looking for someone who closes the gap between what was asked for and what is actually needed.

**STAR scaffold — anchor on [`Plane_Forge`](https://github.com/BrendanJamesLynskey/Plane_Forge).**

- **Situation.** Side project building parametric aircraft for FDM 3D printing on an Ender-3 V2 Neo. Existing slicers don't validate aircraft-model structural integrity — thin walls on swept wings, fragile elevator joints, unprintable overhangs at fuselage/canopy junctions.
- **Task.** Make printable aircraft designs reliable end-to-end — from parametric edit through STL export to slicer hand-off — without the user discovering print failures three hours into a print.
- **Action.** Built a printability validator into the export path: build-volume bounds, thin-wall detection, watertightness check on the STL, overhang heuristics. Added presets (Spitfire, P-51, Cessna 172) so users with no aero knowledge could start from a known-good baseline. The validator runs before export, not after slicing.
- **Result.** Print-failure rate on the Ender-3 dropped to near-zero on validated exports. The design loop tightened from "print and find out" to "validate, then print". Code is in the React/Three.js front end; the validator is a small piece of an otherwise unrelated tool. ([Plane_Forge README](https://github.com/BrendanJamesLynskey/Plane_Forge))

**Pitfalls.** Don't list features. Lead with the gap you saw and the specific failure mode you eliminated.

---

## Q2 — Owning a complex system end-to-end

> "Walk me through a system you owned from architecture through to verification. Where did you make the most consequential decisions?"

**Why this question.** They want to see you make architecture-level calls and stand behind them in front of a panel. The "most consequential decisions" framing tests whether you can prioritise.

**STAR scaffold — anchor on [`LLM_Transformer_Decoder_RTL`](https://github.com/BrendanJamesLynskey/LLM_Transformer_Decoder_RTL).**

- **Situation.** A synthesisable SystemVerilog implementation of a GPT-2/LLaMA-style pre-norm transformer decoder block, intended as a teaching artefact and a baseline for FPGA inference work.
- **Task.** End-to-end ownership: architecture, RTL, verification, and a Python golden model — all consistent, all reproducible.
- **Action.** Three consequential decisions:
  1. **Q8.8 signed fixed-point** rather than FP16 — drops floating-point units, maps 16×16 multipliers directly to DSP48 primitives, makes area and power tractable on a small FPGA. Cost: 1/256 resolution, requires careful range analysis at every accumulation point.
  2. **Pre-norm decoder** rather than post-norm — matches modern LLaMA-class designs and gives more stable gradients on deep stacks. Pre-norm also means the residual path is unscaled, which simplifies the HW datapath.
  3. **Three-layer verification** (CocoTB + SV testbenches + Python golden model, 84 tests) — surface mismatches early, against a model that's independent of the RTL.
- **Result.** 84 passing tests, KV-cache support, parameters centralised in one package file so D_MODEL / N_HEADS / sequence length scale by editing a single file. The verification rigour is what makes the RTL trustable as a baseline. ([repo](https://github.com/BrendanJamesLynskey/LLM_Transformer_Decoder_RTL))

**Pitfalls.** Don't say "I built a decoder". Say "I made these three calls, and here's the alternative I considered for each".

---

## Q3 — Technical disagreement

> "Describe a time you held an unpopular technical position and saw it through. What was the cost?"

**Why this question.** They want to see if you push back, and whether you can do it without burning bridges. The "what was the cost" framing tests honesty.

**STAR scaffold — adapt with a real professional incident if you have one. The portfolio shape is:**

- **Situation.** A technical decision where the obvious answer (e.g. "just use FP16 for the RTL transformer") was wrong for the context (FPGA area, DSP48 mapping, no FP units).
- **Task.** Convince the room — or, in solo work, convince yourself across a self-imposed sceptic round — to take the harder path.
- **Action.** Three concrete moves: (1) costed both options in the actual currency that matters (DSP48 utilisation, watts, latency), (2) built a small prototype of the disputed approach to make the abstract concrete, (3) wrote down the trade-off so the decision could be challenged later. For the fixed-point call, the numbers said Q8.8 with 32-bit accumulation gave acceptable accuracy at one-quarter of the FP16 area cost.
- **Result.** Decision held; the cost was in calibration time (range analysis at every accumulator, softmax PWL-approximation tuning, RSqrt LUT design for LayerNorm) rather than in design-review meetings.

**Pitfalls.** Don't tell a story where you were always right. NVIDIA wants to hear that you measure twice and you accept being wrong when the data says so.

---

## Q4 — Learning a new stack fast

> "Tell me about a technology you knew nothing about a year ago and are now productive in. How did you bootstrap?"

**Why this question.** The NVIDIA stack moves fast — TensorRT-LLM, NIM, NeMo Aligner becoming NeMo RL, MX-FP4 native on Blackwell. They want to see how you metabolise change.

**STAR scaffold — anchor on the [LLMs hub](https://github.com/BrendanJamesLynskey/LLMs) and topic series.**

- **Situation.** Coming from a hardware/FPGA/DSP background into agentic AI and the modern LLM stack — vLLM, TensorRT-LLM, MCP, the post-training pipeline (SFT/RLHF/DPO/CAI), evals, RAG.
- **Task.** Get to working knowledge across the stack, not surface-level — deep enough to make architecture calls, not just to copy tutorials.
- **Action.** Built rather than read: a presentation series of around seventy self-contained topic decks (`NVIDIA_GPU_01` through `_37`, `FT_01-05`, `RAG_01-07`, `LLM_Eval_01-05`, `MCP_01-05`, etc.), each forcing the synthesis of one topic into one artefact. Each deck includes an interactive demo or calculator where the topic admits one (KV cache sizer, engine-config helper, eval framework picker). The constraint of "one topic per deck, self-contained, interactive" forces honest understanding.
- **Result.** Working knowledge across the GenAI/LLM stack from tensor cores up to LLMOps; the artefacts double as evidence in interviews. The same approach is now applied to NVIDIA cert prep — see the [LLMs hub](https://github.com/BrendanJamesLynskey/LLMs).

**Pitfalls.** Don't list courses you took. Show the artefacts that prove the learning happened.

---

## Q5 — Hardware/software boundary

> "Tell me about a time you had to make a decision that crossed the hardware/software boundary. How did you reason about it?"

**Why this question.** This is the most NVIDIA-specific behavioural question on the panel. They want the hardware-software co-design instinct that distinguishes their engineers.

**STAR scaffold — anchor on [`LLM_Transformer_Decoder_RTL`](https://github.com/BrendanJamesLynskey/LLM_Transformer_Decoder_RTL).**

- **Situation.** Implementing softmax in fixed-point RTL. Software softmax is `exp(x_i) / Σ exp(x_j)` — straightforward. Hardware softmax has to fit in a few cycles, can't run a full exponential, and has to handle the full Q8.8 range without overflowing the 32-bit accumulator.
- **Task.** Pick a softmax implementation that's accurate enough for transformer attention and cheap enough to fit in the area budget.
- **Action.** Considered three approaches — full Taylor exp, log-domain, piecewise-linear (PWL) approximation. Modelled each in the Python golden model first, ran them against attention scores from a reference forward pass, measured per-token output drift. PWL gave the best accuracy/area trade: 8 segments cover the relevant range, error stays under 1/256 across the Q8.8 representable range.
- **Result.** Softmax PWL approximation in the RTL, validated in CocoTB against the Python golden model. The discipline — model in software, characterise the error, then synthesise — generalises: same pattern for the LayerNorm RSqrt LUT (32 entries plus Newton–Raphson refinement, division replaced with right-shift).

**Pitfalls.** Don't talk about hardware in software terms or vice versa. Show that you switched mental model and the error budget came from one side, the area budget from the other.

---

## Q6 — Spotting a problem nobody else saw

> "Tell me about a problem you found that nobody had flagged. How did you find it, and what did you do?"

**Why this question.** Tests whether you read code and systems critically rather than just running them.

**STAR scaffold — verification-driven, anchor on the RTL repo's testbench layer.**

- **Situation.** During RTL bring-up of a transformer block, the unit tests for individual modules (LayerNorm, attention, FFN) all passed; the integrated decoder passed simple end-to-end tests but produced subtly wrong outputs on long sequences.
- **Task.** Find it before someone built on the wrong baseline.
- **Action.** Built a third verification layer — a Python golden model that ran the same fixed-point arithmetic, segment by segment, on the same inputs as the SystemVerilog testbench. Diffed activations layer by layer. The mismatch was at an interface boundary: a one-cycle latency difference between two FSMs meant a downstream module was reading state one position behind on certain pipeline interleavings.
- **Result.** Bug surfaced, a small RTL change to assert validity before issuing the read, regression test added to keep it that way. The lesson reinforced: integration tests pass long before correctness is established; you need a ground-truth model running in lockstep.

**Pitfalls.** Don't tell a story where the bug was obvious. NVIDIA-grade bugs hide in interface timing, range edges, and off-by-one in array indices.

---

## Q7 — A mistake and what you learned

> "Tell me about a technical mistake. What did you change in your approach afterwards?"

**Why this question.** NVIDIA is looking for the absence of defensiveness. People who can name their mistakes correct course faster.

**STAR scaffold — adapt with a real incident. Sample shape that maps to the portfolio:**

- **Situation.** Early in the LLM hub work, started writing topic decks without a consistent structure — each one re-invented its own layout, terminology, and depth. Five decks in, cross-referencing was painful and the series didn't read as a series.
- **Task.** Decide whether to push forward or pay the cost of refactoring.
- **Action.** Refactored: standard hero/intro/sections/footer structure, shared dark theme, consistent terminology table per series, interactive demo as a first-class element. Cost: a week of churn on the early decks.
- **Result.** The next sixty-plus decks slotted into the structure cleanly; cross-references became trivial; topic-series indexes (`LLM_Hub_*`) became a real navigation layer. The lesson: **convention before content**. Now the cert-prep repo's `CLAUDE.md` codifies conventions before the first note gets written.

**Pitfalls.** Don't tell a sanitised story where the mistake was small. The specific lesson is more compelling than the size of the failure.

---

## Q8 — Mentoring and teaching

> "How do you transfer expertise to others? Give me an example where it worked."

**Why this question.** Senior NVIDIA engineers are expected to multiply other engineers, not just ship code.

**STAR scaffold — anchor on the Companion JOS series and the [LLMs hub](https://github.com/BrendanJamesLynskey/LLMs).**

- **Situation.** Julius O. Smith's online DSP textbooks (`Mathematics of the DFT`, `Introduction to Digital Filters`, `Spectral Audio Signal Processing`, `Physical Audio Signal Processing`, `Audio Signal Processing in Faust`) are dense and rarely have interactive companion material.
- **Task.** Make the textbook material accessible to engineers who learn by doing, not just by reading.
- **Action.** Five Companion repos, one per textbook, with interactive demos: live filter design, real-time spectral plots, physical-modelling synth examples, Faust playground. Each demo points back to the relevant chapter. Same approach later applied to the LLM hub.
- **Result.** Five companion repos plus around seventy LLM/NVIDIA topic decks, all public, all interactive, all link back to primary sources. A junior engineer can use any one as a starting point and follow the references inwards. The principle: **show, don't tell, and link to the proof**.

**Pitfalls.** Don't claim "I mentored people". Point at artefacts.

---

## Q9 — Simplifying or scaling a process

> "Tell me about a time you took a process that was working and made it scale."

**Why this question.** NVIDIA hires for scale. They want people who instinctively design processes that don't cap out at one engineer.

**STAR scaffold — anchor on the Opus-orchestrator / Sonnet-executor pattern.**

- **Situation.** Building deep technical content (presentations, notes, exercises, mock interviews) at the rate of multiple repos per week. Doing all of it personally caps at maybe one repo per week and burns out fast.
- **Task.** Get the throughput up by an order of magnitude without losing technical accuracy.
- **Action.** Adopted an orchestrator pattern: an Opus model owns architecture, voice, and final review; Sonnet sub-agents execute each phase in parallel. Phase 0 (scaffold) is orchestrator-only; Phases 1–6 (skeleton, notes, cheatsheets, exercises, mock interviews, presentations) are delegated in parallel batches with strict no-commit rules so the orchestrator merges. Phase 7 is orchestrator review and cross-reference cleanup.
- **Result.** This very repo (NVIDIA cert prep) was built with that pattern: ~21k words of notes in three parallel batches, five exercises in two parallel batches, mock-interview material likewise, and the orchestrator personally writes the high-judgement files (this one). Throughput up roughly 5×, accuracy maintained because the orchestrator owns the seams.

**Pitfalls.** Don't describe automation as a substitute for judgement. Describe automation as the way you reserve judgement for the calls that need it.

---

## Q10 — Working under ambiguity

> "Describe a project where the spec was vague or changing. How did you make progress?"

**Why this question.** Research-flavoured projects at NVIDIA rarely arrive with a clean spec. They want to see if you progress without one.

**STAR scaffold — anchor on [`MCP_Gateway_Playground`](https://github.com/BrendanJamesLynskey/MCP_Gateway_Playground).**

- **Situation.** Model Context Protocol was a moving target — protocol revisions every few months (2024-11-05, 2025-03-26, 2025-06-18 spec versions), security model evolving, transport layer shifting from HTTP+SSE to Streamable HTTP, no canonical reference implementation.
- **Task.** Build a working playground that demonstrates the gateway pattern (one client → one gateway → multiple MCP servers each exposing tools, resources, and prompts) against a protocol that won't sit still.
- **Action.** Scoped tightly: pick one spec revision, freeze it for the playground, document the freeze. Built the gateway as a thin aggregator (stdio to the servers, SSE to the client) so the protocol layer was isolated. Added both Docker and no-Docker run modes so the playground works in restricted environments. When the spec moves, only the gateway needs to follow.
- **Result.** Self-contained playground with three custom MCP servers, working web client, runs in `docker compose up` or `./scripts/dev.sh`. The architectural choice (gateway as the only stateful piece) is what made the moving spec tolerable.

**Pitfalls.** Don't say "I just kept going". Say "I drew this boundary, and that's why moving requirements stayed contained".

---

## Q11 — Influencing without authority

> "Tell me about a time you needed a decision from outside your team. How did you build the case?"

**Why this question.** NVIDIA is a 30,000-person company; nothing meaningful ships without cross-team alignment.

**STAR scaffold — adapt with a real cross-team incident. Sample shape:**

- **Situation.** A technical decision (e.g. which precision to standardise on for an inference serving stack — FP8 vs INT8 vs mixed) where the decision lives with another team but affects yours materially.
- **Task.** Get the right call made, with the people who own it, without treading on roles.
- **Action.** Three moves: (1) **frame the decision in their currency** — for an inference team, latency p95 and accuracy on their evals; for a hardware team, area and watts; for a platform team, support burden, (2) **bring data, not opinions** — measured numbers on a real model, not estimates, (3) **write the case down** — a short doc with the trade-off matrix the deciding team can take to their own review. Made one ask, then stayed out of the way.
- **Result.** Decision made within their cycle, with traceable reasoning. The doc became the artefact someone else could point at six months later when the question came back.

**Pitfalls.** Don't take credit for a decision you didn't own. Take credit for the reasoning trail.

---

## Q12 — Speed vs quality

> "Tell me about a time you had to ship faster than felt comfortable. How did you decide what to cut?"

**Why this question.** They want to see your instincts on what's load-bearing and what isn't.

**STAR scaffold — anchor on the cert-prep build itself.**

- **Situation.** Building a multi-file cert-prep portfolio piece (this repo) in one extended session with a fixed scope: 10 notes, 4 cheatsheets, 5 exercises, 4 mock-interview files, 5 presentations, plus README/syllabus/study plan.
- **Task.** Decide what gets full effort and what gets the minimum viable version, given a time budget.
- **Action.** Three cuts: (1) **cross-reference instead of duplicate** — where existing portfolio repos already covered a topic in depth, the note links to them and synthesises rather than re-derives. Saves thousands of words without losing a reader who follows the links. (2) **mark exercises as "written but not hardware-verified"** rather than burning hours on environment setup that can't be reproduced from this build context. The README states the contract explicitly, so the user smoke-tests on first run. (3) **delegated parallel-safe phases to sub-agents** while reserving the high-judgement file (this one) for the orchestrator. Phase ordering is explicit so review checkpoints land in the right places.
- **Result.** Repo shipped end-to-end in one session with the high-leverage files (this one, the system-design scenarios, the SYLLABUS mapping) given full orchestrator attention, the rest production-quality but parallel-built. The cuts are documented so the gaps are addressable, not hidden.

**Pitfalls.** Don't say "I cut quality". Say "I cut these specific things, and here's how I made the cut visible so it could be paid back later".

---

## How to use this file

- Read each prompt cold; speak the answer aloud before reading the scaffold.
- Replace any scaffold with your own incident if you have one stronger — but keep the same level of concrete detail (specific numbers, specific decisions, specific alternatives considered).
- The portfolio anchors are public artefacts; an interviewer who asks "show me" can. Be ready to walk through the actual code.
- Don't memorise — these are scaffolds, not scripts. Memorised answers fail the follow-up questions.

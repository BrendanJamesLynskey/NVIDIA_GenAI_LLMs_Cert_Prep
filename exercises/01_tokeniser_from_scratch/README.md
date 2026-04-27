# 01 — Tokeniser From Scratch

This exercise implements Byte-Pair Encoding (BPE) from scratch in pure Python, without any deep-learning dependencies. You train a BPE vocabulary on a bundled excerpt from *Pride and Prejudice* (Project Gutenberg, public domain), then encode and decode sample sentences. The goal is to make the tokenisation pipeline concrete before relying on black-box implementations such as `tiktoken` or the Hugging Face `tokenizers` library. The exercise sits at the start of the pipeline described in [notes/02\_transformer\_architecture.md](../../notes/02_transformer_architecture.md): raw text → integer token IDs → embedding lookup.

---

## Hardware requirements

No GPU required. Pure Python — runs on any machine with Python 3.9 or later. RAM requirement is negligible for the bundled corpus.

---

## The BPE algorithm

BPE was originally a data-compression algorithm (Gage, 1994) and was adapted for neural machine translation by Sennrich et al. (2016). GPT-2, GPT-4, and the LLaMA family all use BPE variants at the tokeniser level.

**Step-by-step description of the merge loop:**

1. **Initialise.** Split the corpus into whitespace-delimited words. Represent each word as a sequence of individual characters, appending a special end-of-word marker `</w>` to distinguish, say, `est` (a suffix) from `est</w>` (a complete word). The initial vocabulary is the set of all distinct characters plus `</w>`.

2. **Count adjacent pairs.** Scan every word in the corpus and count how many times each adjacent pair of tokens appears, weighted by the word's frequency. For example, if the word `the</w>` appears 200 times, the pairs `(t, h)`, `(h, e)`, and `(e, </w>)` each receive 200 votes.

3. **Merge the best pair.** The pair with the highest count is merged into a single new token. Add the new token to the vocabulary and record the merge rule. Apply the merge to every occurrence in the corpus representation.

4. **Repeat.** Go to step 2 and repeat until the vocabulary reaches the target size.

**Why it works for language modelling.** Common character sequences — morphemes, frequent words — get merged first. The resulting vocabulary is compact: frequent words become single tokens, rare words are decomposed into sub-word units, and completely novel words can still be approximated from their component characters. A vocabulary of 32 000–50 000 BPE tokens is sufficient to cover virtually all English text and most other languages.

**GPT-2 / LLaMA differences.** GPT-2 operates on Unicode bytes (256 base tokens) rather than characters. This eliminates unknown-token issues for arbitrary Unicode but increases the initial alphabet. LLaMA and its derivatives use SentencePiece with BPE, which additionally handles whitespace as an explicit token prefix (the `▁` prefix) rather than appending `</w>`.

---

## File layout

| File | Purpose |
|---|---|
| `bpe.py` | `BPETokeniser` class: `train()`, `encode()`, `decode()` |
| `train_and_test.py` | Script: trains on `corpus.txt`, prints vocabulary stats and round-trip results |
| `test_bpe.py` | pytest smoke tests |
| `corpus.txt` | ~150 lines from *Pride and Prejudice* (checked in; no download needed) |
| `requirements.txt` | `pytest` only |

---

## Setup

```bash
cd exercises/01_tokeniser_from_scratch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No downloads at first run — `corpus.txt` is checked in.

---

## Run

**Training demo:**

```bash
python train_and_test.py
```

**Smoke tests (expected to pass in under 10 seconds):**

```bash
pytest test_bpe.py -v
```

---

## Expected output

`train_and_test.py` prints:

```
Corpus length : ~7,000 characters
Merge rules learned : ~220
First 20 merge rules: common character pairs such as ('t', 'h') -> 'th', ('th', 'e') -> 'the', etc.
All round-trips passed.
Corpus compression: ~7,000 chars -> ~3,000 tokens (ratio ~0.43 tokens/char)
```

The exact merge order depends on the corpus. A compression ratio below 0.5 tokens/char at vocab_size=300 indicates the tokeniser is learning meaningful sub-word units.

`pytest test_bpe.py -v` reports all tests passing. Rough estimate: under 5 seconds on any modern CPU.

---

## What to study from this

- **The merge loop is the entire algorithm.** There is no neural network; the vocabulary is learned purely from frequency statistics. Understanding this makes it intuitive why tokenisers are trained on large, domain-representative corpora.
- **The end-of-word marker matters.** Without it, BPE cannot distinguish `the` as a prefix from `the` as a complete word, leading to incorrect decode.
- **Vocabulary size is a trade-off.** Larger vocabularies compress more aggressively (fewer tokens per character) but require larger embedding tables. GPT-2 uses 50 257 tokens; LLaMA-3 uses 128 256.
- **Encoding new text applies merge rules in training order.** The order of rules is the learned representation — changing it produces different tokenisations.
- **Embedding table context.** Each integer token ID from this exercise corresponds to one row in the embedding matrix E ∈ ℝ^{V × d_model} described in [notes/02\_transformer\_architecture.md](../../notes/02_transformer_architecture.md). The full pipeline is: raw text → this tokeniser → integer IDs → embedding lookup → transformer blocks.

---

## Further reading

- Sennrich et al. (2016), "Neural Machine Translation of Rare Words with Subword Units": <https://arxiv.org/abs/1508.07909>
- Radford et al. (2019), GPT-2 paper and tokeniser: <https://github.com/openai/gpt-2>
- Hugging Face tokenisers documentation: <https://huggingface.co/docs/tokenizers>
- [notes/02\_transformer\_architecture.md](../../notes/02_transformer_architecture.md) — embedding table and full pipeline context

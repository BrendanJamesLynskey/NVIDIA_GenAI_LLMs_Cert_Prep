# NOTE: written but not hardware-verified — smoke-test before use
"""
train_and_test.py — Train a BPE tokeniser on corpus.txt and demonstrate
encode / decode round-trips.

Run:
    python train_and_test.py

Expected output: vocabulary statistics, the first 20 merge rules, and
encode/decode results for a few sample sentences.
"""

import pathlib
import sys
from bpe import BPETokeniser


def main() -> None:
    corpus_path = pathlib.Path(__file__).parent / 'corpus.txt'
    if not corpus_path.exists():
        print(f"ERROR: corpus.txt not found at {corpus_path}", file=sys.stderr)
        sys.exit(1)

    corpus = corpus_path.read_text(encoding='utf-8')
    print(f"Corpus length : {len(corpus):,} characters")
    print(f"Corpus words  : {len(corpus.split()):,}")
    print()

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    vocab_size = 300
    tok = BPETokeniser()
    tok.train(corpus, vocab_size=vocab_size)

    print(f"Requested vocab size : {vocab_size}")
    print(f"Actual vocab size    : {tok.vocab_size()}")
    print(f"Merge rules learned  : {len(tok.merge_rules)}")
    print()

    # -----------------------------------------------------------------------
    # Show first 20 merge rules
    # -----------------------------------------------------------------------
    print("First 20 merge rules (most frequent adjacent pairs):")
    for i, (left, right) in enumerate(tok.most_common_merges(20), start=1):
        merged = left + right
        # Display </w> explicitly so the output is unambiguous.
        left_disp = repr(left) if left == '</w>' else left
        right_disp = repr(right) if right == '</w>' else right
        merged_disp = repr(merged) if '</w>' in merged else merged
        print(f"  {i:>3}. {left_disp!r} + {right_disp!r}  ->  {merged_disp!r}")
    print()

    # -----------------------------------------------------------------------
    # Vocabulary sample — show tokens that contain the end-of-word marker
    # (these are the complete-word tokens, i.e. the most common full words)
    # -----------------------------------------------------------------------
    complete_word_tokens = sorted(
        t for t in tok.vocab if t.endswith('</w>')
    )
    print(f"Complete-word tokens in vocabulary ({len(complete_word_tokens)} total), "
          f"first 30:")
    for t in complete_word_tokens[:30]:
        print(f"  {repr(t)}")
    print()

    # -----------------------------------------------------------------------
    # Encode / decode round-trips
    # -----------------------------------------------------------------------
    samples = [
        "It is a truth universally acknowledged",
        "Mr. Darcy walked off",
        "She was a woman of mean understanding",
        "The quick brown fox jumps over the lazy dog",
    ]

    print("Encode / decode round-trips:")
    all_ok = True
    for text in samples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        ok = decoded.strip() == text.strip()
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(f"  [{status}]  {repr(text)}")
        print(f"           ids    : {ids}")
        print(f"           decoded: {repr(decoded)}")
    print()

    if all_ok:
        print("All round-trips passed.")
    else:
        print("WARNING: one or more round-trips produced mismatches.")
        print("This can happen for tokens containing characters absent from "
              "the training corpus.")

    # -----------------------------------------------------------------------
    # Compression ratio — average tokens per character on the corpus itself
    # -----------------------------------------------------------------------
    all_ids = tok.encode(corpus)
    compression = len(all_ids) / len(corpus)
    print(f"\nCorpus compression: {len(corpus):,} chars -> {len(all_ids):,} tokens "
          f"(ratio {compression:.3f} tokens/char)")


if __name__ == '__main__':
    main()

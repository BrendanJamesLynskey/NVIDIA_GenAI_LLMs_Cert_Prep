# NOTE: written but not hardware-verified — smoke-test before use
"""
test_bpe.py — pytest smoke tests for the BPE tokeniser.

Run:
    pytest test_bpe.py -v

Tests cover:
  - encode -> decode round-trip for in-corpus and partially in-corpus text.
  - vocabulary determinism (same corpus + vocab_size always produces the same vocab).
  - merge-rule ordering is stable.
  - vocab size is at least the number of base characters and at most vocab_size.
"""

import pathlib
import pytest
from bpe import BPETokeniser


CORPUS_PATH = pathlib.Path(__file__).parent / 'corpus.txt'
SMALL_CORPUS = (
    "the cat sat on the mat the cat sat on the hat "
    "a man a plan a canal panama "
    "to be or not to be that is the question "
    "all that glitters is not gold "
    "the quick brown fox jumps over the lazy dog"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def trained_tokeniser_small():
    tok = BPETokeniser()
    tok.train(SMALL_CORPUS, vocab_size=80)
    return tok


@pytest.fixture(scope='module')
def trained_tokeniser_corpus():
    if not CORPUS_PATH.exists():
        pytest.skip(f"corpus.txt not found at {CORPUS_PATH}")
    corpus = CORPUS_PATH.read_text(encoding='utf-8')
    tok = BPETokeniser()
    tok.train(corpus, vocab_size=300)
    return tok


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_simple_sentence_small_corpus(self, trained_tokeniser_small):
        """Encode then decode should reproduce the original text exactly."""
        text = "the cat sat on the mat"
        ids = trained_tokeniser_small.encode(text)
        recovered = trained_tokeniser_small.decode(ids)
        assert recovered == text, (
            f"Round-trip failed.\n  Input  : {repr(text)}\n  Output : {repr(recovered)}"
        )

    def test_multiple_words_small_corpus(self, trained_tokeniser_small):
        """Multi-word sentence round-trip on small corpus."""
        text = "the quick brown fox"
        ids = trained_tokeniser_small.encode(text)
        recovered = trained_tokeniser_small.decode(ids)
        assert recovered == text

    def test_roundtrip_corpus(self, trained_tokeniser_corpus):
        """Sample sentence from the training corpus should round-trip."""
        text = "It is a truth universally acknowledged"
        ids = trained_tokeniser_corpus.encode(text)
        recovered = trained_tokeniser_corpus.decode(ids)
        assert recovered == text

    def test_single_word(self, trained_tokeniser_small):
        """Single word round-trip."""
        text = "the"
        ids = trained_tokeniser_small.encode(text)
        recovered = trained_tokeniser_small.decode(ids)
        assert recovered == text

    def test_ids_are_integers(self, trained_tokeniser_small):
        """encode() must return a list of ints."""
        ids = trained_tokeniser_small.encode("the cat")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)


# ---------------------------------------------------------------------------
# Vocabulary determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_vocab_on_repeated_training(self):
        """Training twice on the same corpus with the same vocab_size must
        produce identical vocabularies and merge rules."""
        tok1 = BPETokeniser()
        tok2 = BPETokeniser()
        tok1.train(SMALL_CORPUS, vocab_size=80)
        tok2.train(SMALL_CORPUS, vocab_size=80)

        assert tok1.vocab == tok2.vocab, "Vocabularies differ across two training runs."
        assert tok1.merge_rules == tok2.merge_rules, (
            "Merge rules differ across two training runs."
        )

    def test_vocab_size_bounds(self, trained_tokeniser_small):
        """Vocabulary must contain at least the base characters and at most
        the requested vocab_size tokens."""
        base_chars = set()
        for word in SMALL_CORPUS.split():
            for ch in word:
                base_chars.add(ch)
        base_chars.add('</w>')

        assert trained_tokeniser_small.vocab_size() >= len(base_chars), (
            "Vocabulary smaller than the number of base characters."
        )
        assert trained_tokeniser_small.vocab_size() <= 80, (
            "Vocabulary exceeds the requested vocab_size."
        )

    def test_merge_rules_are_ordered(self, trained_tokeniser_small):
        """Merge rules should be a non-empty list of 2-tuples of strings."""
        rules = trained_tokeniser_small.merge_rules
        assert len(rules) > 0
        for rule in rules:
            assert isinstance(rule, tuple) and len(rule) == 2
            assert isinstance(rule[0], str) and isinstance(rule[1], str)


# ---------------------------------------------------------------------------
# Encode correctness
# ---------------------------------------------------------------------------

class TestEncodeCorrectness:
    def test_encode_non_empty(self, trained_tokeniser_small):
        """encode() must return a non-empty list for non-empty input."""
        ids = trained_tokeniser_small.encode("cat")
        assert len(ids) > 0

    def test_all_ids_in_vocab(self, trained_tokeniser_small):
        """All returned IDs must be valid vocab indices."""
        ids = trained_tokeniser_small.encode("the quick brown fox")
        valid_ids = set(trained_tokeniser_small.vocab.values())
        for i in ids:
            assert i in valid_ids, f"ID {i} not found in vocabulary."

    def test_more_merges_fewer_tokens(self):
        """A tokeniser with more merges should produce fewer tokens on average."""
        tok_small = BPETokeniser()
        tok_large = BPETokeniser()
        tok_small.train(SMALL_CORPUS, vocab_size=50)
        tok_large.train(SMALL_CORPUS, vocab_size=100)
        text = "the cat sat on the mat"
        ids_small = tok_small.encode(text)
        ids_large = tok_large.encode(text)
        assert len(ids_large) <= len(ids_small), (
            "Larger vocabulary should not produce more tokens than smaller vocabulary."
        )

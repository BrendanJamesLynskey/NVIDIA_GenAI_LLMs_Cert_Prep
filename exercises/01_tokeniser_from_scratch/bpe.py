# NOTE: written but not hardware-verified — smoke-test before use
"""
bpe.py — Byte-Pair Encoding tokeniser implemented from scratch in pure Python.

The algorithm:
  1. Initialise the vocabulary as the set of individual characters in the corpus.
  2. Represent the corpus as a list of words, each word split into characters
     (plus a special end-of-word token, here appended as part of the character
     sequence rather than as a separate symbol, so that the tokeniser can
     reconstruct word boundaries on decode).
  3. Repeatedly count all adjacent character-pair frequencies across the corpus.
  4. Merge the most frequent pair into a single new token; add the new token to
     the vocabulary and record the merge rule.
  5. Re-encode the corpus using the new token and repeat until the target
     vocabulary size is reached.

This matches the original BPE paper by Sennrich et al. (2016) and the variant
used in GPT-2 (Radford et al., 2019), which operates on Unicode code points
rather than raw bytes but is otherwise structurally identical.

The vocabulary returned contains both the base characters and every merged
token; the merge_rules list records the order in which merges were learned,
which is the information required to encode new text.
"""

import re
import collections
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_word_freqs(corpus: str) -> Dict[Tuple[str, ...], int]:
    """
    Tokenise the corpus into whitespace-separated words and return a frequency
    dict mapping each word (represented as a tuple of characters) to its count.

    A special end-of-word marker '</w>' is appended to each word so that the
    tokeniser can later distinguish "est" (suffix) from "est</w>" (end-of-word
    standalone token) — the same technique used in the GPT-2 tokeniser.
    """
    word_freqs: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
    for word in re.findall(r'\S+', corpus):
        # Represent the word as individual characters plus end-of-word marker.
        chars = tuple(list(word) + ['</w>'])
        word_freqs[chars] += 1
    return dict(word_freqs)


def _get_pair_freqs(
    word_freqs: Dict[Tuple[str, ...], int]
) -> Dict[Tuple[str, str], int]:
    """
    Count every adjacent pair across all words, weighted by word frequency.
    Returns a dict mapping (left_token, right_token) -> total_count.
    """
    pair_freqs: Dict[Tuple[str, str], int] = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        for left, right in zip(word, word[1:]):
            pair_freqs[(left, right)] += freq
    return dict(pair_freqs)


def _merge_pair(
    pair: Tuple[str, str],
    word_freqs: Dict[Tuple[str, ...], int]
) -> Dict[Tuple[str, ...], int]:
    """
    Apply a merge rule to all words in word_freqs.

    Every occurrence of the adjacent pair (pair[0], pair[1]) in any word is
    replaced by the concatenated token pair[0]+pair[1].  Returns a new dict
    with the same frequencies but updated word representations.
    """
    merged_token = pair[0] + pair[1]
    new_word_freqs: Dict[Tuple[str, ...], int] = {}

    for word, freq in word_freqs.items():
        new_word: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BPETokeniser:
    """
    A minimal Byte-Pair Encoding tokeniser.

    Usage::

        tok = BPETokeniser()
        tok.train(open('corpus.txt').read(), vocab_size=300)
        ids = tok.encode("Hello world")
        text = tok.decode(ids)
        assert text == "Hello world"
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}          # token string  -> integer id
        self.id_to_token: Dict[int, str] = {}    # integer id    -> token string
        self.merge_rules: List[Tuple[str, str]] = []   # ordered merge rules

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: str, vocab_size: int = 300) -> None:
        """
        Learn BPE merge rules from *corpus* until the vocabulary reaches
        *vocab_size* tokens (base characters + merged tokens combined).

        Parameters
        ----------
        corpus:     Raw text — can be multi-line; all whitespace is treated
                    equivalently as a word boundary.
        vocab_size: Target vocabulary size.  Must be >= the number of unique
                    characters in the corpus.
        """
        # Build initial vocabulary from individual characters.
        word_freqs = _get_word_freqs(corpus)

        # Collect all base characters that appear in the corpus, plus </w>.
        base_chars: set = set()
        for word in word_freqs:
            for ch in word:
                base_chars.add(ch)

        # Initialise vocab with base characters.
        self.vocab = {}
        for idx, ch in enumerate(sorted(base_chars)):
            self.vocab[ch] = idx
        self.merge_rules = []

        # Iteratively merge the most frequent pair until we hit vocab_size.
        while len(self.vocab) < vocab_size:
            pair_freqs = _get_pair_freqs(word_freqs)
            if not pair_freqs:
                break  # No more pairs to merge (corpus fully encoded).

            # Choose the most frequent pair; break ties lexicographically for
            # determinism across Python versions.
            best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
            if pair_freqs[best_pair] < 2:
                break  # No pair occurs more than once — nothing useful to merge.

            # Record the merge rule and add the new token to the vocabulary.
            self.merge_rules.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)

            # Apply the merge to the corpus representation.
            word_freqs = _merge_pair(best_pair, word_freqs)

        # Build the reverse mapping for decode.
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Encode *text* to a list of integer token IDs using the learned merge
        rules.  Unknown characters (not present in the training vocabulary) are
        skipped with a warning rather than raising an exception.

        The encoding procedure applies merge rules in the order they were
        learned, which is the same order used during training.
        """
        if not self.merge_rules:
            raise RuntimeError("Call train() before encode().")

        # Split into words and apply the merge rules to each word independently.
        token_ids: List[int] = []
        for raw_word in re.findall(r'\S+', text):
            # Initialise word as individual characters + end-of-word marker.
            word = list(raw_word) + ['</w>']

            # Filter out characters absent from the vocabulary.
            word = [ch for ch in word if ch in self.vocab]

            # Apply each merge rule in order.
            for left, right in self.merge_rules:
                merged = left + right
                i = 0
                new_word: List[str] = []
                while i < len(word):
                    if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word

            token_ids.extend(self.vocab[tok] for tok in word if tok in self.vocab)

        return token_ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of integer token IDs back to a string.

        The end-of-word marker '</w>' is replaced by a space so that word
        boundaries are restored.  A trailing space from the final '</w>' is
        stripped.
        """
        if not self.id_to_token:
            raise RuntimeError("Call train() before decode().")

        raw = ''.join(self.id_to_token.get(i, '') for i in token_ids)
        # '</w>' marks the end of each word; replace with a space.
        text = raw.replace('</w>', ' ').rstrip(' ')
        return text

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.vocab)

    def most_common_merges(self, n: int = 20) -> List[Tuple[str, str]]:
        """Return the first *n* merge rules (most frequent merges learned)."""
        return self.merge_rules[:n]

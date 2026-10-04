"""ROUGE (Lin, 2004) implemented from scratch.

Tokenization and LCS backtracking follow Google's ``rouge-score`` package, the
de facto reference used in summarization papers. That way the numbers here are
directly comparable to published results (tests cross-check the two).

* ROUGE-N:    clipped n-gram overlap
* ROUGE-L:    longest common subsequence over the whole text
* ROUGE-Lsum: summary-level LCS. Each reference sentence is matched against
              the union of its LCS with every candidate sentence. Sentences
              are separated by newlines.
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from nltk.stem import PorterStemmer

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RougeScore:
    precision: float
    recall: float
    fmeasure: float

    @classmethod
    def from_counts(cls, hits: int, pred_total: int, ref_total: int) -> "RougeScore":
        precision = hits / pred_total if pred_total else 0.0
        recall = hits / ref_total if ref_total else 0.0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return cls(precision, recall, f)


class RougeScorer:
    def __init__(self, rouge_types: Iterable[str] = ROUGE_TYPES, use_stemmer: bool = True):
        self.rouge_types = tuple(rouge_types)
        unknown = set(self.rouge_types) - set(ROUGE_TYPES)
        if unknown:
            raise ValueError(f"Unsupported ROUGE types {sorted(unknown)}; choose from {ROUGE_TYPES}")
        self._stemmer = PorterStemmer() if use_stemmer else None
        self._stem_cache: dict[str, str] = {}

    def tokenize(self, text: str) -> list[str]:
        tokens = _NON_ALNUM_RE.sub(" ", text.lower()).split()
        if self._stemmer is None:
            return tokens
        return [self._stem(t) if len(t) > 3 else t for t in tokens]

    def score(self, reference: str, prediction: str) -> dict[str, RougeScore]:
        """Score one prediction. For ROUGE-Lsum, separate sentences with newlines."""
        ref_tokens, pred_tokens = self.tokenize(reference), self.tokenize(prediction)
        scores = {}
        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                scores[rouge_type] = rouge_l(ref_tokens, pred_tokens)
            elif rouge_type == "rougeLsum":
                scores[rouge_type] = rouge_lsum(
                    [self.tokenize(s) for s in reference.split("\n")],
                    [self.tokenize(s) for s in prediction.split("\n")],
                )
            else:
                scores[rouge_type] = rouge_n(ref_tokens, pred_tokens, int(rouge_type[-1]))
        return scores

    def _stem(self, token: str) -> str:
        stemmed = self._stem_cache.get(token)
        if stemmed is None:
            stemmed = self._stem_cache[token] = self._stemmer.stem(token)
        return stemmed


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(ref_tokens: Sequence[str], pred_tokens: Sequence[str], n: int) -> RougeScore:
    return rouge_n_from_counts(ngrams(ref_tokens, n), ngrams(pred_tokens, n))


def rouge_n_from_counts(ref_ngrams: Counter, pred_ngrams: Counter) -> RougeScore:
    hits = sum((ref_ngrams & pred_ngrams).values())
    return RougeScore.from_counts(hits, sum(pred_ngrams.values()), sum(ref_ngrams.values()))


def lcs_table(a: Sequence[str], b: Sequence[str]) -> list[list[int]]:
    rows, cols = len(a), len(b)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        ai, row, prev = a[i - 1], table[i], table[i - 1]
        for j in range(1, cols + 1):
            row[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], row[j - 1])
    return table


def lcs_indices(ref: Sequence[str], pred: Sequence[str]) -> list[int]:
    """Indices into ``ref`` of one longest common subsequence (rouge-score tie-breaking)."""
    table = lcs_table(ref, pred)
    i, j, indices = len(ref), len(pred), []
    while i > 0 and j > 0:
        if ref[i - 1] == pred[j - 1]:
            indices.append(i - 1)
            i, j = i - 1, j - 1
        elif table[i][j - 1] > table[i - 1][j]:
            j -= 1
        else:
            i -= 1
    return indices[::-1]


def rouge_l(ref_tokens: Sequence[str], pred_tokens: Sequence[str]) -> RougeScore:
    if not ref_tokens or not pred_tokens:
        return RougeScore(0.0, 0.0, 0.0)
    lcs = lcs_table(ref_tokens, pred_tokens)[-1][-1]
    return RougeScore.from_counts(lcs, len(pred_tokens), len(ref_tokens))


def rouge_lsum(ref_sentences: list[list[str]], pred_sentences: list[list[str]]) -> RougeScore:
    ref_sentences = [s for s in ref_sentences if s]
    pred_sentences = [s for s in pred_sentences if s]
    ref_total = sum(len(s) for s in ref_sentences)
    pred_total = sum(len(s) for s in pred_sentences)
    if not ref_total or not pred_total:
        return RougeScore(0.0, 0.0, 0.0)

    # Each token may be credited at most as often as it occurs on both sides.
    ref_budget = Counter(t for s in ref_sentences for t in s)
    pred_budget = Counter(t for s in pred_sentences for t in s)
    hits = 0
    for ref in ref_sentences:
        union = sorted(set().union(*(lcs_indices(ref, pred) for pred in pred_sentences)))
        for token in (ref[i] for i in union):
            if ref_budget[token] > 0 and pred_budget[token] > 0:
                hits += 1
                ref_budget[token] -= 1
                pred_budget[token] -= 1
    return RougeScore.from_counts(hits, pred_total, ref_total)

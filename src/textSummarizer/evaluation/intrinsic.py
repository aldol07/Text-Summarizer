"""Reference-free quality signals for when no gold summary exists.

ROUGE needs a human reference. For arbitrary user input, these proxies are
shown instead:

* coverage:    cosine between the summary's and the document's TF-IDF vectors
               (does the summary talk about what the document talks about?)
* redundancy:  mean pairwise cosine between summary sentences (lower is better)
* compression: summary words / document words

Both texts are vectorized over the document's sentences, so this works for
extractive and (later) abstractive summaries alike.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from textSummarizer.extractive.vectorizer import TfidfVectorizer
from textSummarizer.features import compression_ratio
from textSummarizer.preprocessing import Document


@dataclass(frozen=True)
class IntrinsicMetrics:
    coverage: float
    redundancy: float
    compression: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def intrinsic_metrics(doc: Document, summary: Document) -> IntrinsicMetrics:
    if len(doc) == 0 or len(summary) == 0:
        return IntrinsicMetrics(0.0, 0.0, 0.0)

    tm = TfidfVectorizer(normalize=False).fit_transform(doc.term_lists)
    doc_vec = tm.matrix.sum(axis=0)
    summary_vecs = np.array([_project(s.terms, tm.vocabulary, tm.idf) for s in summary.sentences])

    coverage = _cosine(summary_vecs.sum(axis=0), doc_vec)
    pairs = list(combinations(range(len(summary_vecs)), 2))
    redundancy = float(np.mean([_cosine(summary_vecs[i], summary_vecs[j]) for i, j in pairs])) if pairs else 0.0

    return IntrinsicMetrics(
        coverage=round(coverage, 4),
        redundancy=round(redundancy, 4),
        compression=compression_ratio(doc.word_count, summary.word_count),
    )


def _project(terms: list[str], vocabulary: dict[str, int], idf: np.ndarray) -> np.ndarray:
    """TF-IDF vector in the document's term space. Unknown terms are dropped."""
    vec = np.zeros(len(vocabulary))
    for term, count in Counter(terms).items():
        col = vocabulary.get(term)
        if col is not None:
            vec[col] = count * idf[col]
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0

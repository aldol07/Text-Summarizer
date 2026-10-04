"""TF-IDF vector space model implemented with numpy.

Each sentence is treated as a "document", so IDF rewards terms that are
specific to a few sentences and penalizes terms spread across the whole text.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TermMatrix:
    matrix: np.ndarray  # shape (n_sentences, n_terms)
    vocabulary: dict[str, int]
    idf: np.ndarray

    @property
    def terms(self) -> list[str]:
        return sorted(self.vocabulary, key=self.vocabulary.__getitem__)


class TfidfVectorizer:
    """
    tf  = raw count, ``1 + log(count)`` if ``sublinear_tf``, or 0/1 if ``binary``
    idf = ``log((1 + n) / (1 + df)) + 1`` (smoothed, never zero)
    Rows are L2-normalized when ``normalize`` is set.
    """

    def __init__(
        self,
        sublinear_tf: bool = False,
        binary: bool = False,
        use_idf: bool = True,
        normalize: bool = True,
    ):
        self.sublinear_tf = sublinear_tf
        self.binary = binary
        self.use_idf = use_idf
        self.normalize = normalize

    def fit_transform(self, docs: Sequence[Sequence[str]]) -> TermMatrix:
        vocabulary: dict[str, int] = {}
        for doc in docs:
            for term in doc:
                vocabulary.setdefault(term, len(vocabulary))

        counts = np.zeros((len(docs), len(vocabulary)), dtype=np.float64)
        for row, doc in enumerate(docs):
            for term, count in Counter(doc).items():
                counts[row, vocabulary[term]] = count

        if self.binary:
            tf = (counts > 0).astype(np.float64)
        elif self.sublinear_tf:
            tf = np.zeros_like(counts)
            np.log(counts, out=tf, where=counts > 0)
            tf[counts > 0] += 1.0
        else:
            tf = counts

        n_docs = len(docs)
        if self.use_idf:
            df = (counts > 0).sum(axis=0)
            idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        else:
            idf = np.ones(len(vocabulary))

        weights = tf * idf
        if self.normalize:
            weights = l2_normalize(weights)
        return TermMatrix(matrix=weights, vocabulary=vocabulary, idf=idf)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows; all-zero rows get similarity 0."""
    unit = l2_normalize(matrix)
    return np.clip(unit @ unit.T, 0.0, 1.0)

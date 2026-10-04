import math

import numpy as np

from textSummarizer.extractive.base import ExtractiveSummarizer
from textSummarizer.extractive.graph import pagerank
from textSummarizer.extractive.vectorizer import TfidfVectorizer, cosine_similarity
from textSummarizer.preprocessing import Document, Preprocessor

SIMILARITY_FUNCTIONS = ("overlap", "cosine")


class TextRankSummarizer(ExtractiveSummarizer):
    """TextRank (Mihalcea & Tarau, 2004).

    Builds a weighted sentence graph and ranks nodes with PageRank. The default
    edge weight is the paper's normalized word overlap::

        sim(Si, Sj) = |Si ∩ Sj| / (log|Si| + log|Sj|)

    ``similarity="cosine"`` uses TF-IDF cosine instead.
    """

    name = "textrank"
    description = "PageRank over a sentence-similarity graph (Mihalcea & Tarau, 2004)"

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        similarity: str = "overlap",
        damping: float = 0.85,
    ):
        super().__init__(preprocessor)
        if similarity not in SIMILARITY_FUNCTIONS:
            raise ValueError(f"similarity must be one of {SIMILARITY_FUNCTIONS}")
        self.similarity = similarity
        self.damping = damping

    def score_sentences(self, doc: Document) -> np.ndarray:
        if self.similarity == "cosine":
            weights = cosine_similarity(TfidfVectorizer().fit_transform(doc.term_lists).matrix)
        else:
            weights = overlap_similarity(doc.term_lists)
        return pagerank(weights, damping=self.damping)


def overlap_similarity(term_lists: list[list[str]]) -> np.ndarray:
    n = len(term_lists)
    sets = [set(terms) for terms in term_lists]
    logs = [math.log(len(terms)) if terms else 0.0 for terms in term_lists]
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(sets[i] & sets[j])
            if shared:
                denominator = logs[i] + logs[j]
                weights[i, j] = weights[j, i] = shared / denominator if denominator > 0 else float(shared)
    return weights

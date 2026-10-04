import numpy as np

from textSummarizer.extractive.base import ExtractiveSummarizer
from textSummarizer.extractive.graph import pagerank
from textSummarizer.extractive.vectorizer import TfidfVectorizer, cosine_similarity
from textSummarizer.preprocessing import Document, Preprocessor


class LexRankSummarizer(ExtractiveSummarizer):
    """LexRank (Erkan & Radev, 2004).

    Eigenvector centrality on a graph whose edges are IDF-modified cosine
    similarities. With a ``threshold`` (paper default 0.1), edges are binarized
    so a sentence's rank depends on *how many* similar sentences it has, not
    their total weight. ``threshold=None`` gives "continuous LexRank".
    IDF is estimated from the document's own sentences (no external corpus).
    """

    name = "lexrank"
    description = "Eigenvector centrality on a thresholded cosine graph (Erkan & Radev, 2004)"

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        threshold: float | None = 0.1,
        damping: float = 0.85,
    ):
        super().__init__(preprocessor)
        if threshold is not None and not 0.0 <= threshold < 1.0:
            raise ValueError("threshold must be in [0, 1) or None")
        self.threshold = threshold
        self.damping = damping

    def score_sentences(self, doc: Document) -> np.ndarray:
        similarity = cosine_similarity(TfidfVectorizer().fit_transform(doc.term_lists).matrix)
        if self.threshold is not None:
            similarity = (similarity >= self.threshold).astype(np.float64)
        return pagerank(similarity, damping=self.damping)

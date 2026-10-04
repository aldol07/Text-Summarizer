import numpy as np

from textSummarizer.extractive.base import ExtractiveSummarizer
from textSummarizer.extractive.vectorizer import TfidfVectorizer, l2_normalize
from textSummarizer.preprocessing import Document, Preprocessor

SCORINGS = ("sum", "cosine")


class TfidfCentroidSummarizer(ExtractiveSummarizer):
    """Centroid-based summarization (MEAD; Radev et al., 2004).

    The centroid holds the mean TF-IDF weight of each term across sentences,
    pruned to its ``centroid_terms`` heaviest terms. A sentence scores as the
    **sum of the centroid weights of the distinct terms it contains** (MEAD's
    centroid value). That rewards sentences covering many central terms.

    ``scoring="cosine"`` scores by cosine similarity to the centroid instead.
    Cosine ignores sentence length, so it tends to prefer short sentences,
    which hurts ROUGE recall. It is kept for ablations.
    """

    name = "tfidf"
    description = "Sum of centroid TF-IDF weights of the sentence's terms (MEAD centroid)"

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        centroid_terms: int | None = 20,
        scoring: str = "sum",
    ):
        super().__init__(preprocessor)
        if scoring not in SCORINGS:
            raise ValueError(f"scoring must be one of {SCORINGS}")
        self.centroid_terms = centroid_terms
        self.scoring = scoring

    def score_sentences(self, doc: Document) -> np.ndarray:
        weights = TfidfVectorizer(normalize=False).fit_transform(doc.term_lists).matrix
        if weights.size == 0:
            return np.zeros(len(doc))

        centroid = weights.mean(axis=0)
        if self.centroid_terms is not None and self.centroid_terms < centroid.size:
            cutoff = np.partition(centroid, -self.centroid_terms)[-self.centroid_terms]
            centroid = np.where(centroid >= cutoff, centroid, 0.0)

        if self.scoring == "cosine":
            return l2_normalize(weights) @ l2_normalize(centroid[None, :])[0]
        return (weights > 0).astype(np.float64) @ centroid

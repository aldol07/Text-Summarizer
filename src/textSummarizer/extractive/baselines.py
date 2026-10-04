"""Reference baselines every summarization paper reports against."""

import numpy as np

from textSummarizer.extractive.base import ExtractiveSummarizer
from textSummarizer.preprocessing import Document, Preprocessor


class LeadSummarizer(ExtractiveSummarizer):
    """Lead-k: the first k sentences. A strong baseline on news text."""

    name = "lead"
    description = "First sentences of the text (Lead-k baseline)"

    def score_sentences(self, doc: Document) -> np.ndarray:
        return np.arange(len(doc), 0, -1, dtype=np.float64)


class RandomSummarizer(ExtractiveSummarizer):
    """Uniformly random sentences: the floor any real method must beat."""

    name = "random"
    description = "Random sentences (lower-bound baseline)"

    def __init__(self, preprocessor: Preprocessor | None = None, seed: int | None = 42):
        super().__init__(preprocessor)
        self.seed = seed

    def score_sentences(self, doc: Document) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(doc))

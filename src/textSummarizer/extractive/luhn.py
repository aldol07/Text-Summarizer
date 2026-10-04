import math
from collections import Counter

import numpy as np

from textSummarizer.extractive.base import ExtractiveSummarizer
from textSummarizer.preprocessing import Document, Preprocessor


class LuhnSummarizer(ExtractiveSummarizer):
    """Luhn (1958): sentences dense in significant words are important.

    Significant words are the most frequent content terms. Within a sentence,
    significant words separated by at most ``max_gap`` insignificant words
    form a cluster scored ``significant_count ** 2 / cluster_span``. A sentence
    scores as its best cluster.
    """

    name = "luhn"
    description = "Word-frequency clusters (Luhn, 1958)"

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        top_fraction: float = 0.15,
        min_frequency: int = 2,
        max_gap: int = 4,
    ):
        super().__init__(preprocessor)
        self.top_fraction = top_fraction
        self.min_frequency = min_frequency
        self.max_gap = max_gap

    def score_sentences(self, doc: Document) -> np.ndarray:
        significant = self._significant_terms(doc)
        return np.array([self._sentence_score(s.norms, s.is_content, significant) for s in doc.sentences])

    def _significant_terms(self, doc: Document) -> set[str]:
        freq = Counter(t for terms in doc.term_lists for t in terms)
        if not freq:
            return set()
        ranked = [t for t, _ in freq.most_common()]
        top = ranked[: max(1, math.ceil(self.top_fraction * len(ranked)))]
        frequent = {t for t in top if freq[t] >= self.min_frequency}
        return frequent or set(top)  # short texts: every term appears once

    def _sentence_score(self, norms, is_content, significant: set[str]) -> float:
        positions = [i for i, (n, keep) in enumerate(zip(norms, is_content, strict=True)) if keep and n in significant]
        if not positions:
            return 0.0

        best, start, count = 0.0, positions[0], 1
        for prev, pos in zip(positions, positions[1:], strict=False):
            if pos - prev - 1 <= self.max_gap:
                count += 1
                continue
            best = max(best, count**2 / (prev - start + 1))
            start, count = pos, 1
        return max(best, count**2 / (positions[-1] - start + 1))

"""Common interface for extractive summarizers.

Subclasses only implement :meth:`ExtractiveSummarizer.score_sentences`.
Length control, position weighting, minimum-length filtering, MMR
re-ranking, and output formatting are shared here, so every algorithm is
compared under identical selection rules.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from textSummarizer.extractive.result import ScoredSentence, SummaryResult
from textSummarizer.extractive.selection import (
    min_max,
    mmr_select,
    position_prior,
    resolve_length,
    select_top_k,
)
from textSummarizer.extractive.vectorizer import TfidfVectorizer, cosine_similarity
from textSummarizer.preprocessing import Document, Preprocessor

_TERMINAL = (".", "!", "?", '"', "'", ")", ":", "…")


def join_sentences(sentences: list[str]) -> str:
    """Join extracted sentences into running text.

    Headlines and dialogue turns often lack final punctuation. Without one they
    would run into the next sentence ("...for Public Schools The city council..."),
    so a period is added.
    """
    return " ".join(s if s.endswith(_TERMINAL) else f"{s}." for s in sentences)


class ExtractiveSummarizer(ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    def __init__(self, preprocessor: Preprocessor | None = None):
        self.preprocessor = preprocessor or Preprocessor()

    @abstractmethod
    def score_sentences(self, doc: Document) -> np.ndarray:
        """Return one importance score per sentence (higher is more important)."""

    def summarize(self, text: str, **options) -> SummaryResult:
        return self.summarize_document(self.preprocessor.process(text), **options)

    def summarize_document(
        self,
        doc: Document,
        num_sentences: int | None = None,
        ratio: float | None = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        position_weight: float = 0.0,
        min_words: int = 0,
    ) -> SummaryResult:
        if not 0.0 <= position_weight <= 1.0:
            raise ValueError("position_weight must be in [0, 1]")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")

        n = len(doc)
        k = resolve_length(n, num_sentences, ratio)
        if n == 0:
            return SummaryResult(method=self.name, summary="", selected_indices=[])

        scores = np.asarray(self.score_sentences(doc), dtype=np.float64)
        if scores.shape != (n,):
            raise RuntimeError(f"{type(self).__name__} returned {scores.shape} scores for {n} sentences")

        relevance = min_max(scores)
        if position_weight > 0:
            relevance = (1.0 - position_weight) * relevance + position_weight * position_prior(n)

        candidates = np.array([s.word_count >= min_words for s in doc.sentences])
        if candidates.sum() < k:
            candidates[:] = True  # too few long sentences: don't starve the summary

        if use_mmr and k > 1:
            similarity = cosine_similarity(TfidfVectorizer().fit_transform(doc.term_lists).matrix)
            selected = mmr_select(relevance, similarity, k, candidates, mmr_lambda)
        else:
            selected = select_top_k(relevance, k, candidates)

        selected = sorted(selected)  # present in original reading order
        chosen = set(selected)
        sentences = [
            ScoredSentence(
                index=s.index, text=s.text, score=round(float(relevance[s.index]), 4), selected=s.index in chosen
            )
            for s in doc.sentences
        ]
        return SummaryResult(
            method=self.name,
            summary=join_sentences([doc.sentences[i].text for i in selected]),
            selected_indices=selected,
            sentences=sentences,
        )

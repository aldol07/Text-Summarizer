"""Turning sentence scores into a summary: length control, top-k, and MMR."""

import math

import numpy as np

DEFAULT_NUM_SENTENCES = 3


def resolve_length(
    n_sentences: int,
    num_sentences: int | None = None,
    ratio: float | None = None,
) -> int:
    """Number of sentences to select, clamped to ``[1, n_sentences]``."""
    if n_sentences == 0:
        return 0
    if num_sentences is not None and ratio is not None:
        raise ValueError("Pass either num_sentences or ratio, not both")
    if num_sentences is not None:
        if num_sentences < 1:
            raise ValueError("num_sentences must be >= 1")
        k = num_sentences
    elif ratio is not None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        k = math.ceil(ratio * n_sentences)
    else:
        k = DEFAULT_NUM_SENTENCES
    return max(1, min(k, n_sentences))


def min_max(scores: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A constant vector maps to all ones (every sentence ties)."""
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64))
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def position_prior(n_sentences: int) -> np.ndarray:
    """Linearly decaying prior: 1.0 for the first sentence down to 0.0 for the last."""
    if n_sentences <= 1:
        return np.ones(n_sentences)
    return 1.0 - np.arange(n_sentences) / (n_sentences - 1)


def select_top_k(relevance: np.ndarray, k: int, candidates: np.ndarray) -> list[int]:
    """Highest-scoring candidates; ties are broken by earlier position."""
    order = [i for i in np.argsort(-relevance, kind="stable") if candidates[i]]
    return [int(i) for i in order[:k]]


def mmr_select(
    relevance: np.ndarray,
    similarity: np.ndarray,
    k: int,
    candidates: np.ndarray,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

    Greedily picks ``argmax  lambda * rel(i) - (1 - lambda) * max_{j in S} sim(i, j)``,
    trading relevance against redundancy with what is already selected.
    ``lambda_ = 1`` reduces to plain top-k.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("mmr_lambda must be in [0, 1]")

    remaining = [int(i) for i in np.flatnonzero(candidates)]
    selected: list[int] = []
    while remaining and len(selected) < k:
        if selected:
            redundancy = similarity[np.ix_(remaining, selected)].max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        mmr = lambda_ * relevance[remaining] - (1.0 - lambda_) * redundancy
        best = remaining[int(np.argmax(mmr))]
        selected.append(best)
        remaining.remove(best)
    return selected

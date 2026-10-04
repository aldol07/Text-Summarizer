"""Graph centrality for graph-based summarizers (TextRank, LexRank)."""

import numpy as np


def pagerank(
    weights: np.ndarray,
    damping: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> np.ndarray:
    """Weighted PageRank by power iteration.

    ``weights[i, j]`` is the edge weight from node i to node j. Self-loops are
    ignored. Nodes with no outgoing edges ("dangling") jump to every node
    uniformly, so the transition matrix is always stochastic. Returns a
    probability vector (sums to 1): the stationary distribution of a random
    surfer who follows an edge with probability ``damping`` and teleports
    otherwise.
    """
    if not 0.0 < damping < 1.0:
        raise ValueError("damping must be in (0, 1)")

    w = np.array(weights, dtype=np.float64)
    n = w.shape[0]
    if n == 0:
        return np.zeros(0)
    np.fill_diagonal(w, 0.0)

    row_sums = w.sum(axis=1, keepdims=True)
    transition = np.divide(w, row_sums, out=np.zeros_like(w), where=row_sums > 0)
    transition[row_sums[:, 0] == 0] = 1.0 / n

    rank = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        updated = (1.0 - damping) / n + damping * (transition.T @ rank)
        converged = np.abs(updated - rank).sum() < tol
        rank = updated
        if converged:
            break
    return rank / rank.sum()

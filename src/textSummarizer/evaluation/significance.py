"""Bootstrap confidence intervals and paired significance tests (Koehn, 2004)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_resamples, values.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return ConfidenceInterval(float(values.mean()), float(low), float(high))


def paired_bootstrap_pvalue(
    system: np.ndarray,
    baseline: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> float:
    """One-sided p-value for "system scores higher than baseline on the same documents".

    Resamples documents with replacement and counts how often the mean
    per-document difference is <= 0. Small values mean the improvement is
    unlikely to be a sampling accident.
    """
    diff = np.asarray(system, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if diff.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n_resamples, diff.size))].mean(axis=1)
    return float((means <= 0).mean())

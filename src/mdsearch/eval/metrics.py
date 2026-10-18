from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def precision_at_k(rels: Sequence[int | bool], k: int) -> float:
    """Missing results (fewer than k returned) count as non-relevant, as in the paper."""
    return float(sum(bool(r) for r in rels[:k])) / k


def average_precision_at_k(
    rels: Sequence[int | bool], k: int, n_relevant: int | None = None
) -> float:
    """AP@k = 1/min(k, n_relevant) * sum_i P(i) * rel(i).

    The paper does not say what "total relevant items" is. By default it is the number of
    relevant results within the returned top-k (AP over what was retrieved); pass the number of
    relevant products in the whole collection if you can compute it.
    """
    top = [bool(r) for r in rels[:k]]
    total = sum(top) if n_relevant is None else n_relevant
    if total == 0:
        return 0.0
    hits, acc = 0, 0.0
    for i, r in enumerate(top, start=1):
        if r:
            hits += 1
            acc += hits / i
    return acc / min(k, total)


def ndcg_at_k(
    gains: Sequence[float], k: int, ideal_gains: Sequence[float] | None = None
) -> float:
    """For graded labels (ESCI/WANDS later). `ideal_gains` = all known gains for the query."""

    def dcg(g: Sequence[float]) -> float:
        return sum(x / math.log2(i + 2) for i, x in enumerate(g[:k]))

    ideal = sorted(ideal_gains if ideal_gains is not None else gains, reverse=True)
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


def bootstrap_ci(
    values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (queries are the resampling unit)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)

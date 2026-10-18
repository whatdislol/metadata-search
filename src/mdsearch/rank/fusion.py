"""Reciprocal Rank Fusion, used for the hybrid (BM25 + dense) baseline."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class ScoreSource(Protocol):
    ids: list[str]

    def scores(self, query: str, mask: np.ndarray | None = None) -> np.ndarray: ...


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """RRF over ranked lists of ids: score(d) = sum over lists of 1 / (k + rank)."""
    out: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            out[doc_id] = out.get(doc_id, 0.0) + 1.0 / (k + rank)
    return out


def rrf_from_scores(
    arrays: Sequence[np.ndarray], k: int = 60, mask: np.ndarray | None = None
) -> np.ndarray:
    """RRF over aligned score arrays. Ranks are computed among unmasked items only."""
    n = len(arrays[0])
    idx = np.arange(n) if mask is None else np.flatnonzero(mask)
    out = np.zeros(n, dtype=float)
    for a in arrays:
        a = np.asarray(a, dtype=float)
        order = idx[np.argsort(-a[idx], kind="stable")]
        out[order] += 1.0 / (k + np.arange(1, len(order) + 1))
    if mask is not None:
        out = np.where(mask, out, -np.inf)
    return out


class HybridRetriever:
    def __init__(self, sources: Sequence[ScoreSource], k: int = 60) -> None:
        self.sources = list(sources)
        self.k = k
        self.ids = self.sources[0].ids

    def scores(self, query: str, mask: np.ndarray | None = None) -> np.ndarray:
        return rrf_from_scores([s.scores(query, mask) for s in self.sources], self.k, mask)

"""Match free-form tags to products by text similarity (no fixed fields, no per-field rules).

Each tag is embedded with the same encoder as the documents and compared with every product.
  mode="filter": for each tag keep the `top_frac` most similar products of the current pool
                 (rank-based, so there is no similarity threshold to calibrate per dataset).
  mode="soft":   remove nothing; add `weight` * mean tag similarity to the first-stage score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..query.schema import Tag
from ..rank.dense import DenseRetriever
from .metadata_filter import FilterStep


@dataclass
class TagMatch:
    mask: np.ndarray
    boost: np.ndarray  # added to first-stage scores (zeros in filter mode)
    steps: list[FilterStep] = field(default_factory=list)


class TagMatcher:
    def __init__(
        self,
        dense: DenseRetriever,
        *,
        mode: str = "filter",
        top_frac: float = 0.3,
        weight: float = 0.5,
        kinds: Iterable[str] | None = None,
    ) -> None:
        if mode not in ("filter", "soft"):
            raise ValueError("mode must be 'filter' or 'soft'")
        if not 0 < top_frac <= 1:
            raise ValueError("top_frac must be in (0, 1]")
        self.dense = dense
        self.mode = mode
        self.top_frac = top_frac
        self.weight = weight
        # None = use every tag; otherwise only tags whose role label is in this set are applied
        self.kinds = frozenset(kinds) if kinds is not None else None

    def apply(self, tags: list[Tag], mask: np.ndarray) -> TagMatch:
        n = len(mask)
        boost = np.zeros(n, dtype=float)
        if self.kinds is not None:
            tags = [t for t in tags if t.kind in self.kinds]
        if not tags:
            return TagMatch(mask, boost)
        sims = np.stack([self.dense.scores(t.text) for t in tags])  # (tags, products)
        steps: list[FilterStep] = []
        if self.mode == "soft":
            boost = self.weight * sims.mean(axis=0)
            return TagMatch(mask, boost, steps)

        new = mask.copy()
        for tag, s in zip(tags, sims):
            pool = np.flatnonzero(new)
            if len(pool) == 0:
                steps.append(FilterStep(f"tag:{tag.text}", tag.kind, 0))
                continue
            keep_n = max(1, math.ceil(self.top_frac * len(pool)))
            # exactly keep_n products; ties (e.g. duplicate texts) are broken by catalog order
            best = pool[np.argsort(-s[pool], kind="stable")[:keep_n]]
            keep = np.zeros(n, dtype=bool)
            keep[best] = True
            steps.append(FilterStep(f"tag:{tag.text}", tag.kind, int(new.sum() - keep.sum())))
            new = keep
        return TagMatch(new, boost, steps)

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Scores every document in the collection for a query (aligned with `ids`)."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        self.ids = list(ids)
        self._bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])

    def scores(self, query: str, mask: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)

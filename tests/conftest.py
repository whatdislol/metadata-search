from __future__ import annotations

import json
import zlib
from typing import Any, Sequence

import numpy as np
import pytest

from mdsearch.catalog import Catalog
from mdsearch.data.schema import Product


class FakeLLM:
    """Deterministic stand-in: `handler(prompt, system, schema) -> str`."""

    provider = "fake"
    model = "fake-1"
    temperature = 0.0
    max_workers = 4

    def __init__(self, handler=None) -> None:
        self.handler = handler or (lambda prompt, system, schema: "ok")
        self.calls: list[tuple[str, str | None, Any]] = []

    def complete(self, prompt, *, system=None, schema=None):
        self.calls.append((prompt, system, schema))
        return self.handler(prompt, system, schema)

    def complete_many(self, prompts, *, system=None, schema=None):
        return [self.complete(p, system=system, schema=schema) for p in prompts]


class FakeEncoder:
    """Hashed bag-of-words embeddings: deterministic, no model download."""

    name = "fake-bow"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], *, kind: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class FakeScorer:
    """Scores a doc by word overlap with the query."""

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        q = set(query.lower().split())
        return [float(len(q & set(d.lower().split()))) for d in docs]


@pytest.fixture
def products() -> list[Product]:
    return [
        Product("a", "LEGO Classic creative bricks", "Build anything creative", "LEGO", 29.99, 4.8, 120, 4.0),
        Product("b", "LEGO Technic racing car", "Advanced building set", "LEGO", 79.0, 4.6, 40, 9.0),
        Product("c", "Hasbro Monopoly board game", "Family board game classic", "Hasbro", 19.5, 4.2, 300, 8.0),
        Product("d", "Fisher-Price Rattle", "Soft rattle for babies", "Fisher-Price", 6.0, 4.9, 15, 0.25),
        Product("e", "Mystery toy", "No metadata at all", None, None, None, None, None),
    ]


@pytest.fixture
def catalog(products) -> Catalog:
    return Catalog(products)


def json_reply(**kw) -> str:
    return json.dumps(kw)

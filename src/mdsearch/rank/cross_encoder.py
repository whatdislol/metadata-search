from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol, Sequence


class Scorer(Protocol):
    def score(self, query: str, docs: Sequence[str]) -> Sequence[float]: ...


class CrossEncoderScorer:
    """Lazy wrapper around a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/msmarco-MiniLM-L12-en-de-v1",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise RuntimeError("Install the ML extras: pip install -e '.[ml]'") from e
            self._model = CrossEncoder(
                self.name, max_length=self.max_length, device=self.device
            )
        return self._model

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        if not docs:
            return []
        model = self._load()
        out = model.predict([(query, d) for d in docs], batch_size=self.batch_size)
        return [float(x) for x in out]


class CachedScorer:
    """Remembers every (query, document) score on disk. Methods that rerank overlapping candidates
    (rerank, rerank_hard, rerank_soft, ...) and repeated runs only pay for pairs not seen before."""

    def __init__(self, inner: "CrossEncoderScorer", path: str | Path) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "scorer")
        self.hits = self.misses = 0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY, v REAL)")
        self._prefix = f"{self.name}|{getattr(inner, 'max_length', '')}|"

    def _key(self, query: str, doc: str) -> str:
        return hashlib.sha1((self._prefix + query + "\0" + doc).encode()).hexdigest()

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        keys = [self._key(query, d) for d in docs]
        known: dict[str, float] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            marks = ",".join("?" * len(chunk))
            known.update(self._db.execute(f"SELECT k, v FROM s WHERE k IN ({marks})", chunk).fetchall())
        todo = [i for i, k in enumerate(keys) if k not in known]
        self.hits += len(docs) - len(todo)
        self.misses += len(todo)
        if todo:
            fresh = self.inner.score(query, [docs[i] for i in todo])
            rows = [(keys[i], float(v)) for i, v in zip(todo, fresh)]
            self._db.executemany("INSERT OR REPLACE INTO s VALUES (?, ?)", rows)
            self._db.commit()
            known.update(rows)
        return [known[k] for k in keys]

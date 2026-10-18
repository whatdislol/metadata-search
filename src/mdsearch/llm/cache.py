"""On-disk cache for LLM calls.

Why: the cluster is shared, runs should be deterministic, and re-running an
evaluation must not re-query prompts that were already answered.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .base import LLM


def make_key(
    provider: str,
    model: str,
    system: str | None,
    prompt: str,
    schema: dict[str, Any] | None,
    temperature: float | None,
) -> str:
    payload = json.dumps(
        [provider, model, system, prompt, schema, temperature], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteCache:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedLLM:
    """Wraps any LLM; satisfies the same interface."""

    def __init__(self, inner: LLM, cache: SQLiteCache, max_workers: int | None = None) -> None:
        self.inner = inner
        self.cache = cache
        self.provider = inner.provider
        self.model = inner.model
        self.max_workers = max_workers or getattr(inner, "max_workers", 8)
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, system: str | None, schema: dict[str, Any] | None) -> str:
        return make_key(
            self.provider, self.model, system, prompt, schema, getattr(self.inner, "temperature", None)
        )

    def complete(
        self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None
    ) -> str:
        key = self._key(prompt, system, schema)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        value = self.inner.complete(prompt, system=system, schema=schema)
        if value.strip():  # never cache empty replies: they are usually transient failures
            self.cache.set(key, value)
        return value

    def complete_many(
        self,
        prompts: list[str],
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda p: self.complete(p, system=system, schema=schema), prompts))

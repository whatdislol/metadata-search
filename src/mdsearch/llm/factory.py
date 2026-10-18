"""Build an LLM from configs/llm.yaml. This is the only place that knows provider kinds."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from ..config import config_dir, load_env, load_yaml, resolve_path
from .base import LLM, LLMError
from .cache import CachedLLM, SQLiteCache
from .openai_compat import OpenAICompatLLM

Builder = Callable[..., LLM]

_KINDS: dict[str, Builder] = {"openai_compat": OpenAICompatLLM}


def register_kind(kind: str, builder: Builder) -> None:
    """Add a provider type, e.g. a native Anthropic adapter, without touching the pipeline."""
    _KINDS[kind] = builder


def get_llm(
    name: str | None = None,
    *,
    config_path: str | Path | None = None,
    cache: bool = True,
) -> LLM:
    load_env()
    cfg = load_yaml(config_path or config_dir() / "llm.yaml")
    name = os.environ.get("MDSEARCH_LLM_PROVIDER") or name or cfg.get("default")
    providers: dict[str, Any] = cfg.get("providers", {})
    if name not in providers:
        raise LLMError(f"Unknown LLM provider {name!r}. Available: {sorted(providers)}")

    params = dict(providers[name])
    kind = params.pop("kind", "openai_compat")
    if kind not in _KINDS:
        raise LLMError(f"Unknown provider kind {kind!r}. Registered: {sorted(_KINDS)}")

    llm = _KINDS[kind](provider=name, **params)
    if not cache:
        return llm
    cache_path = resolve_path(cfg.get("cache_path", ".cache/llm_cache.sqlite"))
    return CachedLLM(llm, SQLiteCache(cache_path))

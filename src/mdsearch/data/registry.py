"""Which preparer handles which `dataset.kind`. A new dataset = one module with a `prepare(cfg)`
function plus one line here; every script then works with it through `--config`."""

from __future__ import annotations

import importlib
from typing import Any, Callable

# kind -> module (inside mdsearch.data) that defines `prepare(cfg, *, limit=None, source=None)`
PREPARERS: dict[str, str] = {
    "toys_kaggle": "toys_kaggle",
    "trec_product": "trec_product",
    "wands": "wands",
    "esci": "esci",
}


def get_preparer(kind: str) -> Callable[..., None]:
    if kind not in PREPARERS:
        raise SystemExit(f"unknown dataset.kind {kind!r}; known: {sorted(PREPARERS)}")
    return importlib.import_module(f"{__package__}.{PREPARERS[kind]}").prepare


def prepare_dataset(cfg: dict[str, Any], *, limit: int | None = None, source: str | None = None) -> None:
    get_preparer(cfg["dataset"]["kind"])(cfg, limit=limit, source=source)

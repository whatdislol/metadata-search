"""Glue that turns an experiment YAML into live objects. Used by the scripts."""

from __future__ import annotations

from typing import Any

from .catalog import Catalog
from .config import resolve_path
from .data.schema import load_products
from .llm import LLM
from .methods import build_methods
from .pipeline import SearchPipeline
from .rank.cross_encoder import CrossEncoderScorer
from .rank.dense import SentenceTransformerEncoder
from .representation import get_representation


def load_catalog(cfg: dict[str, Any]) -> Catalog:
    path = resolve_path(cfg["dataset"]["processed"])
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/01_prepare_toys.py first.")
    return Catalog(load_products(path))


def style_paths(cfg: dict[str, Any], style: str):
    """(queries file, results dir) for a query style: 'specific' (default) or 'broad'."""
    if style == "specific":
        return resolve_path(cfg["queries"]["path"]), resolve_path(cfg["eval"]["out_dir"])
    if style == "broad":
        return resolve_path(cfg["queries"]["path_broad"]), resolve_path(cfg["eval"]["out_dir_broad"])
    raise SystemExit(f"unknown --style {style!r}; use 'specific' or 'broad'")


def make_methods(
    cfg: dict[str, Any],
    catalog: Catalog,
    llm: LLM | None,
    only: tuple[str, ...] | None = None,
) -> dict[str, SearchPipeline]:
    rep, enc, ret = cfg["representation"], cfg["encoders"], cfg["retrieval"]
    max_chars = rep.get("max_chars", 2000)
    return build_methods(
        catalog,
        encoder=SentenceTransformerEncoder(enc["dense"]),
        scorer=CrossEncoderScorer(enc["cross_encoder"]),
        llm=llm,
        dense_rep=get_representation(rep["dense"], max_chars),
        bm25_rep=get_representation(rep["bm25"], max_chars),
        rerank_rep=get_representation(rep["rerank"], max_chars),
        rerank_depth=ret.get("rerank_depth", 100),
        rrf_k=ret.get("rrf_k", 60),
        embedding_cache=resolve_path(enc["embedding_cache"]) if enc.get("embedding_cache") else None,
        only=only,
    )

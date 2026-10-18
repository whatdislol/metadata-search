"""Glue that turns an experiment YAML into live objects. Used by the scripts."""

from __future__ import annotations

from typing import Any

from .catalog import Catalog
from .config import resolve_path
from .data.schema import load_products
from .llm import LLM
from .methods import build_methods
from .query.schema import ATTRIBUTE_KINDS
from .pipeline import SearchPipeline
from .rank.cross_encoder import CachedScorer, CrossEncoderScorer
from .rank.dense import SentenceTransformerEncoder
from .representation import get_representation


def _tag_kinds(value) -> tuple[str, ...] | None:
    """retrieval.tags.kinds: null = every tag; a list of tag roles; or "attributes" for the usual set."""
    if value in (None, "all"):
        return None
    return ATTRIBUTE_KINDS if value == "attributes" else tuple(value)


def _scorer(enc: dict) -> CrossEncoderScorer | CachedScorer:
    ce = CrossEncoderScorer(
        enc["cross_encoder"], batch_size=enc.get("batch_size", 32),
        max_length=enc.get("cross_encoder_max_length", 512), device=enc.get("device"),
    )
    return CachedScorer(ce, resolve_path(enc["score_cache"])) if enc.get("score_cache") else ce


def load_catalog(cfg: dict[str, Any]) -> Catalog:
    path = resolve_path(cfg["dataset"]["processed"])
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/01_prepare.py --config <config> first.")
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
    tags = ret.get("tags", {})  # settings for the generated-tag method (qam_tags)
    max_chars = rep.get("max_chars", 2000)
    return build_methods(
        catalog,
        encoder=SentenceTransformerEncoder(
            enc["dense"], batch_size=enc.get("batch_size", 32), device=enc.get("device"),
            max_seq_length=enc.get("max_seq_length"),
        ),
        scorer=_scorer(enc),
        llm=llm,
        dense_rep=get_representation(rep["dense"], max_chars),
        bm25_rep=get_representation(rep["bm25"], max_chars),
        rerank_rep=get_representation(rep["rerank"], max_chars),
        rerank_depth=ret.get("rerank_depth", 100),
        rrf_k=ret.get("rrf_k", 60),
        embedding_cache=resolve_path(enc["embedding_cache"]) if enc.get("embedding_cache") else None,
        only=only,
        tag_mode=tags.get("mode", "filter"),
        tag_top_frac=tags.get("top_frac", 0.3),
        tag_weight=tags.get("weight", 0.5),
        tag_kinds=_tag_kinds(tags.get("kinds")),
    )

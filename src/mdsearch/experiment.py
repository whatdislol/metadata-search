"""Glue that turns an experiment YAML into live objects. Used by the scripts."""

from __future__ import annotations

from typing import Any

from .catalog import Catalog
from .config import resolve_path
from .data.schema import load_products
from .llm import LLM
from .methods import MetaReps, build_methods, parse_method
from .query.schema import ATTRIBUTE_KINDS
from .pipeline import SearchPipeline
from .rank.cross_encoder import CachedScorer, CrossEncoderScorer
from .rank.dense import SentenceTransformerEncoder
from .representation import NativeMetadata, get_representation


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


def _meta_reps(rep: dict, plain: dict, catalog: Catalog, wanted: bool) -> MetaReps | None:
    """The `_meta` representations: the plain ones plus `Metadata: key=value; ...` from the corpus's own metadata.

    representation.metadata: fields (attribute names; "Brand|Manufacturer" = first found), category_levels
    (last N levels of the category path), max_value_chars, position (before|after), use (which components
    read the metadata: any of dense, bm25, rerank; default all). The LLM-written product tags
    (scripts/06_tag_products.py) are not used here."""
    m = rep.get("metadata") or {}
    fields, levels = list(m.get("fields") or []), int(m.get("category_levels") or 0)
    if not fields and levels <= 0:
        return None
    if wanted and (m.get("file") or m.get("kinds")):
        print("representation.metadata: 'file' and 'kinds' belong to the unused LLM tagging and are ignored")
    use = set(m.get("use") or ("dense", "bm25", "rerank"))
    bad = use - {"dense", "bm25", "rerank"}
    if bad:
        raise SystemExit(f"representation.metadata.use: unknown component(s) {sorted(bad)}")

    def one(component: str):
        if component not in use:
            return plain[component]  # the very same object: build_methods then reuses the plain index
        return NativeMetadata(
            plain[component], fields, levels, m.get("max_value_chars", 80), m.get("position", "before")
        )

    reps = MetaReps(one("dense"), one("bm25"), one("rerank"))
    if wanted:
        probe = NativeMetadata(plain["dense"], fields, levels, m.get("max_value_chars", 80))
        have = [p for p in catalog.products if probe.items(p)]
        print(f"product metadata: {len(have)} of {len(catalog)} products have some ({len(have) / max(len(catalog), 1):.1%})")
        if have:
            print(f"  e.g. {probe.meta_text(have[0])}")
        if len(have) < len(catalog):
            print("  products without metadata are read as plain text (re-run script 01 if none have any)")
    return reps


def make_methods(
    cfg: dict[str, Any],
    catalog: Catalog,
    llm: LLM | None,
    only: tuple[str, ...] | None = None,
) -> dict[str, SearchPipeline]:
    rep, enc, ret = cfg["representation"], cfg["encoders"], cfg["retrieval"]
    tags = ret.get("tags", {})  # settings for the _hard / _soft tag variants
    max_chars = rep.get("max_chars", 2000)
    plain = {c: get_representation(rep[c], max_chars) for c in ("dense", "bm25", "rerank")}
    wanted_meta = any((p := parse_method(n)) is not None and p.meta for n in (only or ()))
    return build_methods(
        catalog,
        encoder=SentenceTransformerEncoder(
            enc["dense"], batch_size=enc.get("batch_size", 32), device=enc.get("device"),
            max_seq_length=enc.get("max_seq_length"),
        ),
        scorer=_scorer(enc),
        llm=llm,
        dense_rep=plain["dense"],
        bm25_rep=plain["bm25"],
        rerank_rep=plain["rerank"],
        rerank_depth=ret.get("rerank_depth", 100),
        rrf_k=ret.get("rrf_k", 60),
        embedding_cache=resolve_path(enc["embedding_cache"]) if enc.get("embedding_cache") else None,
        only=only,
        tag_top_frac=tags.get("top_frac", 0.3),
        tag_weight=tags.get("weight", 0.5),
        tag_kinds=_tag_kinds(tags.get("kinds")),
        meta=_meta_reps(rep, plain, catalog, wanted_meta),
    )

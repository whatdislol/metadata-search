"""The five methods of the QAM paper plus LLM preprocessing variants (hard/soft tag filter, expansion)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .catalog import Catalog
from .filtering.metadata_filter import MetadataFilter
from .filtering.tag_matcher import TagMatcher
from .llm import LLM
from .pipeline import SearchPipeline
from .query.combined import CombinedParser
from .query.parser import QueryParser
from .query.tag_parser import TagParser
from .rank.bm25 import BM25Retriever
from .rank.cross_encoder import Scorer
from .rank.dense import DenseRetriever, Encoder
from .rank.fusion import HybridRetriever
from .representation import DocRepresentation

ALL_METHODS = ("keyword", "semantic", "rerank", "hybrid", "qam")
BASE_METHODS = ALL_METHODS
META_BASES = ("keyword", "semantic", "rerank", "hybrid")  # bases that can search product text with metadata
# A method name is a base method plus optional suffixes, always in this order:
#
#     <base>[_meta][_hard|_soft][_expand]          base: keyword, semantic, rerank, hybrid, qam
#
#   _meta    DOCUMENT side: the products' own text carries LLM-written metadata tags ("Metadata: brand=X;
#            color=Y", see product_tags.py and representation/tagged.py). The same models (BM25, nomic
#            encoder, cross-encoder) read that text; keyword/semantic/hybrid/rerank search it, so
#            `semantic_meta` is `semantic` with a metadata-carrying embedder. Needs a tag file.
#   _hard    QUERY side: the LLM writes tags (colour = blue, brand = Under Armour, ...); for each tag value
#            only the top `retrieval.tags.top_frac` most similar products of the remaining pool are kept
#   _soft    the same tags, but nothing is removed: the tag similarity (z-scored over the top
#            `rerank_depth` candidates) times `retrieval.tags.weight` is added to the candidates' z-scored
#            ranking score (the cross-encoder score where there is one, else the first-stage score)
#   _expand  the LLM also writes an expansion of the query and the first stage searches with it; the
#            cross-encoder always sees the user's original query
#
#   Examples: rerank_meta, hybrid_meta_soft_expand, keyword_expand, qam_soft.
#   `qam` is the paper's method: a HARD filter on fixed fields the LLM fills (brand, price, rating, age).
#   `qam_soft` keeps that hard fixed-field filter and adds the SOFT tag boost for what the fields cannot say.
#   `qam_hard` does not exist: two hard filters in a row overlap (brand, price, audience) and compound errors.
#   `qam_meta` does not exist: qam relies on structured fields, not on tags written into the text.
#
# Every method searches the user's own query (plus the expansion for _expand), except methods built on
# `qam`, which search the LLM's filter-free text as in the paper. Which tags a tag filter uses is a
# setting, not a method: retrieval.tags.kinds (null = all, a list of roles such as [brand, color], or
# "attributes").
# Only the five base methods are built by default; every variant is built only when named in `only`.
_METHOD_NAME = re.compile(r"^(keyword|semantic|rerank|hybrid|qam)(_meta)?(_hard|_soft)?(_expand)?$")
_RENAMED = re.compile(r"^qam_tags(?:_(hard|soft))?(_expand)?$")


class Method(NamedTuple):
    base: str
    mode: str | None  # query-side tag filter: "filter" (hard), "soft" or None
    expand: bool  # query expansion
    meta: bool  # document-side metadata tags in the product text


def parse_method(name: str) -> Method | None:
    """The parts of a method name, or None if the name is unknown."""
    m = _METHOD_NAME.match(name)
    if not m:
        return None
    base, meta, filt, expand = m.group(1), bool(m.group(2)), m.group(3), bool(m.group(4))
    if base == "qam" and (meta or filt == "_hard"):
        return None
    return Method(base, {"_hard": "filter", "_soft": "soft"}.get(filt), expand, meta)


def uses_llm(name: str) -> bool:
    """True for every method that asks an LLM to parse the query (qam, and any _hard/_soft/_expand variant).
    `_meta` alone does not: the product metadata is the corpus's own (category, attributes)."""
    p = parse_method(name)
    return p is not None and (p.base == "qam" or p.mode is not None or p.expand)


def method_catalog() -> list[str]:
    """Every valid method name, grouped by base method (the order of BASE_METHODS)."""
    names = (
        f"{b}{m}{f}{e}"
        for b in BASE_METHODS for m in ("", "_meta") for f in ("", "_hard", "_soft") for e in ("", "_expand")
    )
    return [n for n in names if parse_method(n) is not None]


_FIRST_STAGE = {"keyword": "BM25", "semantic": "dense", "rerank": "dense", "hybrid": "BM25 + dense (RRF)", "qam": "dense"}


def describe_method(name: str) -> dict[str, str]:
    """What a method does, derived from its name (no models or data needed). Keys: method, documents,
    llm (query side), filter, first_stage (what is searched and with what), second_stage, datasets."""
    parsed = parse_method(name)
    if parsed is None:
        raise _unknown([name])
    base, mode, expand, meta = parsed
    fixed = base == "qam"
    steps = (["fixed fields"] if fixed else []) + (["tags"] if mode else []) + (["expansion"] if expand else [])
    kinds = (["hard on fixed fields"] if fixed else []) + (
        [] if mode is None else ["hard on tags (top share per tag value)"] if mode == "filter" else ["soft on tags (score boost)"]
    )
    text = "LLM filter-free text" if fixed else "your query"
    return {
        "method": name,
        "documents": "title + description + product metadata" if meta else "title + description",
        "llm": ", ".join(steps) if steps else "-",
        "filter": " + ".join(kinds) if kinds else "none",
        "first_stage": f"{_FIRST_STAGE[base]} on {text}" + (" + expansion" if expand else ""),
        "second_stage": "cross-encoder on your query" if base in ("rerank", "qam") else "none",
        "datasets": "needs structured fields (toys)" if fixed else "needs product metadata (category / attributes)" if meta else "any",
    }


def _unknown(names: list[str]) -> ValueError:
    hints = []
    for n in names:
        if n.startswith("qam_meta"):
            hints.append(f"{n!r} is not offered (qam uses structured fields, not tags in the text).")
        if n.startswith("qam_hard"):
            hints.append(f"{n!r} is not offered (two hard filters in a row); use qam_soft{n[len('qam_hard'):]} or rerank_hard.")
        m = _RENAMED.match(n)
        if m:
            hints.append(f"{n!r} was renamed: use rerank_{m.group(1) or 'hard'}{m.group(2) or ''}")
    return ValueError(
        f"Unknown method(s) {names}. {' '.join(hints)} Names are <base>[_meta][_hard|_soft][_expand] with base in "
        f"{', '.join(BASE_METHODS)} (see the top of methods.py)."
    )


@dataclass
class MetaReps:
    """The representations (product text with metadata tags) the `_meta` methods use, per component."""

    dense: DocRepresentation
    bm25: DocRepresentation
    rerank: DocRepresentation


def meta_cache_path(path: str | Path | None) -> Path | None:
    """Embeddings of the metadata-carrying text live next to the plain ones: x.npz -> x_meta.npz."""
    if path is None:
        return None
    p = Path(path)
    return p.with_name(p.stem + "_meta" + p.suffix)


def build_methods(
    catalog: Catalog,
    *,
    encoder: Encoder,
    scorer: Scorer,
    llm: LLM | None,
    dense_rep: DocRepresentation,
    bm25_rep: DocRepresentation,
    rerank_rep: DocRepresentation,
    rerank_depth: int = 100,
    rrf_k: int = 60,
    embedding_cache: str | Path | None = None,
    only: tuple[str, ...] | None = None,
    tag_top_frac: float = 0.3,
    tag_weight: float = 0.5,
    tag_kinds: tuple[str, ...] | None = None,
    meta: MetaReps | None = None,
) -> dict[str, SearchPipeline]:
    """Builds the named methods in the order given (default: the five base methods).

    `llm` may be None if no requested method needs one (see `uses_llm`); `meta` is needed for `_meta` methods.
    """
    names = list(dict.fromkeys(only or ALL_METHODS))
    unknown = [n for n in names if parse_method(n) is None]
    if unknown:
        raise _unknown(unknown)
    for n in names:
        if uses_llm(n) and llm is None:
            raise ValueError(f"The {n!r} method needs an LLM")
        if parse_method(n).meta and meta is None:  # type: ignore[union-attr]
            raise ValueError(
                f"The {n!r} method needs product metadata: set representation.metadata.fields and/or "
                "category_levels in the config (see scripts/tools/profile_metadata.py)"
            )

    ids = catalog.ids
    bm25 = BM25Retriever(ids, catalog.texts(bm25_rep))
    dense = DenseRetriever.build(ids, catalog.texts(dense_rep), encoder, embedding_cache)
    hybrid = HybridRetriever([bm25, dense], k=rrf_k)
    first_stage = {"keyword": bm25, "semantic": dense, "rerank": dense, "hybrid": hybrid, "qam": dense}
    reranked = {"rerank", "qam"}  # the others have no second stage

    stages_meta: dict[str, object] | None = None
    if meta is not None and any(parse_method(n).meta for n in names):  # type: ignore[union-attr]
        # a component whose representation is the plain one (metadata switched off for it) is reused as is
        bm25_m = bm25 if meta.bm25 is bm25_rep else BM25Retriever(ids, catalog.texts(meta.bm25))
        dense_m = dense if meta.dense is dense_rep else DenseRetriever.build(
            ids, catalog.texts(meta.dense), encoder, meta_cache_path(embedding_cache)
        )
        stages_meta = {
            "dense": dense_m, "first": {
                "keyword": bm25_m, "semantic": dense_m, "rerank": dense_m,
                "hybrid": HybridRetriever([bm25_m, dense_m], k=rrf_k),
            },
        }

    methods: dict[str, SearchPipeline] = {}
    for name in names:
        base, mode, expand, with_meta = parse_method(name)  # type: ignore[misc]
        parts: dict = {}
        if base == "qam":  # fixed metadata fields (hard); tags, if any, are applied on top
            fields = QueryParser(llm, expand=expand)  # type: ignore[arg-type]
            parts["metadata_filter"] = MetadataFilter(catalog)
            parts["parser"] = fields if mode is None else CombinedParser(fields, TagParser(llm))  # type: ignore[arg-type]
        elif mode is not None or expand:  # tags are generated even when only the expansion is used
            parts["parser"] = TagParser(llm, expand=expand)  # type: ignore[arg-type]
        stage = first_stage[base]
        tag_index = dense
        rerank_text = rerank_rep.render
        if with_meta:
            stage = stages_meta["first"][base]  # type: ignore[index]
            tag_index = stages_meta["dense"]  # type: ignore[index]
            rerank_text = meta.rerank.render  # type: ignore[union-attr]
        if mode is not None:
            parts["tag_matcher"] = TagMatcher(
                tag_index, mode=mode, top_frac=tag_top_frac, weight=tag_weight, kinds=tag_kinds
            )
        methods[name] = SearchPipeline(
            name, catalog=catalog, first_stage=stage,  # type: ignore[arg-type]
            reranker=scorer if base in reranked else None,
            rerank_depth=rerank_depth, rerank_text=rerank_text, product_metadata=with_meta, **parts,
        )
    return methods

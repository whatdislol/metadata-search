"""The five methods of the QAM paper plus LLM preprocessing variants (hard/soft tag filter, expansion)."""

from __future__ import annotations

import re
from pathlib import Path

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
_METHOD_NAME = re.compile(r"^(keyword|semantic|rerank|hybrid|qam)(_hard|_soft)?(_expand)?$")
_RENAMED = re.compile(r"^qam_tags(?:_(hard|soft))?(_expand)?$")


def parse_method(name: str) -> tuple[str, str | None, bool] | None:
    """(base method, tag mode "filter"/"soft" or None, expand the query?), or None if the name is unknown."""
    m = _METHOD_NAME.match(name)
    if not m or (m.group(1) == "qam" and m.group(2) == "_hard"):
        return None
    mode = {"_hard": "filter", "_soft": "soft"}.get(m.group(2))
    return m.group(1), mode, bool(m.group(3))


def uses_llm(name: str) -> bool:
    """True for every method that asks an LLM to parse the query (qam, and any _hard/_soft/_expand variant)."""
    p = parse_method(name)
    return p is not None and (p[0] == "qam" or p[1] is not None or p[2])


def method_catalog() -> list[str]:
    """Every valid method name, grouped by base method (the order of BASE_METHODS)."""
    names = (f"{b}{f}{e}" for b in BASE_METHODS for f in ("", "_hard", "_soft") for e in ("", "_expand"))
    return [n for n in names if parse_method(n) is not None]


_FIRST_STAGE = {"keyword": "BM25", "semantic": "dense", "rerank": "dense", "hybrid": "BM25 + dense (RRF)", "qam": "dense"}


def describe_method(name: str) -> dict[str, str]:
    """What a method does, derived from its name (no models or data needed). Keys: method, llm, filter,
    first_stage (what is searched and with what), second_stage, datasets."""
    parsed = parse_method(name)
    if parsed is None:
        raise _unknown([name])
    base, mode, expand = parsed
    fixed = base == "qam"
    steps = (["fixed fields"] if fixed else []) + (["tags"] if mode else []) + (["expansion"] if expand else [])
    kinds = (["hard on fixed fields"] if fixed else []) + (
        [] if mode is None else ["hard on tags (top share per tag value)"] if mode == "filter" else ["soft on tags (score boost)"]
    )
    text = "LLM filter-free text" if fixed else "your query"
    return {
        "method": name,
        "llm": ", ".join(steps) if steps else "-",
        "filter": " + ".join(kinds) if kinds else "none",
        "first_stage": f"{_FIRST_STAGE[base]} on {text}" + (" + expansion" if expand else ""),
        "second_stage": "cross-encoder on your query" if base in ("rerank", "qam") else "none",
        "datasets": "needs structured fields (toys)" if fixed else "any",
    }


def _unknown(names: list[str]) -> ValueError:
    hints = []
    for n in names:
        if n.startswith("qam_hard"):
            hints.append(f"{n!r} is not offered (two hard filters in a row); use qam_soft{n[len('qam_hard'):]} or rerank_hard.")
        m = _RENAMED.match(n)
        if m:
            hints.append(f"{n!r} was renamed: use rerank_{m.group(1) or 'hard'}{m.group(2) or ''}")
    return ValueError(
        f"Unknown method(s) {names}. {' '.join(hints)} Names are <base>[_hard|_soft][_expand] with base in "
        f"{', '.join(BASE_METHODS)} (see the top of methods.py)."
    )


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
) -> dict[str, SearchPipeline]:
    """Builds the named methods in the order given (default: the five base methods).

    `llm` may be None if no requested method needs one (see `uses_llm`).
    """
    names = list(dict.fromkeys(only or ALL_METHODS))
    unknown = [n for n in names if parse_method(n) is None]
    if unknown:
        raise _unknown(unknown)
    for n in names:
        if uses_llm(n) and llm is None:
            raise ValueError(f"The {n!r} method needs an LLM")

    ids = catalog.ids
    bm25 = BM25Retriever(ids, catalog.texts(bm25_rep))
    dense = DenseRetriever.build(ids, catalog.texts(dense_rep), encoder, embedding_cache)
    hybrid = HybridRetriever([bm25, dense], k=rrf_k)
    first_stage = {"keyword": bm25, "semantic": dense, "rerank": dense, "hybrid": hybrid, "qam": dense}
    reranked = {"rerank", "qam"}  # the others have no second stage

    methods: dict[str, SearchPipeline] = {}
    for name in names:
        base, mode, expand = parse_method(name)  # type: ignore[misc]
        parts: dict = {}
        if base == "qam":  # fixed metadata fields (hard); tags, if any, are applied on top
            fields = QueryParser(llm, expand=expand)  # type: ignore[arg-type]
            parts["metadata_filter"] = MetadataFilter(catalog)
            parts["parser"] = fields if mode is None else CombinedParser(fields, TagParser(llm))  # type: ignore[arg-type]
        elif mode is not None or expand:  # tags are generated even when only the expansion is used
            parts["parser"] = TagParser(llm, expand=expand)  # type: ignore[arg-type]
        if mode is not None:
            parts["tag_matcher"] = TagMatcher(
                dense, mode=mode, top_frac=tag_top_frac, weight=tag_weight, kinds=tag_kinds
            )
        methods[name] = SearchPipeline(
            name, catalog=catalog, first_stage=first_stage[base],
            reranker=scorer if base in reranked else None,
            rerank_depth=rerank_depth, rerank_text=rerank_rep.render, **parts,
        )
    return methods

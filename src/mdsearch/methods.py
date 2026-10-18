"""The five methods compared in the QAM paper, built from shared components."""

from __future__ import annotations

import re
from pathlib import Path

from .catalog import Catalog
from .filtering.metadata_filter import MetadataFilter
from .filtering.tag_matcher import TagMatcher
from .llm import LLM
from .pipeline import SearchPipeline
from .query.parser import QueryParser
from .query.tag_parser import TagParser
from .rank.bm25 import BM25Retriever
from .rank.cross_encoder import Scorer
from .rank.dense import DenseRetriever, Encoder
from .rank.fusion import HybridRetriever
from .representation import DocRepresentation

ALL_METHODS = ("keyword", "semantic", "rerank", "hybrid", "qam")
# Filter types compared (one axis) and query expansion (a second, optional axis):
#
#     qam              HARD filter on fixed metadata fields the LLM fills (brand, price, rating, age)
#     qam_tags_hard    HARD filter on free-form tags the LLM writes, matched to products by text
#     qam_tags_soft    SOFT filter on the same tags: nothing removed, `tag_weight` x similarity is added
#                      to the first-stage score
#     qam_tags         qam_tags_hard or qam_tags_soft, as set by retrieval.tags.mode
#
#   Add `_expand` to any of them (qam_expand, qam_tags_hard_expand, ...) for query expansion: the LLM also
#   writes an expansion of the query (product type, synonyms, typical features) and the first stage
#   searches with it. Expansion is not a filter. The cross-encoder always sees the user's original query.
#   `rerank_expand` is the control: no filter, expansion only (compare with `rerank`).
#
# Which text the tag methods and rerank_expand search with is a setting too: retrieval.tags.query_text is
# "semantic" (the LLM's shortened query, default) or "original" (the user's query, plus the expansion).
# Which tags a tag filter uses is a setting, not a method: retrieval.tags.kinds (null = all tags, or a
# list of tag roles such as [brand, color, size]; "attributes" = the usual attribute roles).
# Everything except the five methods above is built only when named in `only`.
_METHOD_NAME = re.compile(r"^(rerank|qam|qam_tags(?:_hard|_soft)?)(_expand)?$")


def uses_llm(name: str) -> bool:
    """True for every method that asks an LLM to parse the query (qam*, rerank_expand, ...)."""
    return name == "qam" or parse_method(name) is not None and name not in ALL_METHODS


def parse_method(name: str, default_mode: str = "filter") -> tuple[str, str | None, bool] | None:
    """(base method, tag mode or None, expand the query?) or None if `name` is not one of the above."""
    m = _METHOD_NAME.match(name)
    if not m:
        return None
    base, expand = m.group(1), bool(m.group(2))
    mode = None
    if base.startswith("qam_tags"):
        mode = {"qam_tags_hard": "filter", "qam_tags_soft": "soft"}.get(base, default_mode)
    return base, mode, expand


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
    tag_mode: str = "filter",
    tag_top_frac: float = 0.3,
    tag_weight: float = 0.5,
    tag_kinds: tuple[str, ...] | None = None,
    query_text: str = "semantic",
) -> dict[str, SearchPipeline]:
    """`llm` may be None if neither "qam" nor "qam_tags" is requested."""
    if query_text not in ("semantic", "original"):
        raise ValueError("query_text must be 'semantic' or 'original'")
    wanted = set(only or ALL_METHODS)
    unknown = sorted(n for n in wanted if n not in ALL_METHODS and parse_method(n) is None)
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; see the naming scheme at the top of methods.py")
    ids = catalog.ids
    bm25 = BM25Retriever(ids, catalog.texts(bm25_rep))
    dense = DenseRetriever.build(ids, catalog.texts(dense_rep), encoder, embedding_cache)
    hybrid = HybridRetriever([bm25, dense], k=rrf_k)

    common = dict(catalog=catalog, rerank_depth=rerank_depth, rerank_text=rerank_rep.render)
    methods: dict[str, SearchPipeline] = {}
    if "keyword" in wanted:
        methods["keyword"] = SearchPipeline("keyword", first_stage=bm25, **common)
    if "semantic" in wanted:
        methods["semantic"] = SearchPipeline("semantic", first_stage=dense, **common)
    if "rerank" in wanted:
        methods["rerank"] = SearchPipeline("rerank", first_stage=dense, reranker=scorer, **common)
    if "hybrid" in wanted:
        methods["hybrid"] = SearchPipeline("hybrid", first_stage=hybrid, **common)
    if "qam" in wanted:
        if llm is None:
            raise ValueError("The 'qam' method needs an LLM")
        methods["qam"] = SearchPipeline(
            "qam",
            first_stage=dense,
            parser=QueryParser(llm),
            metadata_filter=MetadataFilter(catalog),
            reranker=scorer,
            **common,
        )
    for name in dict.fromkeys(n for n in (only or ()) if n not in ALL_METHODS):  # in the given order
        base, mode, expand = parse_method(name, tag_mode)  # type: ignore[misc]
        if llm is None:
            raise ValueError(f"The {name!r} method needs an LLM")
        if base == "qam":  # fixed metadata fields (hard) + expansion
            methods[name] = SearchPipeline(
                name, first_stage=dense, parser=QueryParser(llm, expand=expand),
                metadata_filter=MetadataFilter(catalog), reranker=scorer, **common,
            )
        else:  # rerank_expand (tags are generated but unused) or a tag filter, hard or soft
            matcher = None if mode is None else TagMatcher(
                dense, mode=mode, top_frac=tag_top_frac, weight=tag_weight, kinds=tag_kinds
            )
            methods[name] = SearchPipeline(
                name, first_stage=dense, parser=TagParser(llm, expand=expand),
                tag_matcher=matcher, reranker=scorer, first_stage_query=query_text, **common,
            )
    return methods

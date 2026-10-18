"""The five methods compared in the QAM paper, built from shared components."""

from __future__ import annotations

from pathlib import Path

from .catalog import Catalog
from .filtering.metadata_filter import MetadataFilter
from .filtering.tag_matcher import TagMatcher
from .llm import LLM
from .pipeline import SearchPipeline
from .query.parser import QueryParser
from .query.schema import ATTRIBUTE_KINDS
from .query.tag_parser import TagParser
from .rank.bm25 import BM25Retriever
from .rank.cross_encoder import Scorer
from .rank.dense import DenseRetriever, Encoder
from .rank.fusion import HybridRetriever
from .representation import DocRepresentation

ALL_METHODS = ("keyword", "semantic", "rerank", "hybrid", "qam")
# Experimental methods, built only when asked for by name (so the paper comparison is unchanged).
#   qam_tags       generated tags; hard or soft as set by `tag_mode` (config: retrieval.tags.mode)
#   qam_tags_hard  generated tags, always a hard filter (top `tag_top_frac` of the pool per tag)
#   qam_tags_soft  generated tags, always a soft filter (score boost of weight `tag_weight`)
#   qam_tags_attr  as qam_tags_hard, but only attribute tags (brand, colour, price, ...) are applied;
#                  tags that merely restate the product type are ignored
#   qam_expand     query rewriting only: the LLM expands the query (synonyms, product type, typical
#                  features) and the first stage searches with the expansion; no tag filtering.
#                  The cross-encoder still sees the user's original query.
#   qam_tags_attr_expand  qam_tags_attr plus the same query expansion
EXTRA_METHODS = (
    "qam_tags", "qam_tags_hard", "qam_tags_soft", "qam_tags_attr", "qam_expand", "qam_tags_attr_expand",
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
    tag_mode: str = "filter",
    tag_top_frac: float = 0.3,
    tag_weight: float = 0.5,
    tag_attr_kinds: tuple[str, ...] = ATTRIBUTE_KINDS,
) -> dict[str, SearchPipeline]:
    """`llm` may be None if neither "qam" nor "qam_tags" is requested."""
    wanted = set(only or ALL_METHODS)
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
    attr = tuple(tag_attr_kinds)
    # name -> (tag mode or None = no tag matching, which tag kinds are applied (None = all), expand query?)
    tag_variants = {
        "qam_tags": (tag_mode, None, False),
        "qam_tags_hard": ("filter", None, False),
        "qam_tags_soft": ("soft", None, False),
        "qam_tags_attr": ("filter", attr, False),
        "qam_expand": (None, None, True),
        "qam_tags_attr_expand": ("filter", attr, True),
    }
    for name, (mode, kinds, expand) in tag_variants.items():
        if name in wanted:  # not in ALL_METHODS, so only built when named in `only`
            if llm is None:
                raise ValueError(f"The {name!r} method needs an LLM")
            matcher = None if mode is None else TagMatcher(
                dense, mode=mode, top_frac=tag_top_frac, weight=tag_weight, kinds=kinds
            )
            methods[name] = SearchPipeline(
                name,
                first_stage=dense,
                parser=TagParser(llm, expand=expand),
                tag_matcher=matcher,
                reranker=scorer,
                **common,
            )
    return methods

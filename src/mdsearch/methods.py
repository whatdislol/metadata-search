"""The five methods compared in the QAM paper, built from shared components."""

from __future__ import annotations

from pathlib import Path

from .catalog import Catalog
from .filtering.metadata_filter import MetadataFilter
from .llm import LLM
from .pipeline import SearchPipeline
from .query.parser import QueryParser
from .rank.bm25 import BM25Retriever
from .rank.cross_encoder import Scorer
from .rank.dense import DenseRetriever, Encoder
from .rank.fusion import HybridRetriever
from .representation import DocRepresentation

ALL_METHODS = ("keyword", "semantic", "rerank", "hybrid", "qam")


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
) -> dict[str, SearchPipeline]:
    """`llm` may be None if "qam" is not requested."""
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
    return methods

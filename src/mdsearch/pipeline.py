"""One configurable search pipeline: [parse] -> [filter] -> first stage -> [rerank].

All five methods compared in the QAM replication (keyword, semantic, rerank, hybrid, qam)
are instances of this class with different parts plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from .catalog import Catalog
from .data.schema import Product
from .filtering.metadata_filter import FilterReport, MetadataFilter
from .query.parser import QueryParser
from .query.schema import ParsedQuery
from .rank.cross_encoder import Scorer


class FirstStage(Protocol):
    ids: list[str]

    def scores(self, query: str, mask: np.ndarray | None = None) -> np.ndarray: ...


@dataclass
class Hit:
    id: str
    score: float


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    parsed: ParsedQuery | None = None
    filter_report: FilterReport | None = None
    n_candidates: int = 0  # products left after filtering (== catalog size if no filter)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "hits": [{"id": h.id, "score": h.score} for h in self.hits],
            "parsed": self.parsed.model_dump() if self.parsed else None,
            "filter_report": self.filter_report.to_dict() if self.filter_report else None,
            "n_candidates": self.n_candidates,
        }


@dataclass
class SearchPipeline:
    name: str
    catalog: Catalog
    first_stage: FirstStage
    parser: QueryParser | None = None
    metadata_filter: MetadataFilter | None = None
    reranker: Scorer | None = None
    rerank_depth: int = 100
    rerank_text: Callable[[Product], str] = field(default=lambda p: f"{p.title}. {p.description}")

    def search(self, query: str, k: int = 10, parsed: ParsedQuery | None = None) -> SearchResult:
        if parsed is None and self.parser is not None:
            parsed = self.parser.parse(query)
        return self._run(query, k, parsed)

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[SearchResult]:
        """Parses all queries in parallel first (the slow, LLM-bound part), then searches."""
        parsed: list[ParsedQuery | None] = (
            list(self.parser.parse_many(list(queries))) if self.parser else [None] * len(queries)
        )
        return [self._run(q, k, p) for q, p in zip(queries, parsed)]

    # ------------------------------------------------------------------ private

    def _run(self, query: str, k: int, parsed: ParsedQuery | None) -> SearchResult:
        n = len(self.catalog)
        mask = np.ones(n, dtype=bool)
        report: FilterReport | None = None
        semantic_q = query
        if parsed is not None:
            if parsed.semantic_text.strip():
                semantic_q = parsed.semantic_text
            if self.metadata_filter is not None:
                mask, report = self.metadata_filter.apply(parsed.filters)

        scores = np.where(mask, self.first_stage.scores(semantic_q, mask), -np.inf)
        depth = self.rerank_depth if self.reranker is not None else k
        order = np.argsort(-scores, kind="stable")[: max(depth, k)]
        order = order[np.isfinite(scores[order])]

        if self.reranker is not None and len(order):
            docs = [self.rerank_text(self.catalog.products[i]) for i in order]
            # the cross-encoder sees the FULL original query, as in the paper
            ce = np.asarray(self.reranker.score(query, docs), dtype=float)
            resort = np.argsort(-ce, kind="stable")
            order, final = order[resort], ce[resort]
        else:
            final = scores[order]

        hits = [
            Hit(self.catalog.ids[i], float(s)) for i, s in zip(order[:k], final[:k])
        ]
        return SearchResult(query, hits, parsed, report, int(mask.sum()))

"""Step 1 of QAM: split a query into metadata filters and semantic text with an LLM."""

from __future__ import annotations

import logging

from ..llm import LLM, LLMError, complete_json, parse_reply
from .prompts import PARSE_SYSTEM, build_parse_prompt
from .schema import ParsedQuery

log = logging.getLogger(__name__)


class QueryParser:
    def __init__(self, llm: LLM, retries: int = 2) -> None:
        self.llm = llm
        self.retries = retries
        self.n_failures = 0  # queries that fell back to "no filters"

    def parse(self, query: str) -> ParsedQuery:
        try:
            return complete_json(
                self.llm,
                build_parse_prompt(query),
                ParsedQuery,
                system=PARSE_SYSTEM,
                retries=self.retries,
            )
        except LLMError as e:
            log.warning("Query parsing failed for %r (%s); using no filters", query, e)
            self.n_failures += 1
            return ParsedQuery(semantic_text=query)

    def parse_many(self, queries: list[str]) -> list[ParsedQuery]:
        """Parallel first pass; only malformed replies go through the slower retry path."""
        schema = ParsedQuery.model_json_schema()
        replies = self.llm.complete_many(
            [build_parse_prompt(q) for q in queries], system=PARSE_SYSTEM, schema=schema
        )
        out: list[ParsedQuery] = []
        for q, reply in zip(queries, replies):
            try:
                out.append(parse_reply(reply, ParsedQuery))
            except ValueError:
                out.append(self.parse(q))
        return out

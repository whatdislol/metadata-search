"""Step 1 of QAM: split a query into metadata filters and semantic text with an LLM."""

from __future__ import annotations

import logging

from ..llm import LLM, LLMError, complete_json, parse_reply
from .prompts import PARSE_SYSTEM, build_parse_prompt
from .schema import ParsedQuery
from .tag_parser import EXPAND_SYSTEM

log = logging.getLogger(__name__)


class QueryParser:
    def __init__(self, llm: LLM, retries: int = 2, expand: bool = False) -> None:
        self.llm = llm
        self.retries = retries
        self.expand = expand  # also ask for a query expansion (query rewriting)
        self.system = PARSE_SYSTEM + (EXPAND_SYSTEM if expand else "")
        self.n_failures = 0  # queries that fell back to "no filters"

    def _finish(self, parsed: ParsedQuery) -> ParsedQuery:
        parsed.expansion = " ".join(parsed.expansion.split()) if self.expand else ""
        return parsed

    def parse(self, query: str) -> ParsedQuery:
        try:
            return self._finish(complete_json(
                self.llm,
                build_parse_prompt(query),
                ParsedQuery,
                system=self.system,
                retries=self.retries,
            ))
        except LLMError as e:
            log.warning("Query parsing failed for %r (%s); using no filters", query, e)
            self.n_failures += 1
            return ParsedQuery(semantic_text=query)

    def parse_many(self, queries: list[str]) -> list[ParsedQuery]:
        """Parallel first pass; only malformed replies go through the slower retry path."""
        schema = ParsedQuery.model_json_schema()
        replies = self.llm.complete_many(
            [build_parse_prompt(q) for q in queries], system=self.system, schema=schema
        )
        out: list[ParsedQuery] = []
        for q, reply in zip(queries, replies):
            try:
                out.append(self._finish(parse_reply(reply, ParsedQuery)))
            except ValueError:
                out.append(self.parse(q))
        return out

"""Fixed-field filters from one parser plus free-form tags from another, for `qam_hard` / `qam_soft`."""

from __future__ import annotations

from .parser import QueryParser
from .schema import ParsedQuery
from .tag_parser import TagParser


class CombinedParser:
    """Runs a QueryParser (fixed fields, semantic text, optional expansion) and a TagParser (tags).

    The result is the QueryParser's ParsedQuery with the TagParser's tags added. Each parser sends its own
    LLM prompt, so each reply is cached separately and shared with the methods that use that parser alone.
    """

    def __init__(self, fields: QueryParser, tags: TagParser) -> None:
        self.fields, self.tags = fields, tags
        self.expand = fields.expand

    @property
    def n_failures(self) -> int:
        return self.fields.n_failures + self.tags.n_failures

    @staticmethod
    def _merge(base: ParsedQuery, tagged: ParsedQuery) -> ParsedQuery:
        return base.model_copy(update={"tags": tagged.tags})

    def parse(self, query: str) -> ParsedQuery:
        return self._merge(self.fields.parse(query), self.tags.parse(query))

    def parse_many(self, queries: list[str]) -> list[ParsedQuery]:
        qs = list(queries)
        return [self._merge(a, b) for a, b in zip(self.fields.parse_many(qs), self.tags.parse_many(qs))]

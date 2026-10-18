"""Open-vocabulary query decomposition: the LLM writes its own metadata tags (no fixed fields)."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..llm import LLM, LLMError, complete_json, parse_reply
from .schema import TAG_KINDS, ParsedQuery, Tag

log = logging.getLogger(__name__)

TAG_SYSTEM = """You turn a shopper's free-text product search query into search parameters.

Return a JSON object with:
- "semantic_text": the core of what the shopper wants, as a short search phrase. Never empty.
- "tags": a list of the requirements the products must meet, each one a short phrase you \
write yourself (1-4 words). Use whatever kind of requirement the query contains: price, brand, \
audience or age, material, size, colour, style, room, occasion, feature, quality. Each tag is an \
object {"text": ..., "kind": ...} where "text" is your own wording and "kind" is the role of the tag, \
exactly one of: {kinds}.

Rules:
- Only requirements stated in the query or clearly implied by it. Never invent preferences.
- One requirement per tag; do not repeat the product type as a tag.
- At most 6 tags. If the query states no requirement, return an empty list.
- Keep numbers and units in the tag text ("under 40 pounds", "for 3 year olds")."""
TAG_SYSTEM = TAG_SYSTEM.replace("{kinds}", ", ".join(TAG_KINDS))

EXPAND_SYSTEM = """

Also return "expansion": one short sentence (10-25 words) in the words a matching product listing \
would use: the product type, synonyms, and typical materials, features or uses of this kind of product. \
Stay faithful to the query: add no brand, number, price or requirement the shopper did not mention."""

EXPAND_EXAMPLES = """Examples:

Query: waterproof backpack under 40 pounds for hiking
{{"semantic_text": "hiking backpack", "tags": [{{"text": "waterproof", "kind": "feature"}}, {{"text": "under 40 pounds", "kind": "price"}}], "expansion": "hiking rucksack daypack with water-resistant fabric, padded shoulder straps and several compartments for trail use"}}

Query: peplum top
{{"semantic_text": "peplum top", "tags": [], "expansion": "women's peplum blouse or shirt with a flared ruffle at the waist, fitted bodice, in fabrics such as chiffon or cotton"}}

Query: cozy reading nook creation
{{"semantic_text": "reading nook", "tags": [{{"text": "cozy", "kind": "style"}}, {{"text": "seating", "kind": "feature"}}], "expansion": "comfortable armchair, bean bag or floor cushion with blanket, lamp and small bookshelf for a quiet reading corner"}}

Now the real query.

Query: {query}"""

TAG_USER = """Examples:

Query: Looking for educational toys specifically from LEGO, designed to promote creativity, suitable for children aged 5-8
{{"semantic_text": "educational toys that promote creativity", "tags": [{{"text": "LEGO", "kind": "brand"}}, {{"text": "for ages 5 to 8", "kind": "audience"}}]}}

Query: cozy reading nook creation
{{"semantic_text": "reading nook", "tags": [{{"text": "cozy", "kind": "style"}}, {{"text": "seating", "kind": "feature"}}]}}

Query: waterproof backpack under 40 pounds for hiking
{{"semantic_text": "hiking backpack", "tags": [{{"text": "waterproof", "kind": "feature"}}, {{"text": "under 40 pounds", "kind": "price"}}]}}

Query: monopoly
{{"semantic_text": "monopoly", "tags": []}}

Now the real query.

Query: {query}"""


def build_tag_prompt(query: str, expand: bool = False) -> str:
    return (EXPAND_EXAMPLES if expand else TAG_USER).format(query=query)


class _TagReply(BaseModel):
    semantic_text: str
    tags: list[Tag] = Field(default_factory=list)
    expansion: str = ""


def _to_parsed(reply: _TagReply, query: str, max_tags: int, expand: bool = False) -> ParsedQuery:
    seen: set[str] = set()
    tags: list[Tag] = []
    for t in reply.tags:
        key = t.text.lower()
        if t.text and key not in seen:
            seen.add(key)
            tags.append(t)
    return ParsedQuery(
        semantic_text=reply.semantic_text.strip() or query, tags=tags[:max_tags],
        expansion=" ".join(reply.expansion.split()) if expand else "",
    )


class TagParser:
    """Drop-in replacement for QueryParser that returns generated tags instead of fixed filters."""

    def __init__(self, llm: LLM, retries: int = 2, max_tags: int = 6, expand: bool = False) -> None:
        self.llm = llm
        self.expand = expand  # also ask for a query expansion (query rewriting)
        self.system = TAG_SYSTEM + (EXPAND_SYSTEM if expand else "")
        self.retries = retries
        self.max_tags = max_tags
        self.n_failures = 0  # queries that fell back to "no tags"

    def parse(self, query: str) -> ParsedQuery:
        try:
            reply = complete_json(
                self.llm, build_tag_prompt(query, self.expand), _TagReply, system=self.system,
                retries=self.retries,
            )
            return _to_parsed(reply, query, self.max_tags, self.expand)
        except LLMError as e:
            log.warning("Tag generation failed for %r (%s); using no tags", query, e)
            self.n_failures += 1
            return ParsedQuery(semantic_text=query)

    def parse_many(self, queries: list[str]) -> list[ParsedQuery]:
        schema = _TagReply.model_json_schema()
        replies = self.llm.complete_many(
            [build_tag_prompt(q, self.expand) for q in queries], system=self.system, schema=schema
        )
        out: list[ParsedQuery] = []
        for q, raw in zip(queries, replies):
            try:
                out.append(_to_parsed(parse_reply(raw, _TagReply), q, self.max_tags, self.expand))
            except ValueError:
                out.append(self.parse(q))
        return out

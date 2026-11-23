"""Open-vocabulary query decomposition: the LLM writes its own metadata tags (no fixed fields)."""

from __future__ import annotations

import json
import logging
from typing import Sequence

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

FILL_SYSTEM = """

Also return "metadata": an object that fills in the metadata a matching product's listing would carry, \
using ONLY these keys: {keys}. Values are short (1-4 words) and written the way a listing writes them \
(for "category", the name of the product category, e.g. "Hiking Backpacks"). Fill a key when the query \
states the value or makes it very likely (category, material or type); leave a key out when it is unknown. \
Never invent a brand, colour, size, number or price the shopper did not mention."""

FILL_OPEN_SYSTEM = """

Also return "metadata": an object that fills in the metadata a matching product's listing would carry. \
Prefer these keys, the most common in the catalog: {keys}. You may add other attributes a listing would show \
(lowercase names with underscores, e.g. "water_resistance_level"), but use at most {max} keys in total, \
preferred keys first. Values are short (1-4 words) and written the way a listing writes them (for "category", \
the name of the product category, e.g. "Hiking Backpacks"). Fill a key when the query states the value or \
makes it very likely (category, material or type); leave a key out when it is unknown. Never invent a brand, \
colour, size, number or price the shopper did not mention."""

# keys outside the preferred list that the open-key examples show
_FILL_EXTRA = {0: {"water_resistance_level": "waterproof"}}

# (query, semantic_text, tags, metadata); examples are filtered to the keys that exist in the product text
_FILL_EXAMPLES = (
    ("waterproof backpack under 40 pounds for hiking", "hiking backpack",
     [{"text": "waterproof", "kind": "feature"}, {"text": "under 40 pounds", "kind": "price"}],
     {"category": "Hiking Backpacks", "material": "waterproof nylon"}),
    ("peplum top", "peplum top", [], {"category": "Women's Blouses", "style": "peplum"}),
    ("blue Nike running shoes", "running shoes",
     [{"text": "blue", "kind": "color"}, {"text": "Nike", "kind": "brand"}],
     {"category": "Running Shoes", "brand": "Nike", "color": "blue"}),
)

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


def _fill_examples(keys: Sequence[str], open_keys: bool = False) -> str:
    allowed = set(keys)
    out = ["Examples:\n"]
    for i, (q, sem, tags, meta) in enumerate(_FILL_EXAMPLES):
        meta = {k: v for k, v in meta.items() if k in allowed}
        if open_keys:
            meta.update(_FILL_EXTRA.get(i, {}))
        reply = json.dumps({"semantic_text": sem, "tags": tags, "metadata": meta}, ensure_ascii=False)
        out.append(f"Query: {q}\n{reply}\n")
    return "\n".join(out).replace("{", "{{").replace("}", "}}") + "\nNow the real query.\n\nQuery: {query}"


def build_tag_prompt(query: str, expand: bool = False, fill_keys: Sequence[str] | None = None,
                     open_keys: bool = False) -> str:
    if fill_keys:
        return _fill_examples(fill_keys, open_keys).format(query=query)
    return (EXPAND_EXAMPLES if expand else TAG_USER).format(query=query)


def metadata_text(values: dict[str, str], keys: Sequence[str], max_chars: int = 80,
                  max_items: int | None = None) -> str:
    """`Metadata: key=value; ...` in the format the product text uses, keys in the order of `keys`.

    max_items None: keys outside `keys` and empty values are dropped, so the LLM cannot add fields the
    products do not have. max_items N: the preferred `keys` first, then the LLM's other keys in its order,
    at most N in all."""
    have = {str(k).strip().lower().replace(" ", "_"): " ".join(str(v).split())[:max_chars] for k, v in values.items()}
    order = list(keys)
    if max_items is not None:
        order += [k for k in have if k not in set(keys)]
    items = [f"{k}={have[k]}" for k in order if have.get(k)]
    if max_items is not None:
        items = items[:max_items]
    return "Metadata: " + "; ".join(items) if items else ""


def filled_count(expansion: str) -> int:
    """How many keys a `Metadata: k=v; k=v` string fills (0 for an empty or non-metadata string)."""
    if not expansion.startswith("Metadata: "):
        return 0
    return sum(1 for item in expansion[len("Metadata: "):].split("; ") if "=" in item)


class _TagReply(BaseModel):
    semantic_text: str
    tags: list[Tag] = Field(default_factory=list)
    expansion: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


def _to_parsed(reply: _TagReply, query: str, max_tags: int, expand: bool = False,
               fill_keys: Sequence[str] | None = None, fill_max: int | None = None) -> ParsedQuery:
    seen: set[str] = set()
    tags: list[Tag] = []
    for t in reply.tags:
        key = t.text.lower()
        if t.text and key not in seen:
            seen.add(key)
            tags.append(t)
    return ParsedQuery(
        semantic_text=reply.semantic_text.strip() or query, tags=tags[:max_tags],
        expansion=metadata_text(reply.metadata, fill_keys, max_items=fill_max) if fill_keys
        else " ".join(reply.expansion.split()) if expand else "",
    )


class TagParser:
    """Drop-in replacement for QueryParser that returns generated tags instead of fixed filters."""

    def __init__(self, llm: LLM, retries: int = 2, max_tags: int = 6, expand: bool = False,
                 fill_keys: Sequence[str] | None = None, fill_max: int | None = None) -> None:
        self.llm = llm
        self.fill_keys = list(fill_keys) if fill_keys else None  # fill in these metadata keys (the `_fill` methods)
        # None: only fill_keys; N: fill_keys preferred, other keys allowed, at most N in all
        self.fill_max = fill_max if self.fill_keys and fill_max else None
        self.expand = expand and not self.fill_keys  # also ask for a query expansion (query rewriting)
        if self.fill_keys and self.fill_max:
            fill = FILL_OPEN_SYSTEM.replace("{keys}", ", ".join(self.fill_keys)).replace("{max}", str(self.fill_max))
        elif self.fill_keys:
            fill = FILL_SYSTEM.replace("{keys}", ", ".join(self.fill_keys))
        else:
            fill = EXPAND_SYSTEM if self.expand else ""
        self.system = TAG_SYSTEM + fill
        self.retries = retries
        self.max_tags = max_tags
        self.n_failures = 0  # queries that fell back to "no tags"

    def parse(self, query: str) -> ParsedQuery:
        try:
            reply = complete_json(
                self.llm, build_tag_prompt(query, self.expand, self.fill_keys, bool(self.fill_max)), _TagReply, system=self.system,
                retries=self.retries,
            )
            return _to_parsed(reply, query, self.max_tags, self.expand, self.fill_keys, self.fill_max)
        except LLMError as e:
            log.warning("Tag generation failed for %r (%s); using no tags", query, e)
            self.n_failures += 1
            return ParsedQuery(semantic_text=query)

    def parse_many(self, queries: list[str]) -> list[ParsedQuery]:
        schema = _TagReply.model_json_schema()
        replies = self.llm.complete_many(
            [build_tag_prompt(q, self.expand, self.fill_keys, bool(self.fill_max)) for q in queries], system=self.system, schema=schema
        )
        out: list[ParsedQuery] = []
        for q, raw in zip(queries, replies):
            try:
                out.append(_to_parsed(parse_reply(raw, _TagReply), q, self.max_tags, self.expand, self.fill_keys, self.fill_max))
            except ValueError:
                out.append(self.parse(q))
        return out

"""Document-side metadata: an LLM reads each product once and writes its attributes as tags.

The tags use the same roles as the query tags (query/schema.TAG_KINDS) with free-form values, e.g.
`brand=Under Armour`, `color=blue`, `size=60 ml`. They are stored in a JSONL file next to the processed
products and rendered into the product text by representation.TaggedMetadata, so searching needs no LLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from .data.schema import Product
from .llm import LLM, LLMError, complete_json, parse_reply
from .query.schema import TAG_KINDS, Tag
from .representation.base import truncate

log = logging.getLogger(__name__)

PRODUCT_TAG_SYSTEM = """You read one product listing and write its attributes as metadata tags.

Return a JSON object {"tags": [...]}. Each tag is {"kind": ..., "text": ...} where "kind" is the \
role of the attribute, exactly one of: {kinds}, and "text" is its value: a short phrase (1-4 words) \
copied or lightly normalised from the listing.

Rules:
- Only attributes the listing states or clearly implies. Never guess or invent; leave an attribute out \
when the listing does not say.
- Always include one "product_type" tag naming what the product is (e.g. "phone case", "nail polish").
- Keep numbers and units in the value ("60 ml", "18 inches", "10 pack").
- No marketing words ("best", "premium", "amazing") and no repeated tags.
- At most {max_tags} tags, the most useful for finding this product first."""

EXAMPLES = """Example:

Title: Detroit Axle - 4WD Front Wheel Hub and Bearing Assembly for 2005-2010 Ford F-150, 8-Lug
Description: Replaces OEM part 515078. Fits Ford F-150 4x4 models. Includes ABS sensor wire. One-year warranty.
{{"tags": [{{"kind": "product_type", "text": "wheel hub bearing assembly"}}, {{"kind": "brand", "text": "Detroit Axle"}}, {{"kind": "audience", "text": "2005-2010 Ford F-150 4WD"}}, {{"kind": "feature", "text": "8-lug"}}, {{"kind": "feature", "text": "ABS sensor wire"}}]}}

Now the real product.

Title: {title}
Description: {description}"""


class _Reply(BaseModel):
    tags: list[Tag] = Field(default_factory=list)


def _clean(reply: _Reply, max_tags: int) -> list[Tag]:
    seen: set[tuple[str, str]] = set()
    out: list[Tag] = []
    for t in reply.tags:
        key = (t.kind, t.text.lower())
        if t.text and key not in seen:
            seen.add(key)
            out.append(t)
    return out[:max_tags]


class ProductTagger:
    def __init__(self, llm: LLM, max_tags: int = 8, max_chars: int = 1500, retries: int = 1) -> None:
        self.llm, self.max_tags, self.max_chars, self.retries = llm, max_tags, max_chars, retries
        self.system = PRODUCT_TAG_SYSTEM.replace("{kinds}", ", ".join(TAG_KINDS)).replace("{max_tags}", str(max_tags))
        self.n_failures = 0  # products that ended with no tags because the reply was unusable

    def prompt(self, p: Product) -> str:
        return EXAMPLES.format(title=" ".join(p.title.split()), description=truncate(" ".join(p.description.split()), self.max_chars))

    def tag(self, p: Product) -> list[Tag]:
        try:
            return _clean(complete_json(self.llm, self.prompt(p), _Reply, system=self.system, retries=self.retries), self.max_tags)
        except LLMError as e:
            log.warning("Tagging failed for product %s (%s); no tags", p.id, e)
            self.n_failures += 1
            return []

    def tag_many(self, products: Sequence[Product]) -> list[list[Tag]]:
        """Parallel first pass; only unusable replies go through the slower retry path."""
        schema = _Reply.model_json_schema()
        replies = self.llm.complete_many([self.prompt(p) for p in products], system=self.system, schema=schema)
        out: list[list[Tag]] = []
        for p, raw in zip(products, replies):
            try:
                out.append(_clean(parse_reply(raw, _Reply), self.max_tags))
            except ValueError:
                out.append(self.tag(p))
        return out


def load_product_tags(path: str | Path) -> dict[str, list[Tag]]:
    """{product id: tags} from a JSONL file; an id seen twice keeps its last line."""
    out: dict[str, list[Tag]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[str(rec["id"])] = [Tag(**t) for t in rec["tags"]]
    return out


def append_product_tags(path: str | Path, ids: Iterable[str], tags: Iterable[Sequence[Tag]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for pid, ts in zip(ids, tags):
            f.write(json.dumps({"id": pid, "tags": [t.model_dump() for t in ts]}, ensure_ascii=False) + "\n")


def tag_catalog(
    products: Sequence[Product],
    tagger: ProductTagger,
    path: str | Path,
    *,
    limit: int | None = None,
    batch: int = 200,
    seed: int = 13,
    log=print,
) -> dict[str, int]:
    """Tag the products that have no tags in `path` yet, `batch` at a time, appending after every batch so an
    interrupted run resumes where it stopped. `limit` caps how many products this call tags (a random
    sample, same order every time, so a speed test and the full run use the same sequence)."""
    import random
    import time

    p = Path(path)
    done = set(load_product_tags(p)) if p.exists() else set()
    todo = [x for x in products if x.id not in done]
    random.Random(seed).shuffle(todo)
    if limit is not None:
        todo = todo[:limit]
    log(f"{len(done)} products already tagged, {len(todo)} to tag now")
    t0, n = time.time(), 0
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        append_product_tags(p, [x.id for x in chunk], tagger.tag_many(chunk))
        n += len(chunk)
        rate = n / max(time.time() - t0, 1e-9)
        log(f"  tagged {n}/{len(todo)}   {rate:.1f} products/s   about {(len(todo) - n) / rate / 60:.0f} min left")
    return {"already": len(done), "tagged_now": n, "failures": tagger.n_failures}

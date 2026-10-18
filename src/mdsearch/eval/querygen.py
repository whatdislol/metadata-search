"""Synthetic evaluation queries with *known* constraints.

Unlike the paper (GPT-4o writes queries, then 200 are picked by hand), the constraints here are
sampled from a real product's metadata first, so the ground-truth filters exist independently of
the query parser being evaluated. The LLM only phrases the query; we then check programmatically
that the phrasing actually contains each constraint.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from ..data.schema import Product
from ..llm import LLM, parse_reply
from ..query.schema import Filters


class GeneratedQuery(BaseModel):
    id: int
    text: str
    filters: Filters  # ground-truth constraints
    source_id: str  # the product the constraints were sampled from


class _GenOut(BaseModel):
    query: str


class _BroadOut(BaseModel):
    query: str
    names_used: list[str] = []  # every brand/franchise/character/manufacturer name in the query


GEN_SYSTEM = (
    "You write realistic shopping search queries, as a person would type them to a toy store "
    "search box. Reply with JSON: {\"query\": \"...\"}."
)

GEN_PROMPT = """Write ONE search query for a shopper who wants a product like the one below.

Product: {title}
Details: {details}

The query MUST explicitly state these requirements, using these exact values:
{requirements}

It must also express one preference or use case taken from the product details (for example
who it is for, or what it is good for). Do not mention the product title. Keep it to one or two
sentences, natural wording."""


BROAD_SYSTEM = (
    "You write realistic shopping search queries, as a person would type them to a toy store "
    "search box. The shopper has a need, not a specific item in mind. "
    "Reply with JSON: {\"query\": \"...\", \"names_used\": [\"...\"]}."
)

BROAD_PROMPT = """Write ONE search query for a shopper who wants a KIND of product like the one below
but has not chosen a specific item. Describe the type of product and what it is for, in general
terms, so that many different products in the store could satisfy the query.

Product (for reference only): {title}
Details: {details}

The query MUST state these requirements, using these exact values:
{requirements}

Rules:
- Name the general product type (for example "building set", "board game", "plush toy") and one
  general use, quality or occasion.
- Do NOT mention the product's title, model or set numbers, character or franchise names (the
  required brand is the only name allowed), colours, sizes, piece counts, or any other detail that
  would identify this exact product.
- Write naturally, one or two sentences. Do not use labels such as "brand:" or "rating:".
- In "names_used" list EVERY brand, franchise, character, manufacturer or product-line name that
  appears in your query (include the required brand if you used it). Use [] if there are none."""

STYLES = ("specific", "broad")


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def leaks_title(text: str, p: Product, max_overlap: float = 0.5) -> bool:
    """True if the query repeats most of the product title (so it would identify the product)."""
    title = _words(p.title) - _words(p.brand or "")
    if len(title) < 3:
        return False
    return len(title & _words(text)) / len(title) > max_overlap


def nice_ceiling(x: float) -> float:
    step = 5 if x < 50 else 10 if x < 200 else 50
    return float(math.ceil(x / step) * step)


def sample_filters(
    p: Product, rng: random.Random, max_age: float | None = None
) -> Filters | None:
    """Sample 1-3 constraints that `p` itself satisfies. None if p has no usable metadata.

    `max_age`: only constrain age for products whose minimum age is at most this (skips adult
    items, where "suitable for a child aged 5" would make an incoherent query).
    """
    options: list[str] = []
    if p.brand:
        options.append("brand")
    if p.price is not None and p.price > 0:
        options.append("price")
    if p.rating is not None and p.rating >= 1:
        options.append("rating")
    if p.min_age is not None and (max_age is None or p.min_age <= max_age):
        options.append("age")
    if not options:
        return None
    chosen = rng.sample(options, k=rng.randint(1, min(3, len(options))))
    f = Filters()
    if "brand" in chosen:
        f.brand = p.brand
    if "price" in chosen:
        f.price_max = nice_ceiling(p.price * 1.15)  # type: ignore[operator]
    if "rating" in chosen:
        f.rating_min = float(math.floor(p.rating * 2) / 2)  # type: ignore[operator]
    if "age" in chosen:
        # round UP so the sampled age is never below the product's own minimum age
        f.age_min = f.age_max = float(math.ceil(p.min_age) + rng.randint(0, 2))  # type: ignore[arg-type]
    return f


def _num(v: float) -> str:
    return f"{v:g}"


def describe_requirements(f: Filters, currency: str) -> str:
    lines: list[str] = []
    if f.brand:
        lines.append(f'- brand: "{f.brand}" (write the brand name exactly like this)')
    if f.price_max is not None:
        lines.append(f"- price: at most {currency}{_num(f.price_max)}")
    if f.rating_min is not None:
        lines.append(f"- rating: at least {_num(f.rating_min)} stars")
    if f.age_min is not None:
        lines.append(f"- age: suitable for a child aged {_num(f.age_min)}")
    return "\n".join(lines)


def unrecorded_names(names: Sequence[str], f: Filters) -> list[str]:
    """Names the query uses that are not the recorded brand constraint."""

    def norm(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", x.lower())

    nb = norm(f.brand or "")
    out = []
    for n in names:
        nn = norm(n)
        if nn and not (nb and (nn in nb or nb in nn)):
            out.append(n)
    return out


def mentions_unrecorded_brand(text: str, p: Product, f: Filters) -> bool:
    """True if the query names the product's brand although the brand is NOT one of the recorded
    constraints. Such a query asks for a brand the ground truth (and the judge) does not check."""
    if f.brand or not p.brand:
        return False
    return bool(_words(p.brand) & _words(text))


def mentions_constraints(text: str, f: Filters) -> bool:
    """Programmatic check that the phrasing really contains every constraint."""
    t = text.lower()
    if f.brand and f.brand.lower() not in t:
        return False
    for v in (f.price_max, f.rating_min, f.age_min):
        if v is not None and _num(v) not in text:
            return False
    return True


def _details(p: Product) -> str:
    desc = (p.description or " ".join(p.reviews[:2]))[:500]
    return desc or "(no description)"


def generate_queries(
    llm: LLM,
    products: Sequence[Product],
    n: int,
    *,
    seed: int = 13,
    currency: str = "£",
    max_rounds: int = 6,
    style: str = "specific",
) -> list[GeneratedQuery]:
    """style="specific": the query may describe the source product in detail (known-item search).
    style="broad": the query describes a product type and need; many products can match, and
    queries that repeat the product title are dropped.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    broad = style == "broad"
    system = BROAD_SYSTEM if broad else GEN_SYSTEM
    template = BROAD_PROMPT if broad else GEN_PROMPT
    rng = random.Random(seed)
    pool = list(products)
    rng.shuffle(pool)
    out: list[GeneratedQuery] = []
    cursor = 0
    out_model = _BroadOut if broad else _GenOut
    schema = out_model.model_json_schema()

    for _ in range(max_rounds):
        if len(out) >= n or cursor >= len(pool):
            break
        batch: list[tuple[Product, Filters]] = []
        while len(batch) < (n - len(out)) * 2 and cursor < len(pool):
            p = pool[cursor]
            cursor += 1
            f = sample_filters(p, rng, max_age=12.0 if broad else None)
            if f is not None:
                batch.append((p, f))
        prompts = [
            template.format(
                title=p.title, details=_details(p), requirements=describe_requirements(f, currency)
            )
            for p, f in batch
        ]
        replies = llm.complete_many(prompts, system=system, schema=schema)
        for (p, f), reply in zip(batch, replies):
            if len(out) >= n:
                break
            try:
                parsed_reply = parse_reply(reply, out_model)
            except ValueError:
                continue
            text = parsed_reply.query.strip()
            if broad and (
                leaks_title(text, p)
                or mentions_unrecorded_brand(text, p, f)
                or unrecorded_names(parsed_reply.names_used, f)  # type: ignore[attr-defined]
            ):
                continue
            if text and mentions_constraints(text, f):
                out.append(GeneratedQuery(id=len(out), text=text, filters=f, source_id=p.id))
    return out


def save_queries(path: str | Path, queries: Sequence[GeneratedQuery]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(q.model_dump_json() + "\n")


def load_queries(path: str | Path) -> list[GeneratedQuery]:
    with open(path, encoding="utf-8") as f:
        return [GeneratedQuery(**json.loads(line)) for line in f if line.strip()]
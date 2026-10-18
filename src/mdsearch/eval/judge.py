"""Relevance judging for generated queries.

Two parts, so the LLM never has to be trusted with arithmetic:
  1. hard constraints (brand/price/rating/age) are checked in code against the ground-truth
     filters the query was generated from, with the paper's 20% slack on price and rating;
  2. only products that pass part 1 are shown to the LLM, which judges whether the product also
     fits the query's remaining, subjective intent.

Caveat: if the same LLM parses queries (QAM) and judges results, errors can correlate. Spot-check
a sample of judgments by hand before trusting small differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel

from ..catalog import Catalog
from ..filtering.constraints import satisfies
from ..llm import LLM, complete_json, parse_reply
from .querygen import GeneratedQuery


class _JudgeOut(BaseModel):
    relevant: bool
    reason: str = ""


JUDGE_SYSTEM = (
    "You judge whether a product matches a shopper's search query. Reply with JSON: "
    '{"relevant": true|false, "reason": "<one short sentence>"}.'
)

JUDGE_PROMPT = """Query: {query}

Product title: {title}
Product description: {description}
Some customer reviews: {reviews}

Brand, price, rating and age requirements in the query are checked separately and already
satisfied. Judge ONLY the rest of the query: does this product fit what the shopper is looking
for (type of product, use case, who it is for, qualities wanted)?"""


@dataclass
class Judgment:
    relevant: bool
    hard_ok: bool
    soft_ok: bool | None  # None when the LLM was not consulted or its reply was unusable
    reason: str = ""


class RelevanceJudge:
    def __init__(
        self, llm: LLM, catalog: Catalog, price_tol: float = 0.2, rating_tol: float = 0.2
    ) -> None:
        self.llm = llm
        self.catalog = catalog
        self.price_tol = price_tol
        self.rating_tol = rating_tol

    def _prompt(self, q: GeneratedQuery, product_id: str) -> str:
        p = self.catalog.get(product_id)
        return JUDGE_PROMPT.format(
            query=q.text,
            title=p.title,
            description=(p.description or "(none)")[:600],
            reviews=" || ".join(r[:200] for r in p.reviews[:3]) or "(none)",
        )

    def judge_many(self, pairs: Sequence[tuple[GeneratedQuery, str]]) -> list[Judgment]:
        out: list[Judgment | None] = [None] * len(pairs)
        to_ask: list[int] = []
        for i, (q, pid) in enumerate(pairs):
            hard = satisfies(
                self.catalog.get(pid), q.filters,
                price_tol=self.price_tol, rating_tol=self.rating_tol,
            )
            if hard:
                to_ask.append(i)
            else:
                out[i] = Judgment(False, False, None, "violates a hard constraint")

        schema = _JudgeOut.model_json_schema()
        replies = self.llm.complete_many(
            [self._prompt(*pairs[i]) for i in to_ask], system=JUDGE_SYSTEM, schema=schema
        )
        for i, reply in zip(to_ask, replies):
            try:
                r = parse_reply(reply, _JudgeOut)
            except ValueError:
                try:
                    r = complete_json(
                        self.llm, self._prompt(*pairs[i]), _JudgeOut, system=JUDGE_SYSTEM
                    )
                except Exception:  # unusable judge reply counts as not relevant, flagged
                    out[i] = Judgment(False, True, None, "judge reply unusable")
                    continue
            out[i] = Judgment(r.relevant, True, r.relevant, r.reason)
        return [j for j in out if j is not None]

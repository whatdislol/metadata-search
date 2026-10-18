"""Scalar definition of what it means for a product to satisfy a set of Filters."""

from __future__ import annotations

import re

from ..data.schema import Product
from ..query.schema import Filters


def normalize_brand(s: str | None) -> str:
    return "" if not s else re.sub(r"[^a-z0-9]", "", s.lower())


def brand_matches(query_brand: str | None, product_brand: str | None) -> bool:
    qb, pb = normalize_brand(query_brand), normalize_brand(product_brand)
    if not qb or not pb:
        return False
    return qb in pb or pb in qb


def age_upper(f: Filters) -> float | None:
    """A product fits if its minimum age is <= the oldest age in the query range."""
    return f.age_max if f.age_max is not None else f.age_min


def satisfies(
    p: Product,
    f: Filters,
    *,
    price_tol: float = 0.0,
    rating_tol: float = 0.0,
    missing: str = "exclude",
) -> bool:
    """True if `p` meets every active constraint.

    `*_tol` loosen numeric bounds by a fraction (the paper's judge allows 20%).
    `missing`: what to do when the product lacks a constrained field ("exclude" or "keep").
    """

    def check(value: float | str | None, ok: bool) -> bool:
        if value is None:
            return missing == "keep"
        return ok

    if f.brand is not None and not check(p.brand, brand_matches(f.brand, p.brand)):
        return False
    if f.price_min is not None and not check(
        p.price, p.price is not None and p.price >= f.price_min * (1 - price_tol)
    ):
        return False
    if f.price_max is not None and not check(
        p.price, p.price is not None and p.price <= f.price_max * (1 + price_tol)
    ):
        return False
    if f.rating_min is not None and not check(
        p.rating, p.rating is not None and p.rating >= f.rating_min * (1 - rating_tol)
    ):
        return False
    hi = age_upper(f)
    if hi is not None and not check(p.min_age, p.min_age is not None and p.min_age <= hi):
        return False
    return True

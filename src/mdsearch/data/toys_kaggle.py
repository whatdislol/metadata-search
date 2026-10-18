"""Loader for the Kaggle "Toy Products on Amazon" sample (PromptCloud).

IMPORTANT: this was written from the dataset's published field list, not from the file itself.
Column names are resolved flexibly and the text parsers below (price, age, reviews) are
heuristics. Run scripts/01_prepare_toys.py and check the printed coverage before trusting them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import Product

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_price(s: Any) -> float | None:
    """'£27.49' -> 27.49; for ranges ('£5.99 - £10.99') the first price is used."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = _NUM.search(str(s))
    return float(m.group().replace(",", "")) if m else None


def parse_int(s: Any) -> int | None:
    v = parse_price(s)
    return int(v) if v is not None else None


def parse_rating(s: Any) -> float | None:
    """'4.9 out of 5 stars' -> 4.9"""
    v = parse_price(s)
    return v if v is not None and 0 <= v <= 5 else None


_N = r"(\d+(?:\.\d+)?)"
_UNIT = r"(months?|mths?|years?|yrs?)"
# "6 - 12 years", "6 to 12 months": the unit at the end applies to both numbers. The FIRST number is
# the minimum age. (Amazon also has junk like "8 - 3 years"; the description confirms 8 there.)
_RANGE = re.compile(_N + r"\s*(?:-|\u2013|to)\s*" + _N + r"\s*\+?\s*" + _UNIT, re.I)
# "3 years", "18 months+", "8+ yrs"
_SINGLE = re.compile(_N + r"\s*\+?\s*" + _UNIT, re.I)
# "Age: 5" -- only when the number is NOT followed by a unit (else "age:6 months" would read as 6 years)
_LABELLED = re.compile(
    r"\bages?\s*:?\s*" + _N + r"(?!\s*\+?\s*(?:-|\u2013|to)?\s*(?:\d+(?:\.\d+)?\s*)?" + _UNIT + r")",
    re.I,
)


def _to_years(value: float, unit: str) -> float:
    return value / 12 if unit.lower().startswith(("mo", "mt")) else value


def parse_min_age(*texts: Any) -> float | None:
    """Heuristic: minimum age (years) from the first age mention in the first text that has one.

    '3 years and up' -> 3; '6 - 12 years' -> 6 (first number); '18 months+' -> 1.5; 'Age: 5' -> 5.
    Within a text the EARLIEST mention wins, whatever its unit.
    """
    for text in texts:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        s = str(text)
        found: list[tuple[int, float]] = []  # (position, years)
        for m in _RANGE.finditer(s):
            found.append((m.start(), _to_years(float(m.group(1)), m.group(3))))  # first number, always
        for m in _SINGLE.finditer(s):
            found.append((m.start(), _to_years(float(m.group(1)), m.group(2))))
        for m in _LABELLED.finditer(s):
            found.append((m.start(), float(m.group(1))))
        if found:
            return round(min(found)[1], 2)
    return None


def parse_reviews(s: Any) -> list[str]:
    """Best-effort split of the `customer_reviews` string.

    Assumed format: reviews joined by '|', each 'title // rating // date // author // text'.
    Chunks without '//' are treated as a continuation of the previous review.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    reviews: list[str] = []
    for chunk in str(s).split("|"):
        if "//" in chunk:
            parts = [p.strip() for p in chunk.split("//")]
            body = parts[-1] if len(parts) >= 5 else " ".join(parts[1:])
            text = f"{parts[0]}. {body}".strip(". ").strip()
            if text:
                reviews.append(text)
        elif reviews and chunk.strip():
            reviews[-1] += " " + chunk.strip()
    return reviews


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _clean(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def load_toys_csv(path: str | Path, limit: int | None = None) -> list[Product]:
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)

    title = _col(df, "product_name", "title")
    if title is None:
        raise ValueError(f"No product_name/title column in {path}; columns: {list(df.columns)}")
    ids = _col(df, "uniq_id", "id")
    brand = _col(df, "manufacturer", "brand")
    price = _col(df, "price")
    rating = _col(df, "average_review_rating", "rating")
    n_reviews = _col(df, "number_of_reviews")
    category = _col(df, "amazon_category_and_sub_category", "category")
    desc = _col(df, "description")
    desc2 = _col(df, "product_description")
    info = _col(df, "product_information")
    reviews = _col(df, "customer_reviews")

    products: list[Product] = []
    seen: set[str] = set()
    for i in range(len(df)):
        t = _clean(title.iloc[i])
        if not t:
            continue
        pid = _clean(ids.iloc[i]) if ids is not None else None
        pid = pid or f"toy-{i}"
        if pid in seen:
            continue
        seen.add(pid)

        d = _clean(desc.iloc[i]) if desc is not None else None
        d = d or (_clean(desc2.iloc[i]) if desc2 is not None else None) or ""
        info_text = info.iloc[i] if info is not None else None
        products.append(
            Product(
                id=pid,
                title=t,
                description=d,
                brand=_clean(brand.iloc[i]) if brand is not None else None,
                price=parse_price(price.iloc[i]) if price is not None else None,
                rating=parse_rating(rating.iloc[i]) if rating is not None else None,
                n_reviews=parse_int(n_reviews.iloc[i]) if n_reviews is not None else None,
                min_age=parse_min_age(info_text, d, t),
                category=_clean(category.iloc[i]) if category is not None else None,
                reviews=parse_reviews(reviews.iloc[i]) if reviews is not None else [],
            )
        )
    return products


def describe_coverage(products: list[Product]) -> dict[str, float]:
    """Fraction of products with each metadata field present."""
    n = max(len(products), 1)
    return {
        "n_products": float(len(products)),
        "brand": sum(p.brand is not None for p in products) / n,
        "price": sum(p.price is not None for p in products) / n,
        "rating": sum(p.rating is not None for p in products) / n,
        "min_age": sum(p.min_age is not None for p in products) / n,
        "has_reviews": sum(bool(p.reviews) for p in products) / n,
        "avg_reviews_per_product": sum(len(p.reviews) for p in products) / n,
    }
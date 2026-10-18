from __future__ import annotations

import random

import numpy as np

from mdsearch.catalog import Catalog
from mdsearch.data.schema import Product
from mdsearch.filtering import MetadataFilter, brand_matches, satisfies
from mdsearch.query.schema import Filters


def test_brand_matching_is_punctuation_and_case_insensitive():
    assert brand_matches("fisher price", "Fisher-Price")
    assert brand_matches("LEGO", "LEGO Group")
    assert not brand_matches("LEGO", "Hasbro")
    assert not brand_matches("LEGO", None) and not brand_matches(None, "LEGO")


def test_filter_masks_and_report(catalog):
    mf = MetadataFilter(catalog)
    mask, rep = mf.apply(Filters(brand="LEGO", price_max=50))
    assert [catalog.ids[i] for i in np.flatnonzero(mask)] == ["a"]
    assert rep.total == 5 and rep.remaining == 1
    assert [(s.name, s.removed) for s in rep.steps] == [("brand", 3), ("price_max", 1)]


def test_age_uses_oldest_age_in_range(catalog):
    mf = MetadataFilter(catalog)
    mask, _ = mf.apply(Filters(age_min=5, age_max=8))
    assert sorted(catalog.ids[i] for i in np.flatnonzero(mask)) == ["a", "c", "d"]
    mask, _ = mf.apply(Filters(age_min=3))  # single age: min_age <= 3
    assert sorted(catalog.ids[i] for i in np.flatnonzero(mask)) == ["d"]


def test_missing_metadata_exclude_vs_keep(catalog):
    f = Filters(price_max=100)
    m_ex, _ = MetadataFilter(catalog, missing="exclude").apply(f)
    m_keep, _ = MetadataFilter(catalog, missing="keep").apply(f)
    assert not m_ex[catalog.index("e")] and m_keep[catalog.index("e")]


def test_empty_filters_keep_everything(catalog):
    mask, rep = MetadataFilter(catalog).apply(Filters())
    assert mask.all() and rep.steps == []


def test_tolerance_loosens_numeric_bounds(catalog):
    f = Filters(price_max=25)  # product c costs 19.5, a costs 29.99
    assert not satisfies(catalog.get("a"), f)
    assert satisfies(catalog.get("a"), f, price_tol=0.2)  # 25 * 1.2 = 30
    m, _ = MetadataFilter(catalog, price_tol=0.2).apply(f)
    assert catalog.ids[0] == "a" and m[0]


def test_vectorised_filter_agrees_with_scalar_satisfies():
    rng = random.Random(0)
    prods = [
        Product(
            str(i), f"t{i}",
            brand=rng.choice([None, "LEGO", "Hasbro", "Fisher-Price"]),
            price=rng.choice([None, round(rng.uniform(1, 100), 2)]),
            rating=rng.choice([None, round(rng.uniform(1, 5), 1)]),
            min_age=rng.choice([None, float(rng.randint(0, 12))]),
        )
        for i in range(300)
    ]
    cat = Catalog(prods)
    for missing in ("exclude", "keep"):
        for tol in (0.0, 0.2):
            mf = MetadataFilter(cat, missing=missing, price_tol=tol, rating_tol=tol)
            for _ in range(40):
                f = Filters(
                    brand=rng.choice([None, "lego", "has"]),
                    price_max=rng.choice([None, 30.0, 60.0]),
                    price_min=rng.choice([None, 10.0]),
                    rating_min=rng.choice([None, 3.5]),
                    age_min=rng.choice([None, 4.0]),
                    age_max=rng.choice([None, 6.0]),
                )
                mask, _ = mf.apply(f)
                expected = [
                    satisfies(p, f, price_tol=tol, rating_tol=tol, missing=missing) for p in prods
                ]
                assert mask.tolist() == expected

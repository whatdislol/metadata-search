from __future__ import annotations

import pandas as pd

from mdsearch.data.schema import Product, load_products, save_products
from mdsearch.data.toys_kaggle import (
    describe_coverage,
    load_toys_csv,
    parse_int,
    parse_min_age,
    parse_price,
    parse_rating,
    parse_reviews,
)


def test_parse_price_int_rating():
    assert parse_price("£27.49") == 27.49
    assert parse_price("£1,234.50") == 1234.5
    assert parse_price("£5.99 - £10.99") == 5.99
    assert parse_price(None) is None and parse_price(float("nan")) is None
    assert parse_price("free") is None
    assert parse_int("1,234") == 1234
    assert parse_rating("4.9 out of 5 stars") == 4.9
    assert parse_rating("12") is None


def test_parse_min_age():
    assert parse_min_age("Manufacturer recommended age: 3 years and up") == 3
    assert parse_min_age("suitable from 18 months") == 1.5
    assert parse_min_age("Age: 5") == 5
    assert parse_min_age(None, "nothing here") is None
    assert parse_min_age(None, "8+ years") == 8  # later texts are used when earlier ones are empty


def test_parse_reviews():
    raw = "Great fun // 5.0 // 18 August 2014 // Jay // My kids love it | Meh // 2.0 // 1 May 2015 // Al // Broke fast | still ok"
    assert parse_reviews(raw) == ["Great fun. My kids love it", "Meh. Broke fast still ok"]
    assert parse_reviews(None) == []


def test_load_csv_and_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "uniq_id": ["1", "2", "2", "3"],
            "product_name": ["Car", "Doll", "Doll dup", None],
            "manufacturer": ["Hot Wheels", None, "X", "Y"],
            "price": ["£4.99", "£10.00", "£1", "£2"],
            "number_of_reviews": ["1,200", "3", "1", "1"],
            "average_review_rating": ["4.5 out of 5 stars", None, None, None],
            "description": ["Fast car", None, None, None],
            "product_information": ["Manufacturer recommended age: 3 years and up", None, None, None],
            "customer_reviews": ["Nice // 5.0 // d // a // Loved it", None, None, None],
        }
    )
    csv = tmp_path / "toys.csv"
    df.to_csv(csv, index=False)
    products = load_toys_csv(csv)
    assert [p.id for p in products] == ["1", "2"]  # duplicate id and untitled row dropped
    car = products[0]
    assert (car.brand, car.price, car.rating, car.n_reviews, car.min_age) == ("Hot Wheels", 4.99, 4.5, 1200, 3)
    assert car.reviews == ["Nice. Loved it"]
    assert products[1].brand is None

    out = tmp_path / "p.jsonl"
    assert save_products(out, products) == 2
    assert load_products(out) == products
    cov = describe_coverage(products)
    assert cov["price"] == 1.0 and cov["brand"] == 0.5


def test_missing_title_column_is_an_error(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(csv, index=False)
    try:
        load_toys_csv(csv)
    except ValueError as e:
        assert "product_name" in str(e)
    else:
        raise AssertionError("expected ValueError")

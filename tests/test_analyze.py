from __future__ import annotations

import importlib.util
from pathlib import Path

from mdsearch.eval.querygen import GeneratedQuery
from mdsearch.query.schema import Filters

_spec = importlib.util.spec_from_file_location(
    "analyze_script", Path(__file__).resolve().parents[1] / "scripts" / "07_analyze.py"
)
analyze_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_script)


def test_filter_diff():
    fd = analyze_script.filter_diff
    truth = {"brand": "LEGO", "price_max": 30.0, "age_min": 5.0, "age_max": 5.0}
    assert fd(truth, {"brand": "Lego", "price_max": 30.0, "age_min": 5.0, "age_max": 5.0}) == ([], [])
    # age compared by its upper end, so min-only and min=max parses agree
    assert fd(truth, {"brand": "LEGO", "price_max": 30.0, "age_min": 5.0}) == ([], [])
    assert fd(truth, {"brand": "LEGO", "price_max": 25.0, "age_min": 5.0}) == (["price_max"], [])
    assert fd({"price_max": 10.0}, {"price_max": 10.0, "brand": "Gund", "rating_min": 4.0}) == ([], ["brand", "rating_min"])
    assert fd({"brand": "LEGO"}, {}) == (["brand"], [])


def _pq(method, qid, hits, rels, parsed=None):
    return {"method": method, "query_id": qid, "hits": hits, "rels": rels, "parsed": parsed}


def test_analyze_reports_source_hit_and_parser_diagnosis():
    queries = [
        GeneratedQuery(id=0, text="q0", filters=Filters(brand="LEGO"), source_id="s0"),
        GeneratedQuery(id=1, text="q1", filters=Filters(price_max=10), source_id="s1"),
    ]
    good = {"semantic_text": "x", "filters": {"brand": "LEGO"}}
    bad = {"semantic_text": "x", "filters": {"price_max": 99.0, "brand": "Acme"}}
    per_query = [
        _pq("qam", 0, ["s0", "o1"], [1, 0], good),
        _pq("qam", 1, ["o2", "o3"], [0, 0], bad),
        _pq("kw", 0, ["o4", "s0"], [0, 1]),
        _pq("kw", 1, ["s1", "o5"], [1, 0]),
    ]
    judgments = [
        {"query_id": q, "product_id": h, "relevant": bool(r), "hard_ok": True, "soft_ok": bool(r), "reason": ""}
        for pq in per_query for q, h, r in [(pq["query_id"], h, r) for h, r in zip(pq["hits"], pq["rels"])]
    ]
    text = analyze_script.analyze(per_query, judgments, 2, "qam", queries)
    assert "source hit" in text and "did the parser recover" in text
    assert "exactly the ground-truth constraints: 50.0%" in text
    assert "misses the source product in its top-10 for 1 queries: 1 had a wrong/extra filter" in text
    assert "prec|hard ok" in text


def test_explain_misses_sorts_causes(products):
    by_id = {p.id: p for p in products}
    queries = [
        # source "a" (LEGO, 29.99): the parser invented a Hasbro brand -> excluded by the filter
        GeneratedQuery(id=0, text="q0", filters=Filters(price_max=40), source_id="a"),
        # source "c": filters fine, pool small, but source not in the top hits -> ranked low
        GeneratedQuery(id=1, text="q1", filters=Filters(brand="Hasbro"), source_id="c"),
    ]
    per_query = [
        {"method": "qam", "query_id": 0, "hits": ["b"], "rels": [0], "n_candidates": 3,
         "parsed": {"semantic_text": "x", "filters": {"price_max": 40.0, "brand": "Hasbro"}}},
        {"method": "qam", "query_id": 1, "hits": ["e"], "rels": [0], "n_candidates": 2,
         "parsed": {"semantic_text": "x", "filters": {"brand": "Hasbro"}}},
    ]
    text = analyze_script.explain_misses(per_query, queries, by_id, "qam", show=5)
    assert "2 queries" in text
    assert "excluded: 1" in text and "ranked low (pool<=100): 1" in text
    assert "extra=['brand']" in text


def test_explain_parse_errors_lists_only_missed_constraints():
    queries = [
        GeneratedQuery(id=0, text="a LEGO set", filters=Filters(brand="LEGO"), source_id="s0"),
        GeneratedQuery(id=1, text="a cheap toy", filters=Filters(price_max=10), source_id="s1"),
    ]
    per_query = [
        {"method": "qam", "query_id": 0, "hits": [], "rels": [], "n_candidates": 5,
         "parsed": {"semantic_text": "x", "filters": {}}},  # brand missed
        {"method": "qam", "query_id": 1, "hits": [], "rels": [], "n_candidates": 5,
         "parsed": {"semantic_text": "x", "filters": {"price_max": 10.0, "brand": "Acme"}}},  # extra only
    ]
    text = analyze_script.explain_parse_errors(per_query, queries, "qam")
    assert "1 queries where" in text and "[0] missed=['brand']" in text and "[1]" not in text
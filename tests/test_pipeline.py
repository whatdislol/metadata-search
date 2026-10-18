from __future__ import annotations

import json

from conftest import FakeEncoder, FakeLLM, FakeScorer
from mdsearch.filtering import MetadataFilter
from mdsearch.methods import build_methods
from mdsearch.pipeline import SearchPipeline
from mdsearch.query import QueryParser
from mdsearch.query.schema import ParsedQuery
from mdsearch.rank import BM25Retriever
from mdsearch.representation import TitleDescription


def parse_handler(prompt, system, schema):
    q = prompt.rsplit("Query:", 1)[1]
    if "LEGO" in q:
        return json.dumps(
            {"semantic_text": "creative building", "filters": {"brand": "LEGO", "price_max": 50}}
        )
    return json.dumps({"semantic_text": "board game", "filters": None})


def make_qam(catalog, llm):
    texts = catalog.texts(TitleDescription())
    return SearchPipeline(
        "qam", catalog, BM25Retriever(catalog.ids, texts),
        parser=QueryParser(llm), metadata_filter=MetadataFilter(catalog),
    )


def test_qam_filters_before_ranking(catalog):
    qam = make_qam(catalog, FakeLLM(parse_handler))
    res = qam.search("creative LEGO under 50", k=5)
    assert res.parsed.filters.brand == "LEGO"
    assert [h.id for h in res.hits] == ["a"]  # b is LEGO but too expensive; others wrong brand
    assert res.n_candidates == 1
    assert res.filter_report.remaining == 1


def test_null_filters_from_llm_are_accepted(catalog):
    qam = make_qam(catalog, FakeLLM(parse_handler))
    res = qam.search("a board game", k=3)
    assert res.parsed.filters.is_empty() and res.n_candidates == len(catalog)
    assert res.hits[0].id == "c"


def test_filter_removing_everything_returns_no_hits(catalog):
    llm = FakeLLM(lambda p, s, sc: json.dumps({"semantic_text": "toy", "filters": {"brand": "Nobody"}}))
    res = make_qam(catalog, llm).search("toy by Nobody", k=5)
    assert res.hits == [] and res.n_candidates == 0


def test_parser_falls_back_to_no_filters_on_garbage(catalog):
    llm = FakeLLM(lambda p, s, sc: "garbage")
    qam = make_qam(catalog, llm)
    res = qam.search("monopoly", k=2)
    assert res.parsed == ParsedQuery(semantic_text="monopoly")
    assert qam.parser.n_failures == 1 and res.hits  # still returns unfiltered results


def test_search_many_matches_search_and_batches_parsing(catalog):
    llm = FakeLLM(parse_handler)
    qam = make_qam(catalog, llm)
    queries = ["creative LEGO under 50", "a board game"]
    many = qam.search_many(queries, k=3)
    one = [qam.search(q, k=3) for q in queries]
    assert [[h.id for h in r.hits] for r in many] == [[h.id for h in r.hits] for r in one]


def test_cross_encoder_sees_full_query_and_reorders(catalog):
    texts = catalog.texts(TitleDescription())
    pipe = SearchPipeline(
        "rerank", catalog, BM25Retriever(catalog.ids, texts),
        reranker=FakeScorer(), rerank_depth=5, rerank_text=TitleDescription().render,
    )
    res = pipe.search("racing car", k=2)
    assert res.hits[0].id == "b"


def test_build_methods_produces_all_five(catalog):
    methods = build_methods(
        catalog,
        encoder=FakeEncoder(), scorer=FakeScorer(), llm=FakeLLM(parse_handler),
        dense_rep=TitleDescription(), bm25_rep=TitleDescription(), rerank_rep=TitleDescription(),
    )
    assert list(methods) == ["keyword", "semantic", "rerank", "hybrid", "qam"]
    for m in methods.values():
        assert m.search("LEGO creative", k=3).hits
    assert methods["qam"].search("creative LEGO under 50", k=3).hits[0].id == "a"


def test_qam_requires_llm(catalog):
    import pytest

    with pytest.raises(ValueError):
        build_methods(
            catalog, encoder=FakeEncoder(), scorer=FakeScorer(), llm=None,
            dense_rep=TitleDescription(), bm25_rep=TitleDescription(), rerank_rep=TitleDescription(),
        )

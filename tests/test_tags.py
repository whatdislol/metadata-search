from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import FakeEncoder, FakeLLM, FakeScorer
from mdsearch.filtering.tag_matcher import TagMatcher
from mdsearch.methods import ALL_METHODS, build_methods
from mdsearch.pipeline import SearchPipeline
from mdsearch.query.schema import ParsedQuery, Tag
from mdsearch.query.tag_parser import TagParser
from mdsearch.rank import DenseRetriever
from mdsearch.representation import TitleDescription


def reply(**kw):
    return lambda p, s, sc: json.dumps(kw)


def test_tag_parser_returns_generated_tags():
    llm = FakeLLM(reply(
        semantic_text="building set",
        tags=[{"text": "LEGO", "kind": "brand"}, {"text": " for  ages 5 to 8 ", "kind": "audience"}],
    ))
    parsed = TagParser(llm).parse("LEGO building set for ages 5 to 8")
    assert parsed.semantic_text == "building set"
    assert [t.text for t in parsed.tags] == ["LEGO", "for ages 5 to 8"]
    assert parsed.filters.is_empty()


def test_tag_parser_dedupes_caps_and_fills_empty_semantic_text():
    tags = [{"text": "a"}, {"text": "A"}, {"text": ""}] + [{"text": f"t{i}"} for i in range(10)]
    parsed = TagParser(FakeLLM(reply(semantic_text="  ", tags=tags)), max_tags=4).parse("q")
    assert parsed.semantic_text == "q"
    assert [t.text for t in parsed.tags] == ["a", "t0", "t1", "t2"]  # case-dup and empty dropped


def test_tag_parser_falls_back_on_garbage_and_counts_it():
    parser = TagParser(FakeLLM(lambda p, s, sc: "garbage"))
    assert parser.parse("monopoly") == ParsedQuery(semantic_text="monopoly")
    assert parser.n_failures == 1


def test_tag_parser_parse_many_matches_parse():
    llm = FakeLLM(reply(semantic_text="x", tags=[{"text": "cozy"}]))
    parser = TagParser(llm)
    assert parser.parse_many(["a", "b"]) == [parser.parse("a"), parser.parse("b")]


def make_dense(catalog):
    return DenseRetriever.build(catalog.ids, catalog.texts(TitleDescription()), FakeEncoder())


def test_filter_mode_keeps_top_fraction_of_pool(catalog):
    m = TagMatcher(make_dense(catalog), mode="filter", top_frac=0.4)
    out = m.apply([Tag(text="LEGO racing car")], np.ones(len(catalog), dtype=bool))
    assert out.mask.sum() == 2  # ceil(0.4 * 5)
    assert out.mask[catalog.index("b")]
    assert out.steps[0].removed == 3 and out.steps[0].name == "tag:LEGO racing car"
    assert not out.boost.any()


def test_filter_mode_respects_existing_mask_and_each_tag_narrows(catalog):
    m = TagMatcher(make_dense(catalog), mode="filter", top_frac=0.5)
    start = np.array([True, True, True, False, False])
    out = m.apply([Tag(text="board game"), Tag(text="Monopoly")], start)
    assert out.mask.sum() < start.sum() and not (out.mask & ~start).any()
    assert out.mask[catalog.index("c")]


def test_filter_mode_never_empties_a_nonempty_pool(catalog):
    m = TagMatcher(make_dense(catalog), mode="filter", top_frac=0.01)
    out = m.apply([Tag(text="x"), Tag(text="y"), Tag(text="z")], np.ones(len(catalog), dtype=bool))
    assert out.mask.sum() >= 1


def test_soft_mode_removes_nothing_and_boosts_matching_products(catalog):
    m = TagMatcher(make_dense(catalog), mode="soft", weight=1.0)
    start = np.ones(len(catalog), dtype=bool)
    out = m.apply([Tag(text="soft rattle babies")], start)
    assert out.mask.all() and out.steps == []
    assert out.boost[catalog.index("d")] == out.boost.max()


def test_no_tags_is_a_noop(catalog):
    m = TagMatcher(make_dense(catalog))
    start = np.ones(len(catalog), dtype=bool)
    out = m.apply([], start)
    assert out.mask.all() and not out.boost.any()


def test_invalid_settings_rejected(catalog):
    with pytest.raises(ValueError):
        TagMatcher(make_dense(catalog), mode="hard")
    with pytest.raises(ValueError):
        TagMatcher(make_dense(catalog), top_frac=0)


def tag_pipeline(catalog, **kw):
    dense = make_dense(catalog)
    llm = FakeLLM(reply(semantic_text="toy", tags=[{"text": "LEGO racing car", "kind": "x"}]))
    return SearchPipeline(
        "qam_tags", catalog, dense, parser=TagParser(llm),
        tag_matcher=TagMatcher(dense, **kw),
    )


def test_pipeline_filters_by_tags_and_reports_steps(catalog):
    res = tag_pipeline(catalog, mode="filter", top_frac=0.4).search("fast LEGO car", k=5)
    assert res.n_candidates == 2 and len(res.hits) == 2
    assert res.filter_report.remaining == 2
    assert [s.name for s in res.filter_report.steps] == ["tag:LEGO racing car"]
    assert res.parsed.tags[0].kind == "x"
    assert "tags" in res.to_dict()["parsed"]


def test_pipeline_soft_mode_keeps_every_candidate(catalog):
    res = tag_pipeline(catalog, mode="soft").search("fast LEGO car", k=5)
    assert res.n_candidates == len(catalog) and len(res.hits) == len(catalog)


def test_pipeline_without_tags_matches_plain_search(catalog):
    dense = make_dense(catalog)
    plain = SearchPipeline("p", catalog, dense)
    tagged = SearchPipeline("t", catalog, dense, tag_matcher=TagMatcher(dense))
    q = "board game"
    assert [h.id for h in plain.search(q, k=5).hits] == [h.id for h in tagged.search(q, k=5).hits]


def test_qam_tags_is_opt_in(catalog):
    common = dict(
        encoder=FakeEncoder(), scorer=FakeScorer(), llm=FakeLLM(reply(semantic_text="toy", tags=[])),
        dense_rep=TitleDescription(), bm25_rep=TitleDescription(), rerank_rep=TitleDescription(),
    )
    assert "qam_tags" not in ALL_METHODS
    assert "qam_tags" not in build_methods(catalog, **common)
    got = build_methods(catalog, only=("rerank", "qam_tags"), **common)
    assert list(got) == ["rerank", "qam_tags"]
    assert got["qam_tags"].search("LEGO", k=3).hits
    with pytest.raises(ValueError):
        build_methods(catalog, **{**common, "llm": None}, only=("qam_tags",))


def test_filter_mode_keeps_exactly_the_top_fraction_even_with_tied_scores(catalog):
    class Tied:  # every product has the same similarity to every tag
        def scores(self, text, mask=None):
            return np.zeros(len(catalog))

    m = TagMatcher(Tied(), mode="filter", top_frac=0.4)
    out = m.apply([Tag(text="x")], np.ones(len(catalog), dtype=bool))
    assert out.mask.sum() == 2 and out.steps[0].removed == 3
    assert list(np.flatnonzero(out.mask)) == [0, 1]  # ties go to catalog order


def test_cached_scorer_only_scores_unseen_pairs_and_keeps_order(tmp_path):
    from mdsearch.rank.cross_encoder import CachedScorer

    class Counting:
        name = "m"
        max_length = 512
        calls = 0

        def score(self, query, docs):
            Counting.calls += len(docs)
            return [float(len(d)) for d in docs]

    cs = CachedScorer(Counting(), tmp_path / "ce.sqlite")
    assert cs.score("q", ["aa", "b", "cccc"]) == [2.0, 1.0, 4.0]
    assert cs.score("q", ["cccc", "dd", "aa"]) == [4.0, 2.0, 2.0]  # only "dd" is new
    assert Counting.calls == 4 and cs.hits == 2
    assert CachedScorer(Counting(), tmp_path / "ce.sqlite").score("q", ["b"]) == [1.0]  # persisted
    assert Counting.calls == 4
    assert cs.score("other query", ["b"]) == [1.0] and Counting.calls == 5  # keyed by query too


def test_dense_cache_key_includes_truncation_length():
    from mdsearch.rank.dense import SentenceTransformerEncoder as E

    assert E("m").cache_key == "m" and E("m", max_seq_length=256).cache_key == "m@256"


def test_kind_labels_are_normalised_and_prompt_lists_the_roles():
    from mdsearch.query.schema import TAG_KINDS
    from mdsearch.query.tag_parser import TAG_SYSTEM

    assert Tag(text="red", kind=" Color ").kind == "color" and Tag(text="x", kind="Product Type").kind == "product_type"
    assert all(k in TAG_SYSTEM for k in TAG_KINDS) and "{kinds}" not in TAG_SYSTEM


def test_matcher_applies_only_tags_of_the_requested_kinds(catalog):
    m = TagMatcher(make_dense(catalog), mode="filter", top_frac=0.4, kinds=["brand"])
    ones = np.ones(len(catalog), dtype=bool)
    out = m.apply([Tag(text="LEGO racing car", kind="product_type")], ones)
    assert out.mask.all() and out.steps == []  # restating the product type does nothing
    out = m.apply([Tag(text="LEGO racing car", kind="brand"), Tag(text="x", kind="product_type")], ones)
    assert out.mask.sum() == 2 and [s.name for s in out.steps] == ["tag:LEGO racing car"]


def test_attr_variant_is_built_and_described(catalog):
    from mdsearch.runlog import describe_filtering, filtering_label

    common = dict(
        encoder=FakeEncoder(), scorer=FakeScorer(),
        llm=FakeLLM(reply(semantic_text="toy", tags=[{"text": "LEGO", "kind": "brand"}, {"text": "toy", "kind": "product_type"}])),
        dense_rep=TitleDescription(), bm25_rep=TitleDescription(), rerank_rep=TitleDescription(),
    )
    got = build_methods(catalog, only=("qam_tags_hard", "qam_tags_attr"), **common)
    assert got["qam_tags_attr"].tag_matcher.kinds and got["qam_tags_hard"].tag_matcher.kinds is None
    assert "only" in filtering_label(describe_filtering(got["qam_tags_attr"]))
    assert "only" not in filtering_label(describe_filtering(got["qam_tags_hard"]))
    # the product-type tag is ignored, so the attr variant keeps more candidates than the all-tags one
    assert got["qam_tags_attr"].search("lego", k=5).n_candidates > got["qam_tags_hard"].search("lego", k=5).n_candidates


def test_tag_report_splits_queries_by_attribute_tags():
    from mdsearch.eval.analysis import tag_report

    def row(method, qid, tags, ndcg):
        return {"method": method, "query_id": qid, "nDCG@10": ndcg, "n_candidates": 1,
                "parsed": {"semantic_text": "s", "tags": tags}}

    rows = []
    for q, tags in {"1": [{"text": "red", "kind": "color"}], "2": [{"text": "wig", "kind": "product_type"}]}.items():
        rows += [row("rerank", q, tags, 0.5), row("tags", q, tags, 0.6 if q == "1" else 0.5)]
    text = tag_report(rows, k=10, ref="rerank")
    assert "1 queries have an attribute tag" in text and "-- with attribute tag: 1 queries" in text
    assert "-- without: 1 queries" in text and "color 1" in text and "product_type 1" in text


def test_expansion_is_parsed_and_changes_only_the_first_stage_query(catalog):
    llm = FakeLLM(reply(semantic_text="racing car", tags=[], expansion="  toy  vehicle for racing  "))
    parsed = TagParser(llm, expand=True).parse("fast car")
    assert parsed.expansion == "toy vehicle for racing"
    assert "expansion" in TagParser(llm, expand=True).system and "expansion" not in TagParser(llm).system

    seen = {"first": [], "rerank": []}

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def scores(self, q, mask=None):
            seen["first"].append(q)
            return self.inner.scores(q, mask)

    class SpyScorer(FakeScorer):
        def score(self, q, docs):
            seen["rerank"].append(q)
            return super().score(q, docs)

    pipe = SearchPipeline("e", catalog, Spy(make_dense(catalog)), parser=TagParser(llm, expand=True), reranker=SpyScorer())
    pipe.search("fast car", k=3)
    assert seen["first"] == ["racing car. toy vehicle for racing"] and seen["rerank"] == ["fast car"]


def test_expand_methods_are_built_labelled_and_share_one_parser_prompt(catalog):
    from mdsearch.runlog import describe_filtering, filtering_label

    common = dict(
        encoder=FakeEncoder(), scorer=FakeScorer(),
        llm=FakeLLM(reply(semantic_text="toy", tags=[{"text": "LEGO", "kind": "brand"}], expansion="building bricks")),
        dense_rep=TitleDescription(), bm25_rep=TitleDescription(), rerank_rep=TitleDescription(),
    )
    got = build_methods(catalog, only=("qam_expand", "qam_tags_attr_expand", "qam_tags_attr"), **common)
    assert got["qam_expand"].tag_matcher is None and got["qam_expand"].parser.expand
    assert got["qam_tags_attr_expand"].tag_matcher.kinds and not got["qam_tags_attr"].parser.expand
    assert filtering_label(describe_filtering(got["qam_expand"])) == "none; query expansion"
    assert "query expansion" in filtering_label(describe_filtering(got["qam_tags_attr_expand"]))
    assert "query expansion" not in filtering_label(describe_filtering(got["qam_tags_attr"]))
    assert got["qam_expand"].search("lego", k=3).n_candidates == len(catalog)

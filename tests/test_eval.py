from __future__ import annotations

import json
import random

import pytest

from conftest import FakeEncoder, FakeLLM, FakeScorer
from mdsearch.eval.judge import RelevanceJudge
from mdsearch.eval.metrics import average_precision_at_k, bootstrap_ci, ndcg_at_k, precision_at_k
from mdsearch.eval.querygen import (
    GeneratedQuery,
    describe_requirements,
    generate_queries,
    leaks_title,
    mentions_unrecorded_brand,
    unrecorded_names,
    load_queries,
    mentions_constraints,
    nice_ceiling,
    sample_filters,
    save_queries,
)
from mdsearch.eval.runner import evaluate
from mdsearch.filtering import satisfies
from mdsearch.methods import build_methods
from mdsearch.query.schema import Filters
from mdsearch.representation import TitleDescription


def test_precision_and_ap():
    rels = [1, 0, 1, 0, 0]
    assert precision_at_k(rels, 5) == 0.4
    assert precision_at_k([1], 3) == pytest.approx(1 / 3)  # short lists penalised
    assert average_precision_at_k(rels, 5) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert average_precision_at_k(rels, 5, n_relevant=4) == pytest.approx((1 + 2 / 3) / 4)
    assert average_precision_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_and_bootstrap():
    assert ndcg_at_k([2, 1, 0], 3) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2], 3) < 1.0
    assert ndcg_at_k([0, 0], 2) == 0.0
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert lo < 0.5 < hi


def test_nice_ceiling_and_requirements():
    assert nice_ceiling(34.5) == 35 and nice_ceiling(120.1) == 130 and nice_ceiling(260) == 300
    f = Filters(brand="LEGO", price_max=35, rating_min=4.5, age_min=5, age_max=5)
    text = describe_requirements(f, "£")
    assert "LEGO" in text and "£35" in text and "4.5" in text and "aged 5" in text


def test_sampled_filters_are_satisfied_by_their_source_product(products):
    rng = random.Random(1)
    for _ in range(50):
        for p in products:
            f = sample_filters(p, rng)
            if f is None:
                assert p.id == "e"
            else:
                assert not f.is_empty() and satisfies(p, f)


def test_mentions_constraints():
    f = Filters(brand="LEGO", price_max=35, age_min=5, age_max=5)
    assert mentions_constraints("A lego set for a 5 year old under £35", f)
    assert not mentions_constraints("A set for a 5 year old under £35", f)  # brand missing
    assert not mentions_constraints("A lego set for a 5 year old", f)  # price missing


def test_generate_queries_keeps_only_verified_phrasing(products, tmp_path):
    def handler(prompt, system, schema):
        # a "good" LLM: copies every required value from the requirements block
        reqs = prompt.split("exact values:")[1].split("It must also")[0]
        words = []
        for line in reqs.strip().splitlines():
            if "brand" in line:
                words.append(line.split('"')[1])
            elif "price" in line:
                words.append(line.split("at most ")[1].strip())
            elif "rating" in line:
                words.append(line.split("at least ")[1].split()[0] + " stars")
            elif "age" in line:
                words.append("age " + line.split("aged ")[1].strip())
        return json.dumps({"query": "looking for a toy " + " ".join(words)})

    qs = generate_queries(FakeLLM(handler), products, 3, seed=2)
    assert 1 <= len(qs) <= 3
    by_id = {p.id: p for p in products}
    for q in qs:
        assert mentions_constraints(q.text, q.filters)
        assert satisfies(by_id[q.source_id], q.filters)
    save_queries(tmp_path / "q.jsonl", qs)
    assert load_queries(tmp_path / "q.jsonl") == qs

    # a lazy LLM that ignores the requirements yields nothing
    lazy = generate_queries(FakeLLM(lambda p, s, sc: '{"query": "a toy"}'), products, 3, seed=2)
    assert lazy == []


def test_judge_checks_hard_constraints_in_code_and_asks_llm_only_for_the_rest(catalog):
    asked = []

    def handler(prompt, system, schema):
        asked.append(prompt)
        return json.dumps({"relevant": "Classic" in prompt, "reason": "r"})

    judge = RelevanceJudge(FakeLLM(handler), catalog, price_tol=0.2)
    q = GeneratedQuery(id=0, text="creative LEGO under 30", filters=Filters(brand="LEGO", price_max=30), source_id="a")
    js = judge.judge_many([(q, "a"), (q, "b"), (q, "c")])
    # a: LEGO and 29.99 -> LLM says relevant ("Classic" in prompt); b: 79 too expensive even with
    # tolerance; c: wrong brand. Only `a` should reach the LLM.
    assert [j.relevant for j in js] == [True, False, False]
    assert [j.hard_ok for j in js] == [True, False, False]
    assert len(asked) == 1


def test_judge_unusable_reply_counts_as_not_relevant(catalog):
    judge = RelevanceJudge(FakeLLM(lambda p, s, sc: "garbage"), catalog)
    q = GeneratedQuery(id=0, text="LEGO", filters=Filters(brand="LEGO"), source_id="a")
    (j,) = judge.judge_many([(q, "a")])
    assert j.relevant is False and j.hard_ok is True and j.soft_ok is None


def test_evaluate_end_to_end_with_fakes(catalog, tmp_path):
    def parse_or_judge(prompt, system, schema):
        if system and "judge" in system:
            return json.dumps({"relevant": True, "reason": "ok"})
        return json.dumps({"semantic_text": "creative building", "filters": {"brand": "LEGO"}})

    llm = FakeLLM(parse_or_judge)
    rep = TitleDescription()
    methods = build_methods(
        catalog, encoder=FakeEncoder(), scorer=FakeScorer(), llm=llm,
        dense_rep=rep, bm25_rep=rep, rerank_rep=rep,
    )
    queries = [
        GeneratedQuery(id=0, text="creative building LEGO", filters=Filters(brand="LEGO"), source_id="a"),
        GeneratedQuery(id=1, text="board game Hasbro", filters=Filters(brand="Hasbro"), source_id="c"),
    ]
    report = evaluate(
        methods, queries, RelevanceJudge(llm, catalog), ks=(1, 3), out_dir=tmp_path,
        meta={"llm_provider": "fake"},
    )
    assert set(report["summary"]) == {"keyword", "semantic", "rerank", "hybrid", "qam"}
    for name in report["summary"]:
        for key in ("P@1", "P@3", "mAP@1", "mAP@3"):
            assert 0.0 <= report["summary"][name][key]["mean"] <= 1.0
    # the parser always says "LEGO", so QAM only returns LEGO products: perfect for query 0,
    # nothing relevant for the Hasbro query -> P@1 = 0.5
    assert report["summary"]["qam"]["P@1"]["mean"] == pytest.approx(0.5)
    assert report["meta"]["n_queries"] == 2 and report["meta"]["llm_provider"] == "fake"
    for fname in ("summary.json", "per_query.jsonl", "judgments.jsonl"):
        assert (tmp_path / fname).exists()


def test_sample_filters_max_age_skips_adult_products():
    from mdsearch.data.schema import Product

    adult = Product("x", "Adult party kit", "", "Acme", None, None, None, 18.0)
    kid = Product("y", "Kids puzzle", "", "Acme", None, None, None, 5.0)
    rng = random.Random(0)
    for _ in range(30):
        assert sample_filters(adult, rng, max_age=12) is None or sample_filters(adult, rng, max_age=12).age_min is None
        f = sample_filters(kid, rng, max_age=12)
        assert f is not None
    # without the guard an adult product can get an age constraint
    assert any(sample_filters(adult, random.Random(i)).age_min is not None for i in range(30))


def test_leaks_title():
    from mdsearch.data.schema import Product

    p = Product("x", "Pearlescent Blue Poly RPG Dice Set and Bag", "", "Chessex", None, None, None, None)
    assert leaks_title("a pearlescent blue poly RPG dice set and bag", p)
    assert not leaks_title("dice for role playing games under 10", p)
    short = Product("y", "Toy car", "", None, None, None, None, None)
    assert not leaks_title("toy car", short)  # too few title words to judge


def test_broad_style_uses_broad_prompt_and_drops_title_leaks(products):
    seen_systems = []
    seen_titles = []

    def handler(prompt, system, schema):
        seen_systems.append(system)
        seen_titles.append(prompt.split("Product (for reference only): ")[1].splitlines()[0])
        assert "Rules:" in prompt and "KIND of product" in prompt
        reqs = prompt.split("exact values:")[1].split("Rules:")[0]
        words = []
        for line in reqs.strip().splitlines():
            if "brand" in line:
                words.append(line.split('"')[1])
            elif "price" in line:
                words.append(line.split("at most ")[1].strip())
            elif "rating" in line:
                words.append(line.split("at least ")[1].split()[0] + " stars")
            elif "age" in line:
                words.append("age " + line.split("aged ")[1].strip())
        # a "bad" model that copies the product title for product a, a good one otherwise
        title = prompt.split("Product (for reference only): ")[1].splitlines()[0]
        if "creative bricks" in title:
            return json.dumps({"query": f"{title} " + " ".join(words), "names_used": []})
        if "Technic" in title:  # names a franchise that is not the recorded brand -> dropped
            return json.dumps({"query": "a Batman set " + " ".join(words), "names_used": ["Batman"]})
        return json.dumps({"query": "a gift for a child " + " ".join(words), "names_used": []})

    qs = generate_queries(FakeLLM(handler), products, 4, seed=3, style="broad")
    assert qs and all(q.source_id not in {"a", "b"} for q in qs)  # title-copying and name-leaking queries dropped
    assert all(mentions_constraints(q.text, q.filters) for q in qs)
    assert all(s and "need, not a specific item" in s for s in seen_systems)
    assert "LEGO Technic racing car" in seen_titles  # the name-leaking case was really exercised
    with pytest.raises(ValueError):
        generate_queries(FakeLLM(), products, 1, style="nope")


def test_mentions_unrecorded_brand():
    from mdsearch.data.schema import Product

    p = Product("x", "Classic bricks", "", "LEGO", None, None, None, None)
    assert mentions_unrecorded_brand("a lego building set for age 3", p, Filters(age_min=3, age_max=3))
    assert not mentions_unrecorded_brand("a building set for age 3", p, Filters(age_min=3, age_max=3))
    # when the brand IS a recorded constraint, naming it is required, not a leak
    assert not mentions_unrecorded_brand("a lego building set", p, Filters(brand="LEGO"))


def test_unrecorded_names():
    assert unrecorded_names(["Batman"], Filters(age_min=3)) == ["Batman"]
    assert unrecorded_names(["LEGO"], Filters(brand="LEGO")) == []
    assert unrecorded_names(["Lego", "Star Wars"], Filters(brand="LEGO")) == ["Star Wars"]
    assert unrecorded_names([], Filters()) == []
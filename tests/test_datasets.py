from __future__ import annotations

import gzip
import json

import pytest
from pathlib import Path
from conftest import FakeEncoder, FakeLLM, FakeScorer
from mdsearch.data.evalset import (
    EvalSet, build_qrels_dataset, build_subset, diagnose_ids, iter_jsonl, load_qrels, load_queries_tsv,
    select_queries,
)
from mdsearch.data.registry import PREPARERS, get_preparer, prepare_dataset
from mdsearch.data.schema import load_products
from mdsearch.data.trec_product import clean_text, record_id, to_product
from mdsearch.eval.analysis import analyze_qrels, run_analysis
from mdsearch.eval.dispatch import default_methods, eval_mode, format_outcome, results_dir, run_evaluation
from mdsearch.eval.qrels_eval import ap_at_k, evaluate_qrels
from mdsearch.eval.runner import format_table
from mdsearch.methods import build_methods
from mdsearch.pipeline import SearchPipeline
from mdsearch.rank import BM25Retriever
from mdsearch.representation import TitleDescription

# records copied from the real TREC corpus
REC1 = {"docid": 1, "title": "FYY Leather Case with Mirror for Samsung Galaxy S8 Plus  Black",
        "text": "Product Description Premium PU Leather Top quality. Hand strap makes it easy to carry. "}
REC2 = {"docid": 2, "title": "Playtex Women's 18 Hour Easy On Post Surgery Bra US400C",
        "text": "Product Description Introducing Playtex bra. Wide straps. "
                "Product Description Introducing Playtex bra. Wide straps."}
REC4 = {"docid": 4, "title": "YUEPIN U-Tube Clamp 304 Stainless Steel Hose Pipe (10pcs)",
        "text": "Product Description Material: 304 Stainless Steel Color: Silver Quantity: 10 Pieces"}


def test_clean_text_drops_label_and_verbatim_repeats():
    assert clean_text(REC2["text"]) == "Introducing Playtex bra. Wide straps."
    assert clean_text(REC1["text"]) == "Premium PU Leather Top quality. Hand strap makes it easy to carry."
    assert clean_text("") == "" and clean_text(None) == ""


def test_to_product_uses_string_ids_and_leaves_metadata_empty():
    p = to_product(REC1)
    assert p.id == "1" and p.title.endswith("S8 Plus Black")  # whitespace collapsed
    assert p.brand is None and p.price is None and p.rating is None and p.min_age is None
    assert record_id(REC1) == "1"


def write_jsonl(path, recs):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


@pytest.mark.parametrize("name", ["corpus.jsonl", "corpus.jsonl.gz"])
def test_iter_jsonl_reads_plain_and_gzip(tmp_path, name):
    path = tmp_path / name
    write_jsonl(path, [REC1, REC2])
    assert [r["docid"] for r in iter_jsonl(path)] == [1, 2]


def test_queries_and_qrels_formats(tmp_path):
    q = tmp_path / "q.tsv"
    q.write_text("1\tcell phone wallet case\n\n2\tsurgery bra\n3\t\n", encoding="utf-8")
    assert load_queries_tsv(q) == {"1": "cell phone wallet case", "2": "surgery bra"}
    r = tmp_path / "qrels.txt"
    r.write_text("1 0 1 2\n1 0 2 0\n2 0 4 1\nbad line\n", encoding="utf-8")
    assert load_qrels(r) == {"1": {"1": 2, "2": 0}, "2": {"4": 1}}
    r3 = tmp_path / "qrels3.txt"
    r3.write_text("1 1 3\n", encoding="utf-8")
    assert load_qrels(r3) == {"1": {"1": 3}}


def test_select_queries_requires_text_and_relevant_judgments():
    queries = {"1": "a", "2": "b", "3": "c"}
    qrels = {"1": {"10": 1}, "2": {"11": 0}, "4": {"12": 1}}
    assert select_queries(queries, qrels, max_queries=None) == ["1"]
    many = {str(i): "q" for i in range(50)}
    many_qrels = {str(i): {"d": 1} for i in range(50)}
    a = select_queries(many, many_qrels, max_queries=10, seed=1)
    assert a == select_queries(many, many_qrels, max_queries=10, seed=1) and len(a) == 10


def test_build_subset_keeps_all_judged_and_samples_the_rest():
    recs = [{"docid": i, "title": f"t{i}", "text": "x"} for i in range(100)]
    args = (to_product, record_id)
    products, n = build_subset(iter(recs), *args, {"3", "50"}, n_distractors=10, seed=7)
    ids = {p.id for p in products}
    assert n == 100 and len(products) == 12 and {"3", "50"} <= ids
    again, _ = build_subset(iter(recs), *args, {"3", "50"}, n_distractors=10, seed=7)
    assert {p.id for p in again} == ids  # deterministic
    small, _ = build_subset(iter(recs[:5]), *args, set(), n_distractors=10)
    assert len(small) == 5  # fewer records than the sample size


def test_build_qrels_dataset_filters_queries_and_warns_about_missing_ids():
    recs = [{"docid": i, "title": f"t{i}", "text": "x"} for i in range(1, 51)]
    queries = {"a": "alpha", "b": "beta", "c": "gamma"}
    qrels = {"a": {"5": 2, "6": 0}, "b": {"999": 1}, "c": {"7": 1}}
    logs: list[str] = []
    products, ev = build_qrels_dataset(
        iter(recs), to_product, record_id, queries, qrels,
        {"n_distractors": 5, "seed": 1, "max_queries": 10}, log=logs.append,
    )
    assert set(ev.queries) == {"a", "c"} and ev.qrels["a"] == {"5": 2, "6": 0}  # b lost its only relevant doc
    assert {"5", "6", "7"} <= {p.id for p in products}
    assert any("WARNING" in m for m in logs)  # 999 is not in the corpus: 3 of 4 judged found
    with pytest.raises(SystemExit):
        build_qrels_dataset(iter(recs), to_product, record_id, {"a": "x"}, {"z": {"1": 1}}, {}, log=lambda m: None)


def test_evalset_roundtrip(tmp_path):
    ev = EvalSet({"q": "text"}, {"q": {"d": 2}}, 2, {"n": 1})
    ev.save(tmp_path / "sub" / "e.json")
    back = EvalSet.load(tmp_path / "sub" / "e.json")
    assert back.queries == ev.queries and back.qrels == ev.qrels
    assert back.rel_threshold == 2 and back.notes == {"n": 1}


def test_registry_knows_every_dataset_and_prepares_trec_end_to_end(tmp_path):
    assert set(PREPARERS) == {"toys_kaggle", "trec_product", "wands", "esci"}
    for kind in PREPARERS:
        assert callable(get_preparer(kind))
    with pytest.raises(SystemExit):
        get_preparer("nope")
    with pytest.raises(NotImplementedError):
        get_preparer("wands")({})

    write_jsonl(tmp_path / "c.jsonl.gz", [REC1, REC2, REC4])
    (tmp_path / "q.tsv").write_text("10\tstainless steel clamp\n11\tleather case\n", encoding="utf-8")
    (tmp_path / "qrels.txt").write_text("10 0 4 2\n10 0 2 0\n11 0 1 1\n", encoding="utf-8")
    cfg = {"dataset": {"kind": "trec_product", "corpus": str(tmp_path / "c.jsonl.gz"),
                       "queries": str(tmp_path / "q.tsv"), "qrels": str(tmp_path / "qrels.txt"),
                       "processed": str(tmp_path / "out" / "p.jsonl"), "eval_set": str(tmp_path / "out" / "e.json")},
           "subset": {"n_distractors": 0, "seed": 1}}
    prepare_dataset(cfg)
    assert {p.id for p in load_products(tmp_path / "out" / "p.jsonl")} == {"1", "2", "4"}
    ev = EvalSet.load(tmp_path / "out" / "e.json")
    assert ev.queries == {"10": "stainless steel clamp", "11": "leather case"}
    assert ev.qrels["10"] == {"4": 2, "2": 0}
    with pytest.raises(SystemExit):  # missing file gives a message, not a traceback
        prepare_dataset({**cfg, "dataset": {**cfg["dataset"], "qrels": str(tmp_path / "missing")}})


def test_ap_is_normalised_by_all_relevant_in_the_qrels():
    assert ap_at_k([False, True, False, True, False], 5, n_relevant=2) == pytest.approx(0.5)
    assert ap_at_k([False, True, False, True, False], 5, n_relevant=6) == pytest.approx(0.2)
    assert ap_at_k([False] * 5, 5, n_relevant=0) == 0.0


def kw_pipeline(catalog):
    return SearchPipeline("kw", catalog, BM25Retriever(catalog.ids, catalog.texts(TitleDescription())))


def test_evaluate_qrels_scores_methods_and_records_filtering(catalog, tmp_path):
    queries = {"q1": "LEGO creative bricks", "q2": "soft rattle", "q3": "nothing judged"}
    qrels = {"q1": {"a": 2, "b": 1, "c": 0}, "q2": {"d": 1}, "q3": {"c": 0}}
    rep = evaluate_qrels({"kw": kw_pipeline(catalog)}, queries, qrels, ks=(1, 3), out_dir=tmp_path, verbose=False)
    s = rep["summary"]["kw"]
    assert rep["meta"]["n_queries"] == 2 and rep["meta"]["n_queries_skipped_no_relevant"] == 1
    assert s["P@1"]["mean"] == 1.0 and s["nDCG@1"]["mean"] == pytest.approx(1.0)
    assert s["R@3"]["mean"] == pytest.approx(1.0)
    assert rep["meta"]["filtering"]["kw"]["kind"] == "none"
    assert len((tmp_path / "per_query.jsonl").read_text().splitlines()) == 2
    assert "kw" in format_table(rep, ["P@1", "nDCG@3"])


def qrels_cfg(tmp_path, mode="qrels"):
    return {
        "dataset": {"eval_set": str(tmp_path / "e.json")},
        "eval": {"mode": mode, "ks": [1, 3], "out_dir": str(tmp_path / "res"), "methods": ["rerank", "qam_tags_hard"]},
        "retrieval": {"tags": {"mode": "filter"}},
    }


def tag_reply(p, s, sc):
    return json.dumps({"semantic_text": "toy", "tags": [{"text": "LEGO racing car", "kind": "x"}]})


def test_dispatch_qrels_mode_end_to_end_then_analysis(catalog, tmp_path):
    EvalSet({"q1": "LEGO creative bricks", "q2": "soft rattle"},
            {"q1": {"a": 2, "b": 1}, "q2": {"d": 1}}).save(tmp_path / "e.json")
    methods = build_methods(
        catalog, encoder=FakeEncoder(), scorer=FakeScorer(), llm=FakeLLM(tag_reply),
        dense_rep=TitleDescription(), bm25_rep=TitleDescription(), rerank_rep=TitleDescription(),
        only=("rerank", "qam_tags_hard", "qam_tags_soft"),
    )
    cfg = qrels_cfg(tmp_path)
    out = run_evaluation(cfg, catalog, methods, llm=None, config_name="x", verbose=False)
    kinds = {n: d["kind"] for n, d in out.report["meta"]["filtering"].items()}
    assert kinds == {"rerank": "none", "qam_tags_hard": "hard", "qam_tags_soft": "soft"}
    assert out.out_dir == tmp_path / "res" and (out.out_dir / "summary.json").exists()
    text = format_outcome(out)
    assert "P@1" in text and "nDCG@3" in text and "qam_tags_hard" in text

    report, filtering = run_analysis(cfg, k=3, ref="rerank")
    assert "reference = rerank" in report and "qam_tags_soft" in report
    assert filtering["qam_tags_hard"]["kind"] == "hard"
    suffixed = run_evaluation(cfg, catalog, methods, llm=None, out_suffix="_b", verbose=False)
    assert suffixed.out_dir.name == "res_b"
    limited = run_evaluation(cfg, catalog, methods, llm=None, limit=1, verbose=False)
    assert limited.report["meta"]["n_queries"] == 1


def test_analyze_qrels_reports_pool_sizes_and_paired_comparison():
    def row(m, q, p, g, pool):
        return {"method": m, "query_id": q, "P@3": p, "nDCG@3": g, "R@3": p, "n_candidates": pool}
    rows = [row("a", "1", 1.0, 1.0, 100), row("a", "2", 0.0, 0.0, 100),
            row("b", "1", 0.5, 0.4, 7), row("b", "2", 0.0, 0.0, 3)]
    text = analyze_qrels(rows, k=3, ref="a")
    assert "2 queries" in text and "a" in text and "1/0" in text  # ref beats b once, never loses
    with pytest.raises(SystemExit):
        analyze_qrels(rows, k=7, ref="a")


def test_eval_mode_defaults_and_validation(tmp_path):
    assert eval_mode({"eval": {}}) == "llm_judge" and eval_mode(qrels_cfg(tmp_path)) == "qrels"
    with pytest.raises(SystemExit):
        eval_mode({"eval": {"mode": "magic"}})
    assert default_methods({"eval": {}}) == ("keyword", "semantic", "rerank", "hybrid", "qam")
    assert default_methods(qrels_cfg(tmp_path)) == ("rerank", "qam_tags_hard")
    assert results_dir(qrels_cfg(tmp_path), out_suffix="_x").name == "res_x"


def test_llm_judge_mode_needs_an_llm_and_queries(catalog, tmp_path):
    cfg = {"dataset": {}, "eval": {"mode": "llm_judge", "ks": [3], "out_dir": str(tmp_path / "r"),
                                   "out_dir_broad": str(tmp_path / "rb")},
           "queries": {"path": str(tmp_path / "missing.jsonl"), "path_broad": str(tmp_path / "b.jsonl")}}
    with pytest.raises(SystemExit, match="LLM"):
        run_evaluation(cfg, catalog, {}, llm=None)
    with pytest.raises(SystemExit, match="generate_queries"):
        run_evaluation(cfg, catalog, {}, llm=FakeLLM())


def test_dispatch_llm_judge_mode_end_to_end_then_analysis(catalog, tmp_path):
    from mdsearch.eval.querygen import GeneratedQuery, save_queries
    from mdsearch.query.schema import Filters

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
    save_queries(tmp_path / "q.jsonl", [
        GeneratedQuery(id=0, text="creative building LEGO", filters=Filters(brand="LEGO"), source_id="a"),
        GeneratedQuery(id=1, text="board game Hasbro", filters=Filters(brand="Hasbro"), source_id="c"),
    ])
    cfg = {"dataset": {"processed": str(tmp_path / "p.jsonl")},
           "queries": {"path": str(tmp_path / "q.jsonl"), "path_broad": str(tmp_path / "qb.jsonl")},
           "eval": {"mode": "llm_judge", "ks": [1, 3], "out_dir": str(tmp_path / "r"),
                    "out_dir_broad": str(tmp_path / "rb")}}
    out = run_evaluation(cfg, catalog, methods, llm=llm, config_name="x", verbose=False)
    assert out.out_dir == tmp_path / "r" and out.report["meta"]["eval_mode"] == "llm_judge"
    assert out.report["meta"]["filtering"]["qam"]["kind"] == "hard"
    assert out.report["meta"]["filtering"]["rerank"]["kind"] == "none"
    assert "mAP@3" in format_outcome(out)
    for name in ("summary.json", "per_query.jsonl", "judgments.jsonl"):
        assert (out.out_dir / name).exists()

    text, filtering = run_analysis(cfg, k=1, ref="qam")
    assert "query style: specific" in text and filtering["qam"]["kind"] == "hard"


def test_clean_text_rebuilds_letter_spaced_passages_from_the_real_corpus():
    spaced = ("Cell Phones & Accessories Flip Cases P r o d u c t   D e s c r i p t i o n   "
              "P r e m i u m   P U   L e a t h e r   1 0 0 %   H a n d m a d e .   RFID Technique: Radio waves. "
              "Manufacturer ‏ : ‎ Playtex ASIN ‏ : ‎ B07JZR6626")
    out = clean_text(spaced)
    assert out == ("Cell Phones & Accessories Flip Cases Premium PU Leather 100% Handmade. "
                   "RFID Technique: Radio waves. Manufacturer : Playtex ASIN : B07JZR6626")
    # ordinary text with short runs of single characters is left alone
    assert clean_text("sizes 1 2 3 4 5 6 7 8 available") == "sizes 1 2 3 4 5 6 7 8 available"
    assert clean_text("Product Description a b c Product Description a b c") == "a b c"


def test_diagnose_ids_flags_files_from_different_tracks():
    queries = {"299": "peplum top", "314": "bikini pants"}
    qrels = {"300001": {"404120": 3, "247391": 3, "1285531": 2}}
    text = "\n".join(diagnose_ids(queries, qrels))
    assert "shared query ids: 0" in text and "do not belong together" in text
    assert "299 .. 314" in text and "300001 .. 300001" in text and "{2: 1, 3: 2}" in text
    ok = "\n".join(diagnose_ids({"300001": "q", "300002": "r"}, {"300001": {"1": 1}}))
    assert "shared query ids: 1" in ok and "only 1 of 1" not in ok


def test_queries_header_row_is_skipped_and_ids_must_match_the_qrels(tmp_path):
    nist = tmp_path / "nist.tsv"
    nist.write_text("id\tquery\n300001\tpeplum top\n300002\twomen light weight bikini pants\n", encoding="utf-8")
    assert load_queries_tsv(nist) == {"300001": "peplum top", "300002": "women light weight bikini pants"}
    hf = tmp_path / "hf.tsv"  # same queries, internal ids, no header: cannot be joined to the qrels
    hf.write_text("299\tpeplum top\n314\twomen light weight bikini pants\n", encoding="utf-8")
    qrels = {"300001": {"404120": 3}}
    assert "shared query ids: 1" in "\n".join(diagnose_ids(load_queries_tsv(nist), qrels))
    assert "shared query ids: 0" in "\n".join(diagnose_ids(load_queries_tsv(hf), qrels))
    odd = tmp_path / "odd.tsv"  # a real query that merely starts with "id" is not a header
    odd.write_text("id\tcard holder\n", encoding="utf-8")
    assert load_queries_tsv(odd) == {"id": "card holder"}


def test_negative_grade_is_treated_as_not_relevant_and_does_not_lower_ndcg(catalog, tmp_path):
    base = {"q1": {"a": 2, "b": 1}}
    with_neg = {"q1": {"a": 2, "b": 1, "c": -1, "d": -1, "e": -1}}  # TREC -1: could not be judged
    kw = lambda d: evaluate_qrels({"kw": kw_pipeline(catalog)}, {"q1": "LEGO creative bricks"}, d,
                                  ks=(5,), out_dir=tmp_path, verbose=False)["summary"]["kw"]
    assert kw(with_neg)["nDCG@5"]["mean"] == pytest.approx(kw(base)["nDCG@5"]["mean"])
    assert kw(with_neg)["P@5"]["mean"] == pytest.approx(kw(base)["P@5"]["mean"])


def test_several_query_and_qrels_files_are_merged_and_overlaps_rejected(tmp_path):
    from mdsearch.data.evalset import as_paths, load_qrels_files, load_queries_files

    q1, q2, q3 = (tmp_path / n for n in ("q23.tsv", "q24.tsv", "clash.tsv"))
    q1.write_text("id\tquery\n200001\tred wig\n", encoding="utf-8")
    q2.write_text("id\tquery\n300001\tpeplum top\n", encoding="utf-8")
    q3.write_text("200001\tsomething else\n", encoding="utf-8")
    assert load_queries_files([q1, q2]) == {"200001": "red wig", "300001": "peplum top"}
    with pytest.raises(SystemExit, match="different queries"):
        load_queries_files([q1, q3])
    r1, r2 = tmp_path / "r1.txt", tmp_path / "r2.txt"
    r1.write_text("200001 0 5 2\n", encoding="utf-8")
    r2.write_text("300001 0 7 3\n200001 0 9 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="more than one qrels file"):
        load_qrels_files([r1, r2])
    assert load_qrels_files([r1]) == {"200001": {"5": 2}}
    assert as_paths("a.tsv") == [Path("a.tsv")] and as_paths(["a", "b"]) == [Path("a"), Path("b")]

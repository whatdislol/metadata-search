from __future__ import annotations

import json

import pytest
from conftest import FakeEncoder, FakeLLM, FakeScorer
from mdsearch.filtering import MetadataFilter
from mdsearch.filtering.tag_matcher import TagMatcher
from mdsearch.methods import build_methods
from mdsearch.pipeline import SearchPipeline
from mdsearch.rank import BM25Retriever, DenseRetriever
from mdsearch.representation import TitleDescription
from mdsearch.runlog import describe_filtering, describe_methods, filtering_label, record_run


def base(catalog):
    return SearchPipeline("p", catalog, BM25Retriever(catalog.ids, catalog.texts(TitleDescription())))


def test_no_filter_is_none(catalog):
    d = describe_filtering(base(catalog))
    assert d == {"kind": "none", "components": [], "query_expansion": False} and filtering_label(d) == "none"


def test_metadata_filter_is_hard(catalog):
    p = base(catalog)
    p.metadata_filter = MetadataFilter(catalog, missing="keep", price_tol=0.2)
    d = describe_filtering(p)
    assert d["kind"] == "hard" and d["components"][0]["missing_values"] == "keep"
    assert filtering_label(d) == "hard (metadata fields: missing=keep)"


def test_tag_modes_are_hard_and_soft(catalog):
    dense = DenseRetriever.build(catalog.ids, catalog.texts(TitleDescription()), FakeEncoder())
    p = base(catalog)
    p.tag_matcher = TagMatcher(dense, mode="filter", top_frac=0.3)
    assert describe_filtering(p)["kind"] == "hard"
    assert "top 30%" in filtering_label(describe_filtering(p))
    p.tag_matcher = TagMatcher(dense, mode="soft", weight=0.5)
    assert describe_filtering(p)["kind"] == "soft"
    assert "weight 0.5" in filtering_label(describe_filtering(p))


def test_both_filters_give_hard_plus_soft(catalog):
    dense = DenseRetriever.build(catalog.ids, catalog.texts(TitleDescription()), FakeEncoder())
    p = base(catalog)
    p.metadata_filter = MetadataFilter(catalog)
    p.tag_matcher = TagMatcher(dense, mode="soft")
    assert describe_filtering(p)["kind"] == "hard+soft"


def test_describe_methods_over_built_methods(catalog):
    methods = build_methods(
        catalog, encoder=FakeEncoder(), scorer=FakeScorer(), llm=FakeLLM(),
        dense_rep=TitleDescription(), bm25_rep=TitleDescription(), rerank_rep=TitleDescription(),
        only=("rerank", "qam", "rerank_soft"),
    )
    kinds = {n: d["kind"] for n, d in describe_methods(methods).items()}
    assert kinds == {"rerank": "none", "qam": "hard", "rerank_soft": "soft"}


def test_record_run_saves_output_and_filtering(tmp_path, capsys):
    with record_run("demo", config="c", args={"k": 5}, label="x", root=tmp_path) as run:
        print("hello")
        run.set_filtering({"m": {"kind": "soft", "components": [
            {"type": "soft", "source": "generated tags matched by text", "weight": 0.5}]}})
    (folder,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert folder.name.endswith("_demo_x")
    out = (folder / "output.txt").read_text()
    assert "hello" in out and "filtering [m]: soft (tags: score boost, weight 0.5)" in out
    info = json.loads((folder / "run.json").read_text())
    assert info["status"] == "ok" and info["args"] == {"k": 5}
    assert info["filtering"]["m"]["kind"] == "soft"
    assert (tmp_path / "README.md").exists()
    assert "hello" in capsys.readouterr().out  # still printed to the console


def test_record_run_keeps_a_record_when_the_script_fails(tmp_path):
    with pytest.raises(RuntimeError):
        with record_run("boom", root=tmp_path):
            raise RuntimeError("bad")
    (folder,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    info = json.loads((folder / "run.json").read_text())
    assert info["status"] == "failed" and "bad" in info["error"]

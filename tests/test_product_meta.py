"""Document-side metadata: product tags, their text, and the _meta methods."""

from __future__ import annotations

import json

import pytest
from conftest import FakeEncoder, FakeLLM, FakeScorer
from mdsearch.config import load_experiment_config
from mdsearch.data.schema import Product
from mdsearch.methods import MetaReps, build_methods, meta_cache_path
from mdsearch.product_tags import ProductTagger, append_product_tags, load_product_tags
from mdsearch.query.schema import Tag
from mdsearch.representation import TaggedMetadata, TitleDescription


def reply(**kw):
    return lambda p, s, sc: json.dumps(kw)


def test_config_extends_merges_sections_and_replaces_lists(tmp_path, monkeypatch):
    (tmp_path / "base.yaml").write_text(
        "dataset: {queries: [a, b], processed: base.jsonl}\nsubset: {max_queries: null, seed: 1}\neval: {methods: [x]}\n")
    (tmp_path / "child.yaml").write_text(
        "extends: base\ndataset: {processed: child.jsonl}\nsubset: {max_queries: 5}\neval: {methods: [y, z]}\n")
    monkeypatch.setenv("MDSEARCH_CONFIG_DIR", str(tmp_path))
    cfg = load_experiment_config("child")
    assert cfg["dataset"] == {"queries": ["a", "b"], "processed": "child.jsonl"}  # inherited + overridden
    assert cfg["subset"] == {"max_queries": 5, "seed": 1} and cfg["eval"]["methods"] == ["y", "z"]
    assert "extends" not in cfg
    (tmp_path / "loop.yaml").write_text("extends: loop\n")
    with pytest.raises(ValueError, match="loop"):
        load_experiment_config("loop")


def test_product_tagger_cleans_dedupes_caps_and_counts_failures():
    p = Product("1", "Detroit  Axle hub", "Fits Ford F-150. " * 200)
    llm = FakeLLM(reply(tags=[{"kind": "Brand", "text": " Detroit   Axle "}, {"kind": "brand", "text": "detroit axle"},
                              {"kind": "feature", "text": "8-lug"}, {"kind": "feature", "text": ""},
                              {"kind": "color", "text": "black"}]))
    tagger = ProductTagger(llm, max_tags=3, max_chars=100)
    tags = tagger.tag_many([p])[0]
    assert [(t.kind, t.text) for t in tags] == [("brand", "Detroit Axle"), ("feature", "8-lug"), ("color", "black")]
    prompt = llm.calls[0][0]
    assert "Title: Detroit Axle hub" in prompt and len(prompt.split("Description: ")[-1]) <= 100
    assert "product_type" in tagger.system and "At most 3 tags" in tagger.system

    bad = ProductTagger(FakeLLM(lambda *a: "not json"))
    assert bad.tag_many([p]) == [[]] and bad.n_failures == 1  # falls back to no tags, and says so


def test_product_tags_file_roundtrip_and_last_line_wins(tmp_path):
    path = tmp_path / "sub" / "tags.jsonl"
    append_product_tags(path, ["1", "2"], [[Tag(text="blue", kind="color")], []])
    append_product_tags(path, ["1"], [[Tag(text="red", kind="color")]])
    got = load_product_tags(path)
    assert [(t.kind, t.text) for t in got["1"]] == [("color", "red")] and got["2"] == []


def test_tagged_representation_positions_kinds_and_untagged_products():
    p = Product("1", "Lovelyshop Case", "A blue phone case for Galaxy.")
    tags = {"1": [Tag(text="Lovelyshop", kind="brand"), Tag(text="blue", kind="color"), Tag(text="x", kind="other")]}
    base = TitleDescription()
    before = TaggedMetadata(base, tags).render(p)
    assert before == "Lovelyshop Case. Metadata: brand=Lovelyshop; color=blue; other=x. A blue phone case for Galaxy."
    assert TaggedMetadata(base, tags, "after").render(p) == (
        "Lovelyshop Case. A blue phone case for Galaxy. Metadata: brand=Lovelyshop; color=blue; other=x")
    assert TaggedMetadata(base, tags, kinds=["brand"]).render(p).startswith("Lovelyshop Case. Metadata: brand=Lovelyshop.")
    assert TaggedMetadata(base, {}).render(p) == base.render(p)  # no tags: the plain text
    with pytest.raises(ValueError):
        TaggedMetadata(base, tags, "middle")


def common(llm=None):
    return dict(encoder=FakeEncoder(), scorer=FakeScorer(), llm=llm, dense_rep=TitleDescription(),
                bm25_rep=TitleDescription(), rerank_rep=TitleDescription())


def test_meta_methods_search_the_text_with_metadata(catalog, tmp_path):
    tags = {"e": [Tag(text="Zorblax", kind="brand")]}  # a word that is nowhere in e's own text
    rep = TaggedMetadata(TitleDescription(), tags)
    meta = MetaReps(rep, rep, rep)
    got = build_methods(catalog, only=("keyword", "keyword_meta", "semantic", "semantic_meta", "rerank", "rerank_meta",
                                       "hybrid", "hybrid_meta"), meta=meta, embedding_cache=tmp_path / "d.npz", **common())
    top = lambda name: got[name].search("zorblax", k=1).hits[0].id
    for base in ("keyword", "semantic", "rerank", "hybrid"):
        assert top(f"{base}_meta") == "e"  # found through its metadata tag
    assert top("keyword") != "e" or got["keyword"].search("zorblax", k=5).hits[0].score == 0  # plain BM25 has no match
    assert all(got[n].product_metadata for n in got if n.endswith("_meta")) and not got["rerank"].product_metadata
    # the plain embeddings and the metadata embeddings are cached in separate files
    assert (tmp_path / "d.npz").exists() and meta_cache_path(tmp_path / "d.npz").exists()
    assert meta_cache_path(tmp_path / "d.npz").name == "d_meta.npz" and meta_cache_path(None) is None


def test_meta_rerank_scores_the_cross_encoder_text_with_metadata(catalog):
    seen = []

    class Spy(FakeScorer):
        def score(self, q, docs):
            seen.append(list(docs))
            return super().score(q, docs)

    tags = {"a": [Tag(text="Zorblax", kind="brand")]}
    rep = TaggedMetadata(TitleDescription(), tags)
    kw = {**common(), "scorer": Spy()}
    got = build_methods(catalog, only=("rerank", "rerank_meta"), meta=MetaReps(rep, rep, rep), **kw)
    got["rerank"].search("lego", k=2)
    got["rerank_meta"].search("lego", k=2)
    assert not any("Metadata:" in d for d in seen[0]) and any("Metadata: brand=Zorblax" in d for d in seen[1])


def test_components_without_metadata_reuse_the_plain_index(catalog, tmp_path):
    plain = TitleDescription()
    tagged = TaggedMetadata(plain, {"e": [Tag(text="Zorblax", kind="brand")]})
    kw = {**common(), "dense_rep": plain, "bm25_rep": plain, "rerank_rep": plain}
    got = build_methods(catalog, only=("semantic_meta", "keyword_meta"), meta=MetaReps(plain, tagged, plain),
                        embedding_cache=tmp_path / "d.npz", **kw)  # only BM25 reads the metadata
    assert got["keyword_meta"].search("zorblax", k=1).hits[0].id == "e"
    assert got["semantic_meta"].first_stage is not None
    assert not meta_cache_path(tmp_path / "d.npz").exists()  # no second embedding was needed


def test_meta_methods_need_product_metadata(catalog):
    with pytest.raises(ValueError, match="representation.metadata"):
        build_methods(catalog, only=("rerank_meta",), **common())


def test_meta_methods_combine_with_query_side_steps(catalog, tmp_path):
    llm = FakeLLM(reply(semantic_text="toy", tags=[{"text": "LEGO", "kind": "brand"}], expansion="building bricks"))
    rep = TaggedMetadata(TitleDescription(), {"a": [Tag(text="LEGO", kind="brand")]})
    got = build_methods(catalog, only=("rerank_meta_soft_expand", "hybrid_meta_hard"), meta=MetaReps(rep, rep, rep),
                        embedding_cache=tmp_path / "d.npz", **common(llm))
    soft, hard = got["rerank_meta_soft_expand"], got["hybrid_meta_hard"]
    assert soft.tag_matcher.mode == "soft" and soft.parser.expand and soft.product_metadata
    assert hard.tag_matcher.mode == "filter" and hard.search("lego", k=3).n_candidates < len(catalog)
    from mdsearch.runlog import describe_filtering, filtering_label
    assert filtering_label(describe_filtering(soft)).endswith("query expansion; product metadata in the text)")
    assert filtering_label(describe_filtering(got["hybrid_meta_hard"])).endswith("product metadata in the text)")


def test_tag_catalog_resumes_limits_and_appends_in_batches(tmp_path):
    from mdsearch.product_tags import tag_catalog

    prods = [Product(str(i), f"item {i}", "text") for i in range(10)]
    llm = FakeLLM(reply(tags=[{"kind": "brand", "text": "Acme"}]))
    path, logs = tmp_path / "t.jsonl", []
    s1 = tag_catalog(prods, ProductTagger(llm), path, limit=4, batch=3, log=logs.append)
    assert s1["tagged_now"] == 4 and len(load_product_tags(path)) == 4 and len(llm.calls) == 4
    first = set(load_product_tags(path))
    s2 = tag_catalog(prods, ProductTagger(llm), path, batch=3, log=logs.append)  # resumes: only the other 6
    assert s2 == {"already": 4, "tagged_now": 6, "failures": 0} and len(llm.calls) == 10
    assert len(load_product_tags(path)) == 10 and first < set(load_product_tags(path))
    assert tag_catalog(prods, ProductTagger(llm), path, log=logs.append)["tagged_now"] == 0 and len(llm.calls) == 10
    # the same random order every time: a fresh file with the same limit picks the same products
    other = tmp_path / "o.jsonl"
    tag_catalog(prods, ProductTagger(llm), other, limit=4, log=logs.append)
    assert set(load_product_tags(other)) == first
    assert any("products/s" in line for line in logs)


def test_experiment_builds_meta_representations_from_the_config(catalog, capsys):
    from dataclasses import replace
    from mdsearch.experiment import _meta_reps
    from mdsearch.representation import NativeMetadata

    plain = {c: TitleDescription() for c in ("dense", "bm25", "rerank")}
    assert _meta_reps({}, plain, catalog, True) is None  # nothing configured -> build_methods explains what to do
    assert _meta_reps({"metadata": {"position": "after"}}, plain, catalog, True) is None
    catalog.products[0].extra = {"attrs": {"Brand": "LEGO"}}
    catalog.products[0].category = "Toys > Building > Sets"
    rep = {"metadata": {"fields": ["Brand"], "category_levels": 2}}
    reps = _meta_reps(rep, plain, catalog, True)
    assert all(isinstance(r, NativeMetadata) for r in (reps.dense, reps.bm25, reps.rerank))
    out = capsys.readouterr().out
    assert f"1 of {len(catalog)} products have some" in out and "category=Building > Sets; brand=LEGO" in out
    only_ce = _meta_reps({"metadata": {**rep["metadata"], "use": ["rerank"], "position": "after", "file": "x"}},
                         plain, catalog, False)
    assert only_ce.dense is plain["dense"] and only_ce.bm25 is plain["bm25"]  # same objects: plain indexes are reused
    assert only_ce.rerank.position == "after"
    with pytest.raises(SystemExit, match="unknown component"):
        _meta_reps({"metadata": {**rep["metadata"], "use": ["dense", "x"]}}, plain, catalog, False)


def test_native_metadata_text():
    from mdsearch.representation import NativeMetadata

    p = Product(id="1", title="YUEPIN Clamp", description="steel clamp", category="Tools > Hand Tools > Pipe Clamps",
                extra={"attrs": {"Material": "steel", "Brand": "YUEPIN"}, "info": {"Manufacturer": "YUEPIN", "Size": "10pcs"}})
    base = TitleDescription()
    rep = NativeMetadata(base, ["Brand|Manufacturer", "Color|Colour", "Material", "Size"], 2)
    assert rep.meta_text(p) == "Metadata: category=Hand Tools > Pipe Clamps; brand=YUEPIN; material=steel; size=10pcs"
    assert rep.render(p).startswith("YUEPIN Clamp. Metadata: category=")  # right after the title
    assert NativeMetadata(base, ["Size"], 0, position="after").render(p).endswith("Metadata: size=10pcs")
    bare = Product(id="2", title="Plain", description="d")
    assert rep.render(bare) == base.render(bare)  # nothing to write: the plain text
    assert NativeMetadata(base, ["Material"], 0, max_value_chars=3).meta_text(p) == "Metadata: material=ste"
    with pytest.raises(ValueError):
        NativeMetadata(base, [], 0)


def test_trec_to_product_reads_native_metadata():
    from mdsearch.data.trec_product import to_product

    rec = {"title": "Clamp", "stars": "4.7 out of 5 stars", "ratings": "1,054 ratings", "price": "$1,009.99",
           "category": ["Tools", "Pipe Clamps"], "attrs": {"Brand": "YUEPIN"},
           "info": {"ASIN": "B0", "Best Sellers Rank": "#4", "Size": "\u200e10pcs"}, "docid": 4, "text": "t"}
    p = to_product(rec)
    assert (p.price, p.rating, p.n_reviews, p.category) == (1009.99, 4.7, 1054, "Tools > Pipe Clamps")
    assert p.extra == {"attrs": {"Brand": "YUEPIN"}, "info": {"Size": "10pcs"}}
    old = to_product({"docid": 1, "title": "x", "text": "y"})
    assert old.extra == {} and old.price is None and old.category is None


def test_profile_metadata():
    from mdsearch.metadata_profile import format_profile, profile

    ps = [Product(id=str(i), title="t", category="A > B", price=1.0,
                  extra={"attrs": {"Brand": "X"}, "info": {"Size": "1"} if i % 2 else {}}) for i in range(4)]
    r = profile(ps, min_coverage=0.6)
    assert r["n"] == 4 and r["have"]["category"] == 4 and r["category_depth"] == {2: 4}
    assert r["suggested_fields"] == ["Brand"] and "Size" in [k["key"] for k in r["keys"]]
    assert "fields: [\"Brand\"]" in format_profile(r)


def test_native_metadata_can_add_other_keys_up_to_a_limit():
    from mdsearch.representation import NativeMetadata

    p = Product(id="1", title="Clamp", description="d", category="Tools > Clamps",
                extra={"attrs": {"Brand": "YUEPIN", "Pattern": "plain", "Shape": "U"},
                       "info": {"Manufacturer": "WS", "Item Model Number": "X1"}})
    base = TitleDescription()
    strict = NativeMetadata(base, ["Brand|Manufacturer", "Color"], 1)
    assert strict.meta_text(p) == "Metadata: category=Clamps; brand=YUEPIN"  # Pattern, Shape, model number dropped
    rank = {"item model number": 0, "shape": 1, "pattern": 2}  # most common in the catalog first
    full = NativeMetadata(base, ["Brand|Manufacturer", "Color"], 1, other_keys=True, max_items=4, key_rank=rank)
    # preferred first, then the most common other keys; Manufacturer is an alias of a preferred field: not repeated
    assert full.meta_text(p) == "Metadata: category=Clamps; brand=YUEPIN; item_model_number=X1; shape=U"
    assert len(NativeMetadata(base, ["Brand"], 1, other_keys=True, key_rank=rank).items(p)) == 6  # no limit; Manufacturer is not an alias here
    assert NativeMetadata(base, ["Brand", "Pattern"], 1, max_items=2).meta_text(p) == "Metadata: category=Clamps; brand=YUEPIN"


def test_meta_reps_other_keys_rank_by_catalog_frequency(catalog, capsys):
    from mdsearch.experiment import _meta_reps, fill_max, fill_options

    for i, prod in enumerate(catalog.products):
        prod.extra = {"attrs": {"Brand": "B", "Shape": "S"} if i else {"Brand": "B", "Shape": "S", "Rare": "R"}}
    plain = {c: TitleDescription() for c in ("dense", "bm25", "rerank")}
    m = {"fields": ["Brand"], "other_keys": True}
    assert fill_options(m) == (True, 1) and fill_options({"fields": ["Brand"]}) == (False, None)
    assert fill_max({"representation": {"metadata": {"fields": ["Brand"], "category_levels": 2, "other_keys": True}}}) == 2
    reps = _meta_reps({"metadata": {**m, "max_items": 2}}, plain, catalog, True)
    assert reps.dense.items(catalog.products[0]) == [("brand", "B"), ("shape", "S")]  # Shape is commoner than Rare
    assert "other keys added up to 2 in all" in capsys.readouterr().out


def test_other_keys_order_does_not_depend_on_hash_order(catalog):
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = """
import sys; sys.path.insert(0, 'src')
from mdsearch.catalog import Catalog
from mdsearch.data.schema import Product
from mdsearch.experiment import _meta_reps
from mdsearch.representation import TitleDescription
ps = [Product(id=str(i), title='t', extra={'attrs': {'Brand': 'B', 'Shape': 'S', 'Pattern': 'P', 'Finish': 'F'}}) for i in range(3)]
plain = {c: TitleDescription() for c in ('dense', 'bm25', 'rerank')}
r = _meta_reps({'metadata': {'fields': ['Brand'], 'other_keys': True, 'max_items': 2}}, plain, Catalog(ps), False)
print(r.dense.meta_text(ps[0]))
"""
    root = Path(__file__).resolve().parents[1]
    outs = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout for seed in range(1, 7)}
    assert outs == {"Metadata: brand=B; finish=F\n"}  # ties: alphabetical, the same in every run

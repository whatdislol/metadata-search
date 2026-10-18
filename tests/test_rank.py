from __future__ import annotations

import numpy as np

from conftest import FakeEncoder
from mdsearch.rank import BM25Retriever, DenseRetriever, HybridRetriever, rrf, rrf_from_scores
from mdsearch.representation import TemplatedMetadata, TitleDescription, TitleReviews, get_representation


def test_rrf_over_id_lists():
    s = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert s["a"] == s["b"] and s["a"] > s["c"] > 0 and s["c"] > s["d"] or s["c"] == s["d"]
    assert set(s) == {"a", "b", "c", "d"}


def test_rrf_from_scores_ranks_among_unmasked_only():
    a = np.array([0.1, 0.9, 0.5, 0.7])
    mask = np.array([True, False, True, True])
    out = rrf_from_scores([a], k=0, mask=mask)
    assert out[1] == -np.inf
    # among unmasked, order by a: idx3 (0.7) rank1, idx2 rank2, idx0 rank3
    assert out[3] == 1.0 and out[2] == 0.5 and np.isclose(out[0], 1 / 3)


def test_bm25_prefers_matching_documents():
    r = BM25Retriever(["x", "y", "z"], ["lego bricks creative", "board game family", "baby rattle soft"])
    s = r.scores("creative lego")
    assert s.argmax() == 0


def test_dense_cache_roundtrip_and_invalidation(tmp_path):
    ids, texts = ["a", "b"], ["red car toy", "blue doll toy"]
    calls = []

    class Counting(FakeEncoder):
        def encode(self, texts, *, kind):
            calls.append(kind)
            return super().encode(texts, kind=kind)

    path = tmp_path / "emb.npz"
    d1 = DenseRetriever.build(ids, texts, Counting(), path)
    n = calls.count("document")
    d2 = DenseRetriever.build(ids, texts, Counting(), path)
    assert calls.count("document") == n  # served from cache
    assert np.allclose(d1.embeddings, d2.embeddings)
    DenseRetriever.build(ids, ["changed", "texts"], Counting(), path)
    assert calls.count("document") == n + 1  # fingerprint mismatch -> re-embed
    assert d1.scores("red car").argmax() == 0


def test_hybrid_combines_sources():
    ids, texts = ["a", "b", "c"], ["lego creative bricks", "monopoly board game", "baby soft rattle"]
    bm25 = BM25Retriever(ids, texts)
    dense = DenseRetriever.build(ids, texts, FakeEncoder())
    h = HybridRetriever([bm25, dense])
    assert h.scores("creative lego").argmax() == 0


def test_representations(products):
    p = products[0]
    assert TitleDescription().render(p).startswith("LEGO Classic creative bricks. Build")
    assert "Build anything creative" in TitleReviews().render(p)  # falls back to description
    p.reviews = ["Great bricks"]
    assert "Great bricks" in TitleReviews().render(p)

    templ = TemplatedMetadata(TitleDescription())
    out = templ.render(p)
    assert out.startswith("brand: LEGO | price: 29.99 | rating: 4.8 | min_age: 4 | LEGO Classic")
    assert templ.render(products[-1]).startswith("Mystery toy")  # no metadata -> base only
    assert get_representation("templated:title_reviews").name == "templated[title_reviews]"

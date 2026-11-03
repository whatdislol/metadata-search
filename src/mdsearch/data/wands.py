"""STEP 5 (not implemented): WANDS loader (Wayfair; human-labelled, rich metadata).

Target interface, same as the toys loader plus human relevance labels:

    products: list[Product]        # attributes -> extra, average rating/review count -> rating/n_reviews
    queries:  list[Query]          # text only (labels come from humans, not an LLM)
    qrels:    dict[qid, dict[product_id, int]]   # Exact=2, Partial=1, Irrelevant=0

Then evaluate with eval.metrics.ndcg_at_k so results are comparable with published numbers.
Data: https://github.com/wayfair/WANDS (download manually; not vendored here).
"""

from __future__ import annotations


def load_wands(*args, **kwargs):
    raise NotImplementedError("Step 5")


def prepare(cfg, *, limit=None, source=None):
    """Hook used by scripts/01_prepare.py. To implement it, read the raw WANDS files, build
    `queries` and `qrels` dicts and a record iterator, then call data.evalset.build_qrels_dataset
    (see data/trec_product.py for a complete example) and save the products and the EvalSet."""
    raise NotImplementedError("WANDS loader not written yet; see data/trec_product.py for the pattern")

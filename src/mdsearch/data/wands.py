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

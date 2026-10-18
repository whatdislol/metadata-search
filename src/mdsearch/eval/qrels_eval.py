"""Evaluate methods against human relevance judgments (qrels): no LLM judge needed.

Unjudged products count as not relevant (the usual convention). Metrics per query:
  P@k       share of the top k with grade >= rel_threshold
  AP@k      normalised by min(k, ALL relevant products of the query in the qrels), as the QAM paper
            defines it, so a method that misses relevant products is penalised
  nDCG@k    graded gains, ideal ranking from all graded products of the query in the qrels
  R@k       share of the query's relevant products found in the top k
Queries with no relevant product are skipped (and counted in meta).
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..pipeline import SearchPipeline
from ..runlog import describe_methods
from .metrics import bootstrap_ci, ndcg_at_k


def _stat(values: Sequence[float]) -> dict[str, float]:
    lo, hi = bootstrap_ci(values)
    return {"mean": float(np.mean(values)) if len(values) else float("nan"), "ci_lo": lo, "ci_hi": hi}


def ap_at_k(rel: Sequence[bool], k: int, n_relevant: int) -> float:
    if n_relevant == 0:
        return 0.0
    hits, acc = 0, 0.0
    for i, r in enumerate(rel[:k], start=1):
        if r:
            hits += 1
            acc += hits / i
    return acc / min(k, n_relevant)


def evaluate_qrels(
    methods: Mapping[str, SearchPipeline],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    ks: Sequence[int] = (5, 10, 20),
    rel_threshold: int = 1,
    out_dir: str | Path | None = None,
    meta: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    kmax = max(ks)
    ids = list(queries)
    n_total = len(ids)
    texts = [queries[q] for q in ids]

    results = {}
    for name, pipe in methods.items():
        t0 = time.time()
        log(f"searching {len(texts)} queries with {name} ...")
        results[name] = pipe.search_many(texts, k=kmax)
        log(f"  {name} done in {time.time() - t0:.0f}s")

    summary: dict[str, Any] = {}
    per_query: list[dict[str, Any]] = []
    skipped: set[str] = set()
    for name, res_list in results.items():
        cols: dict[str, list[float]] = {}
        for qid, res in zip(ids, res_list):
            judged = qrels.get(qid, {})
            n_rel = sum(g >= rel_threshold for g in judged.values())
            if n_rel == 0:
                skipped.add(qid)
                continue
            gains = [max(judged.get(h.id, 0), 0) for h in res.hits]  # TREC grade -1 (unjudgeable) counts as 0
            rel = [g >= rel_threshold for g in gains]
            ideal = [g for g in judged.values() if g > 0]
            row: dict[str, float] = {}
            for k in ks:
                row[f"P@{k}"] = sum(rel[:k]) / k
                row[f"AP@{k}"] = ap_at_k(rel, k, n_rel)
                row[f"nDCG@{k}"] = ndcg_at_k(gains, k, ideal)
                row[f"R@{k}"] = sum(rel[:k]) / n_rel
            for key, v in row.items():
                cols.setdefault(key, []).append(v)
            per_query.append(
                {
                    "method": name, "query_id": qid, "n_relevant": n_rel, "gains": gains,
                    "n_candidates": res.n_candidates, "hits": [h.id for h in res.hits],
                    "parsed": res.parsed.model_dump() if res.parsed else None, **row,
                }
            )
        summary[name] = {key: _stat(v) for key, v in cols.items()}

    report = {
        "meta": {
            "date": dt.datetime.now().isoformat(timespec="seconds"),
            "n_queries": n_total - len(skipped),
            "n_queries_skipped_no_relevant": len(skipped),
            "rel_threshold": rel_threshold,
            "filtering": describe_methods(methods),
            **(meta or {}),
        },
        "summary": summary,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with open(out / "per_query.jsonl", "w", encoding="utf-8") as f:
            for row in per_query:
                f.write(json.dumps(row) + "\n")
    return report

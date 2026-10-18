"""Run several search methods over the same queries, judge once, and compute metrics."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..pipeline import SearchPipeline, SearchResult
from ..runlog import describe_methods
from .judge import Judgment, RelevanceJudge
from .metrics import average_precision_at_k, bootstrap_ci, precision_at_k
from .querygen import GeneratedQuery


def evaluate(
    methods: Mapping[str, SearchPipeline],
    queries: Sequence[GeneratedQuery],
    judge: RelevanceJudge,
    *,
    ks: Sequence[int] = (3, 5, 10),
    out_dir: str | Path | None = None,
    meta: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    kmax = max(ks)
    texts = [q.text for q in queries]

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    results: dict[str, list[SearchResult]] = {}
    for name, pipe in methods.items():
        t0 = time.time()
        log(f"searching {len(texts)} queries with {name} ...")
        results[name] = pipe.search_many(texts, k=kmax)
        log(f"  {name} done in {time.time() - t0:.0f}s")

    # Judge every distinct (query, product) pair once, across all methods.
    pairs: dict[tuple[int, str], int] = {}
    pair_list: list[tuple[GeneratedQuery, str]] = []
    for res_list in results.values():
        for q, res in zip(queries, res_list):
            for h in res.hits:
                if (q.id, h.id) not in pairs:
                    pairs[(q.id, h.id)] = len(pair_list)
                    pair_list.append((q, h.id))
    log(f"judging {len(pair_list)} distinct (query, product) pairs (LLM; cached replies are reused) ...")
    t0 = time.time()
    judgments: list[Judgment] = judge.judge_many(pair_list)
    log(f"  judging done in {time.time() - t0:.0f}s")

    summary: dict[str, Any] = {}
    per_query: list[dict[str, Any]] = []
    for name, res_list in results.items():
        p_scores = {k: [] for k in ks}
        ap_scores = {k: [] for k in ks}
        for q, res in zip(queries, res_list):
            rels = [int(judgments[pairs[(q.id, h.id)]].relevant) for h in res.hits]
            for k in ks:
                p_scores[k].append(precision_at_k(rels, k))
                ap_scores[k].append(average_precision_at_k(rels, k))
            per_query.append(
                {
                    "method": name,
                    "query_id": q.id,
                    "rels": rels,
                    "n_candidates": res.n_candidates,
                    "parsed": res.parsed.model_dump() if res.parsed else None,
                    "hits": [h.id for h in res.hits],
                }
            )
        summary[name] = {
            f"P@{k}": _stat(p_scores[k]) for k in ks
        } | {f"mAP@{k}": _stat(ap_scores[k]) for k in ks}

    report = {
        "meta": {
            "date": dt.datetime.now().isoformat(timespec="seconds"),
            "n_queries": len(queries),
            "n_judged_pairs": len(pair_list),
            "n_judge_unusable": sum(j.soft_ok is None and j.hard_ok for j in judgments),
            "filtering": describe_methods(methods),
            **(meta or {}),
        },
        "summary": summary,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
        with open(out / "per_query.jsonl", "w", encoding="utf-8") as f:
            for row in per_query:
                f.write(json.dumps(row) + "\n")
        with open(out / "judgments.jsonl", "w", encoding="utf-8") as f:
            for (q, pid), j in zip(pair_list, judgments):
                f.write(
                    json.dumps(
                        {"query_id": q.id, "product_id": pid, "relevant": j.relevant,
                         "hard_ok": j.hard_ok, "soft_ok": j.soft_ok, "reason": j.reason}
                    )
                    + "\n"
                )
    return report


def _stat(values: Sequence[float]) -> dict[str, float]:
    lo, hi = bootstrap_ci(values)
    return {"mean": float(np.mean(values)) if len(values) else float("nan"), "ci_lo": lo, "ci_hi": hi}


def format_table(report: dict[str, Any], metric_keys: Sequence[str]) -> str:
    names = list(report["summary"])
    w = max([10] + [len(n) + 1 for n in names])
    header = f"{'method':{w}s}" + "".join(f"{m:>16s}" for m in metric_keys)
    rows = [header]
    for n in names:
        cells = "".join(f"{report['summary'][n][m]['mean'] * 100:15.2f}%" for m in metric_keys)
        rows.append(f"{n:{w}s}{cells}")
    return "\n".join(rows)

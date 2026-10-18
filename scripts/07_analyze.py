"""Step 5b: look inside the evaluation results (no LLM, no models, runs in seconds).

    python scripts/07_analyze.py [--style specific|broad] [--k 5] [--ref qam] [--misses 15] [--parse-errors 15]

--misses N also lists N queries where the reference method lacks the source product in its top-10,
with the query, the true and parsed constraints, and whether the parsed filters exclude the source.

Prints, per method (top-k lists):
  * where the non-relevant results come from: violate a hard constraint vs fit the constraints but
    not the subjective part of the query (judged by the LLM)
  * precision among results that satisfy the hard constraints (relevant / (relevant + soft fail)):
    shows ranking quality on the subjective part, separate from the filter's benefit
  * "source hit": how often the product the query was written from is in the top-k. For
    specific (known-item) queries this is the most direct success measure
  * paired bootstrap of P@k(ref) - P@k(method) over queries (is the gap bigger than query noise?)
  * how many queries have any relevant result at all, and how many relevant products the union of
    all methods found per query (a ceiling for P@k)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mdsearch.config import load_experiment_config, resolve_path
from mdsearch.data.schema import load_products
from mdsearch.eval.querygen import load_queries
from mdsearch.filtering import satisfies
from mdsearch.query.schema import Filters
from mdsearch.experiment import style_paths


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def paired_bootstrap(diff: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _norm(s: str | None) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _upper_age(f: dict) -> float | None:
    return f.get("age_max") if f.get("age_max") is not None else f.get("age_min")


def filter_diff(truth: dict, parsed: dict) -> tuple[list[str], list[str]]:
    """(constraints the parser missed or got wrong, constraints it added that the query lacks)."""
    missing: list[str] = []
    extra: list[str] = []
    tb, pb = _norm(truth.get("brand")), _norm(parsed.get("brand"))
    if tb and not (pb and (tb in pb or pb in tb)):
        missing.append("brand")
    if pb and not tb:
        extra.append("brand")
    for key in ("price_min", "price_max", "rating_min"):
        t, v = truth.get(key), parsed.get(key)
        if t is not None and (v is None or abs(t - v) > 1e-6):
            missing.append(key)
        if v is not None and t is None:
            extra.append(key)
    tu, pu = _upper_age(truth), _upper_age(parsed)
    if tu is not None and (pu is None or abs(tu - pu) > 1e-6):
        missing.append("age")
    if pu is not None and tu is None:
        extra.append("age")
    return missing, extra


def analyze(
    per_query: list[dict],
    judgments: list[dict],
    k: int,
    ref: str,
    queries: list | None = None,
) -> str:
    """`queries`: the GeneratedQuery list (ground-truth filters and source products), optional."""
    sources = {q.id: q.source_id for q in queries} if queries else None
    truth = {q.id: q.filters.model_dump() for q in queries} if queries else None
    j = {(r["query_id"], r["product_id"]): r for r in judgments}
    methods = list(dict.fromkeys(r["method"] for r in per_query))
    by_method: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in per_query:
        by_method[r["method"]][r["query_id"]] = r
    qids = sorted(by_method[methods[0]])
    out: list[str] = []

    out.append(f"== composition of the top-{k} lists (share of the k slots; empty slots = list shorter than k)")
    out.append(
        f"{'method':10s}{'relevant':>10s}{'soft fail':>11s}{'hard fail':>11s}{'empty':>8s}"
        f"{'prec|hard ok':>14s}"
    )
    for m in methods:
        rel = soft = hard = 0
        for q in qids:
            hits = by_method[m][q]["hits"][:k]
            for pid in hits:
                r = j[(q, pid)]
                if r["relevant"]:
                    rel += 1
                elif not r["hard_ok"]:
                    hard += 1
                else:
                    soft += 1
        total = k * len(qids)
        empty = total - rel - soft - hard
        cond = rel / (rel + soft) if rel + soft else float("nan")
        out.append(
            f"{m:10s}{rel / total:10.1%}{soft / total:11.1%}{hard / total:11.1%}{empty / total:8.1%}"
            f"{cond:14.1%}"
        )

    if sources:
        out.append(f"\n== source hit: share of queries whose source product is in the top-k")
        ks = sorted({1, 3, k, 10})
        out.append(f"{'method':10s}" + "".join(f"{'top-' + str(x):>9s}" for x in ks))
        for m in methods:
            cells = ""
            for x in ks:
                hit = np.mean([sources[q] in by_method[m][q]["hits"][:x] for q in qids if q in sources])
                cells += f"{hit:9.1%}"
            out.append(f"{m:10s}{cells}")

    if truth and by_method[ref] and by_method[ref][qids[0]].get("parsed"):
        out.append(f"\n== {ref}: did the parser recover the query's constraints?")
        n_ok = n_missing = n_extra = 0
        field_missing: dict[str, int] = defaultdict(int)
        field_extra: dict[str, int] = defaultdict(int)
        bad_parse: set[int] = set()
        for q in qids:
            parsed = by_method[ref][q]["parsed"]["filters"]
            miss, ext = filter_diff(truth[q], parsed)
            n_missing += bool(miss)
            n_extra += bool(ext)
            n_ok += not miss and not ext
            for f_ in miss:
                field_missing[f_] += 1
            for f_ in ext:
                field_extra[f_] += 1
            if miss or ext:
                bad_parse.add(q)
        nq = len(qids)
        out.append(f"exactly the ground-truth constraints: {n_ok / nq:.1%}   "
                   f"missed/wrong >=1: {n_missing / nq:.1%}   added >=1 extra: {n_extra / nq:.1%}")
        out.append(f"  missed/wrong by field: {dict(field_missing) or '-'}")
        out.append(f"  extra by field:        {dict(field_extra) or '-'}")
        if sources:
            miss_src = [q for q in qids if sources[q] not in by_method[ref][q]["hits"][:10]]
            parser_caused = [q for q in miss_src if q in bad_parse]
            out.append(f"  {ref} misses the source product in its top-10 for {len(miss_src)} queries: "
                       f"{len(parser_caused)} had a wrong/extra filter, "
                       f"{len(miss_src) - len(parser_caused)} had the right filters (ranking or rerank depth)")

    out.append(f"\n== paired difference in P@{k}: {ref} minus method (95% bootstrap CI over queries)")
    pk = {m: np.array([sum(by_method[m][q]["rels"][:k]) / k for q in qids]) for m in methods}
    for m in methods:
        if m == ref:
            continue
        d = pk[ref] - pk[m]
        lo, hi = paired_bootstrap(d)
        verdict = "clear" if lo > 0 or hi < 0 else "NOT distinguishable from noise"
        out.append(f"{m:10s}{d.mean():+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {verdict}")

    out.append("\n== query difficulty (relevant products found in the top-k of ANY method)")
    union = []
    for q in qids:
        found = {pid for m in methods for pid, r in zip(by_method[m][q]["hits"][:k], by_method[m][q]["rels"][:k]) if r}
        union.append(len(found))
    union = np.array(union)
    out.append(f"queries with 0 relevant found by any method: {(union == 0).sum()} of {len(qids)}")
    out.append(f"queries with 1-2: {((union >= 1) & (union <= 2)).sum()}   3-{k}: {(union > 2).sum()}")
    out.append(f"mean relevant products found per query (union): {union.mean():.2f}  "
               f"-> P@{k} of a perfect system on these would be about {np.minimum(union, k).mean() / k:.1%}")
    return "\n".join(out)


def explain_misses(
    per_query: list[dict], queries: list, products: dict, ref: str, show: int = 15, k: int = 10
) -> str:
    """Why does `ref` lack the source product in its top-k? Three causes:
    excluded by the parsed filters / in the pool but ranked low / in a pool larger than the
    rerank depth (100), where the semantic stage may have cut it before the cross-encoder."""
    rows = {r["query_id"]: r for r in per_query if r["method"] == ref}
    cats: dict[str, list[int]] = {"excluded": [], "ranked low (pool<=100)": [], "ranked low (pool>100)": []}
    detail: dict[int, str] = {}
    for q in queries:
        r = rows.get(q.id)
        if r is None or q.source_id in r["hits"][:k]:
            continue
        parsed = (r.get("parsed") or {}).get("filters") or {}
        src = products.get(q.source_id)
        in_pool = src is not None and satisfies(src, Filters(**parsed))
        cat = "excluded" if not in_pool else (
            "ranked low (pool<=100)" if r["n_candidates"] <= 100 else "ranked low (pool>100)"
        )
        cats[cat].append(q.id)
        miss, ext = filter_diff(q.filters.model_dump(), parsed)
        detail[q.id] = (
            f"[{q.id}] {cat}; pool={r['n_candidates']}\n"
            f"    query : {q.text[:150]}\n"
            f"    truth : {q.filters.active()}\n"
            f"    parsed: { {a: b for a, b in parsed.items() if b is not None} }"
            f"   missed={miss} extra={ext}\n"
            f"    source: {(src.title[:70] if src else '?')}  (brand={src.brand if src else '?'}, "
            f"price={src.price if src else '?'}, rating={src.rating if src else '?'}, "
            f"min_age={src.min_age if src else '?'})"
        )
    n = sum(len(v) for v in cats.values())
    lines = [f"\n== why {ref} lacks the source product in its top-{k}: {n} queries"]
    lines += [f"  {name}: {len(ids)}" for name, ids in cats.items()]
    shown = 0
    for name, ids in cats.items():
        for qid in ids:
            if shown >= show:
                break
            lines.append(detail[qid])
            shown += 1
    return "\n".join(lines)


def explain_parse_errors(per_query: list[dict], queries: list, ref: str, show: int = 15) -> str:
    """List queries where the parser missed or got wrong a ground-truth constraint."""
    rows = {r["query_id"]: r for r in per_query if r["method"] == ref}
    lines = []
    for q in queries:
        r = rows.get(q.id)
        if r is None or not r.get("parsed"):
            continue
        parsed = r["parsed"]["filters"]
        miss, ext = filter_diff(q.filters.model_dump(), parsed)
        if miss:
            lines.append(
                f"[{q.id}] missed={miss}\n    query : {q.text[:150]}\n"
                f"    truth : {q.filters.active()}\n"
                f"    parsed: { {a: b for a, b in parsed.items() if b is not None} }"
            )
    head = f"\n== {ref}: {len(lines)} queries where a ground-truth constraint was missed or wrong"
    return "\n".join([head] + lines[:show])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--style", default="specific", choices=["specific", "broad"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ref", default="qam")
    ap.add_argument("--misses", type=int, default=0, help="list N source-product misses of --ref")
    ap.add_argument("--parse-errors", type=int, default=0, help="list N queries whose constraints --ref's parser missed")
    args = ap.parse_args()
    cfg = load_experiment_config(args.config)
    queries_path, d = style_paths(cfg, args.style)
    queries = load_queries(queries_path) if queries_path.exists() else None
    print(f"query style: {args.style}   results: {d}\n")
    per_query = read_jsonl(d / "per_query.jsonl")
    print(analyze(per_query, read_jsonl(d / "judgments.jsonl"), args.k, args.ref, queries))
    if args.parse_errors and queries:
        print(explain_parse_errors(per_query, queries, args.ref, show=args.parse_errors))
    if args.misses and queries:
        products = {p.id: p for p in load_products(resolve_path(cfg["dataset"]["processed"]))}
        print(explain_misses(per_query, queries, products, args.ref, show=args.misses))


if __name__ == "__main__":
    main()
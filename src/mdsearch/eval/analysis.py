"""Look inside evaluation results (no LLM, no models; runs in seconds). Used by scripts/05_analyze.py.

llm_judge results (synthetic queries with known constraints):

  * where the non-relevant results come from: violate a hard constraint vs fit the constraints but
    not the subjective part of the query (judged by the LLM)
  * precision among results that satisfy the hard constraints (relevant / (relevant + soft fail)):
    shows ranking quality on the subjective part, separate from the filter's benefit
  * "source hit": how often the product the query was written from is in the top-k. For
    specific (known-item) queries this is the most direct success measure
  * paired bootstrap of P@k(ref) - P@k(method) over queries (is the gap bigger than query noise?)
  * how many queries have any relevant result at all, and how many relevant products the union of
    all methods found per query (a ceiling for P@k)

qrels results (datasets with human judgments): see analyze_qrels.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import resolve_path
from ..data.schema import load_products
from ..experiment import style_paths
from ..filtering import satisfies
from ..query.schema import Filters
from .dispatch import eval_mode, results_dir
from .querygen import load_queries


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


def analyze_qrels(per_query: list[dict], k: int = 10, ref: str | None = None, metric: str = "nDCG") -> str:
    """Per-method summary for qrels evaluations, plus a paired bootstrap against `ref`.

    `pool` is how many products were left after filtering (the whole pool for unfiltered methods), so
    it shows how aggressive a hard filter was. "rel kept" is the average share of a query's relevant
    products that survived the filter (100% = the filter removed no relevant product). "no relevant" counts queries with no relevant product
    in the top k."""
    by_method: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in per_query:
        by_method[r["method"]][r["query_id"]] = r
    methods = list(by_method)
    if not methods:
        return "no results"
    ref = ref if ref in by_method else methods[0]
    pk, mk, rk = f"P@{k}", f"{metric}@{k}", f"R@{k}"
    sample = next(iter(by_method[ref].values()))
    for key in (pk, mk, rk):
        if key not in sample:
            avail = sorted(x for x in sample if "@" in x)
            raise SystemExit(f"{key} not in the results; available: {avail}")
    qids = sorted(set.intersection(*(set(by_method[m]) for m in methods)))
    out = [f"== {len(qids)} queries, k={k}, reference = {ref}",
           f"{'method':16s}{pk:>9s}{mk:>11s}{rk:>9s}{'no relevant':>13s}{'pool mean':>11s}{'pool median':>13s}{'rel kept':>10s}"
           f"   {metric}({ref}) - {metric}(method)  [95% CI]   wins/losses"]
    for m in methods:
        rows = [by_method[m][q] for q in qids]
        p = np.mean([x[pk] for x in rows]) * 100
        g = np.mean([x[mk] for x in rows]) * 100
        rc = np.mean([x[rk] for x in rows]) * 100
        none = sum(x[pk] == 0 for x in rows)
        pools = np.array([x["n_candidates"] for x in rows], dtype=float)
        diff = np.array([by_method[ref][q][mk] - by_method[m][q][mk] for q in qids])
        if m == ref:
            cmp = "-"
        else:
            lo, hi = paired_bootstrap(diff)
            cmp = f"{diff.mean() * 100:+6.2f}  [{lo * 100:+.2f}, {hi * 100:+.2f}]   {int((diff > 0).sum())}/{int((diff < 0).sum())}"
        kept = [x["rel_retained"] for x in rows if x.get("rel_retained") is not None]
        kept_s = f"{np.mean(kept) * 100:9.1f}%" if kept else f"{'-':>10s}"
        out.append(f"{m:16s}{p:8.2f}%{g:10.2f}%{rc:8.2f}%{none:13d}{pools.mean():11.0f}{np.median(pools):13.0f}{kept_s}   {cmp}")
    return "\n".join(out)


def tag_report(per_query: list[dict], k: int = 10, ref: str | None = None, metric: str = "nDCG",
               attr_kinds: tuple[str, ...] | None = None, examples: int = 12) -> str:
    """What the generated tags looked like, and results split by whether a query has attribute tags.

    The tags come from the first method in the results that generated any. Attribute tags are those whose
    role (kind) is in `attr_kinds` (default ATTRIBUTE_KINDS): brand, colour, material, size, audience,
    price, feature. Tags that restate the product type cannot narrow anything useful."""
    from ..query.schema import ATTRIBUTE_KINDS

    attr = set(attr_kinds or ATTRIBUTE_KINDS)
    tags_by_q: dict[str, list[dict]] = {}
    text_by_q: dict[str, str] = {}
    for r in per_query:
        p = r.get("parsed")
        if p and p.get("tags") and r["query_id"] not in tags_by_q:
            tags_by_q[r["query_id"]] = p["tags"]
            text_by_q[r["query_id"]] = p.get("semantic_text", "")
    all_q = sorted({r["query_id"] for r in per_query})
    if not tags_by_q:
        return "no generated tags in these results"
    kinds = Counter(t.get("kind") or "(none)" for ts in tags_by_q.values() for t in ts)
    with_attr = {q for q, ts in tags_by_q.items() if any(t.get("kind") in attr for t in ts)}
    out = [
        f"== generated tags: {len(tags_by_q)} of {len(all_q)} queries got at least one tag "
        f"({sum(len(v) for v in tags_by_q.values())} tags); {len(with_attr)} queries have an attribute tag",
        "tag roles: " + ", ".join(f"{k_} {n}" for k_, n in kinds.most_common()),
        "examples (semantic text | tags):",
    ]
    for q in sorted(tags_by_q)[:examples]:
        out.append(f"  {q}: {text_by_q[q]!r} | " + "; ".join(f"{t['text']} [{t.get('kind') or '-'}]" for t in tags_by_q[q]))

    by_method: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in per_query:
        by_method[r["method"]][r["query_id"]] = r
    methods = list(by_method)
    ref = ref if ref in by_method else methods[0]
    mk = f"{metric}@{k}"
    if mk not in next(iter(by_method[ref].values())):
        return "\n".join(out)
    groups = {"with attribute tag": sorted(with_attr), "without": sorted(set(all_q) - with_attr)}
    out.append(f"\n{metric}@{k} by query group (difference = {ref} minus method; positive = method worse)")
    for label, qs in groups.items():
        qs = [q for q in qs if all(q in by_method[m] for m in methods)]
        if not qs:
            continue
        out.append(f"-- {label}: {len(qs)} queries")
        for m in methods:
            g = np.array([by_method[m][q][mk] for q in qs])
            d = np.array([by_method[ref][q][mk] - by_method[m][q][mk] for q in qs])
            cmp = "-" if m == ref else "{:+6.2f}  [{:+.2f}, {:+.2f}]".format(d.mean() * 100, *(x * 100 for x in paired_bootstrap(d)))
            out.append(f"   {m:18s}{g.mean() * 100:8.2f}%   {cmp}")
    return "\n".join(out)


def run_analysis(
    cfg: dict,
    *,
    style: str = "specific",
    k: int = 5,
    ref: str = "qam",
    misses: int = 0,
    parse_errors: int = 0,
    out_suffix: str = "",
) -> tuple[str, dict]:
    """Text report for the dataset's results (mode from eval.mode) and the filtering recorded with them."""
    d = results_dir(cfg, style, out_suffix)
    if not (d / "summary.json").exists():
        raise SystemExit(f"{d / 'summary.json'} not found. Run scripts/04_evaluate.py first.")
    meta = json.loads((d / "summary.json").read_text(encoding="utf-8")).get("meta", {})
    per_query = read_jsonl(d / "per_query.jsonl")
    if eval_mode(cfg) == "qrels":
        text = analyze_qrels(per_query, k, ref) + "\n\n" + tag_report(per_query, k, ref)
        return f"results: {d}\n\n" + text, meta.get("filtering", {})

    queries_path, _ = style_paths(cfg, style)
    queries = load_queries(queries_path) if queries_path.exists() else None
    parts = [f"query style: {style}   results: {d}\n",
             analyze(per_query, read_jsonl(d / "judgments.jsonl"), k, ref, queries)]
    if parse_errors and queries:
        parts.append(explain_parse_errors(per_query, queries, ref, show=parse_errors))
    if misses and queries:
        products = {p.id: p for p in load_products(resolve_path(cfg["dataset"]["processed"]))}
        parts.append(explain_misses(per_query, queries, products, ref, show=misses))
    return "\n".join(parts), meta.get("filtering", {})

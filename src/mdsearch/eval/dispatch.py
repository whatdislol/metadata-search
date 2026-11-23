"""One entry point for evaluation, whatever the dataset: the config's `eval.mode` decides how
relevance is obtained.

  llm_judge  synthetic queries (scripts/03_generate_queries.py) whose constraints are known; an LLM
             judges the returned products (eval/judge.py). Metrics: P@k, mAP@k.
  qrels      a dataset with human relevance judgments (the EvalSet written by scripts/01_prepare.py).
             No judge. Metrics: P, AP, nDCG, recall at k.

Both write summary.json with `meta.filtering` (hard / soft / none per method).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..catalog import Catalog
from ..config import resolve_path
from ..data.evalset import EvalSet
from ..experiment import style_paths
from ..llm import LLM
from ..pipeline import SearchPipeline
from .judge import RelevanceJudge
from .qrels_eval import METRICS, evaluate_qrels, metric_ks
from .querygen import load_queries
from .runner import evaluate, format_table

MODES = ("llm_judge", "qrels")


def eval_mode(cfg: Mapping[str, Any]) -> str:
    mode = cfg.get("eval", {}).get("mode", "llm_judge")
    if mode not in MODES:
        raise SystemExit(f"eval.mode must be one of {MODES}, got {mode!r}")
    return mode


def default_methods(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    m = cfg.get("eval", {}).get("methods")
    return tuple(m) if m else ("keyword", "semantic", "rerank", "hybrid", "qam")


def results_dir(cfg: Mapping[str, Any], style: str = "specific", out_suffix: str = "") -> Path:
    d = style_paths(dict(cfg), style)[1] if eval_mode(cfg) == "llm_judge" else resolve_path(cfg["eval"]["out_dir"])
    return d.with_name(d.name + out_suffix) if out_suffix else d


def select_by_fill(queries: Mapping[str, str], parser: Any, min_keys: int) -> tuple[dict[str, str], dict[str, int]]:
    """Keeps the queries for which the `_fill` LLM filled at least `min_keys` metadata keys.

    `parser` is the TagParser the `_fill` methods use (same prompt, so its replies are cached and shared).
    Returns (kept queries, {query_id: keys filled} for every query)."""
    from ..query.tag_parser import filled_count

    ids = list(queries)
    parsed = parser.parse_many([queries[q] for q in ids])
    n = {q: filled_count(p.expansion) for q, p in zip(ids, parsed)}
    return {q: queries[q] for q in ids if n[q] >= min_keys}, n


def qrels_queries(cfg: Mapping[str, Any], limit: int | None = None, config_name: str = "") -> tuple[EvalSet, dict[str, str]]:
    """The eval set and its queries (the first `limit` of them) for eval.mode qrels."""
    path = resolve_path(cfg["dataset"]["eval_set"])
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/01_prepare.py --config {config_name or '<config>'}")
    ev = EvalSet.load(path)
    return ev, (dict(list(ev.queries.items())[:limit]) if limit else dict(ev.queries))


@dataclass
class Outcome:
    report: dict[str, Any]
    out_dir: Path
    tables: list[list[str]]  # metric keys, one printed table per entry


def run_evaluation(
    cfg: dict[str, Any],
    catalog: Catalog,
    methods: Mapping[str, SearchPipeline],
    *,
    llm: LLM | None,
    config_name: str = "",
    style: str = "specific",
    limit: int | None = None,
    out_suffix: str = "",
    verbose: bool = True,
    fill_parser: Any = None,
    min_filled_keys: int | None = None,
) -> Outcome:
    ecfg = cfg["eval"]
    mode = eval_mode(cfg)
    ks = ecfg["ks"] if isinstance(ecfg["ks"], dict) else tuple(ecfg["ks"])  # list = all metrics, mapping = per metric
    out_dir = results_dir(cfg, style, out_suffix)
    meta = {"config": config_name, "eval_mode": mode, "encoders": cfg.get("encoders"),
            "representation": cfg.get("representation"), "n_products": len(catalog)}

    if min_filled_keys and mode != "qrels":
        raise SystemExit("eval.min_filled_keys only works with eval.mode qrels")
    if mode == "llm_judge":
        if llm is None:
            raise SystemExit("eval.mode llm_judge needs an LLM (the judge)")
        queries_path, _ = style_paths(cfg, style)
        if not queries_path.exists():
            raise SystemExit(f"{queries_path} not found. Run scripts/03_generate_queries.py --style {style}")
        queries = load_queries(queries_path)
        if limit:
            queries = queries[:limit]
        judge = RelevanceJudge(
            llm, catalog, price_tol=ecfg.get("price_tol", 0.2), rating_tol=ecfg.get("rating_tol", 0.2)
        )
        kl = tuple(ks if not isinstance(ks, dict) else ks.get("P", ()))
        report = evaluate(
            methods, queries, judge, ks=kl, out_dir=out_dir, verbose=verbose,
            meta={**meta, "llm_provider": llm.provider, "llm_model": llm.model, "query_style": style},
        )
        return Outcome(report, out_dir, [[f"P@{k}" for k in kl] + [f"mAP@{k}" for k in kl]])

    ev, queries_d = qrels_queries(cfg, limit, config_name)
    if min_filled_keys:
        if fill_parser is None:
            raise SystemExit("eval.min_filled_keys needs the _fill parser (an LLM and representation.metadata keys)")
        total = len(queries_d)
        queries_d, filled = select_by_fill(queries_d, fill_parser, min_filled_keys)
        meta["query_filter"] = {"min_filled_keys": min_filled_keys, "kept": len(queries_d), "of": total,
                                "filled_per_query": filled}
        if not queries_d:
            raise SystemExit(f"no query has {min_filled_keys} or more metadata keys filled")
    mks = metric_ks(ks)
    report = evaluate_qrels(
        methods, queries_d, ev.qrels, ks=ks, rel_threshold=ecfg.get("rel_threshold", ev.rel_threshold),
        out_dir=out_dir, verbose=verbose,
        meta={**meta, "llm_model": llm.model if llm else None,
              "tags": cfg.get("retrieval", {}).get("tags")},
    )
    return Outcome(report, out_dir, [[f"{m}@{k}" for k in mks[m]] for m in METRICS])


def format_outcome(outcome: Outcome) -> str:
    return "\n\n".join(format_table(outcome.report, keys) for keys in outcome.tables)

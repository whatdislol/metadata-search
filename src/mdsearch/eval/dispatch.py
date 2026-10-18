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
from .qrels_eval import evaluate_qrels
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
) -> Outcome:
    ecfg = cfg["eval"]
    mode = eval_mode(cfg)
    ks = tuple(ecfg["ks"])
    out_dir = results_dir(cfg, style, out_suffix)
    meta = {"config": config_name, "eval_mode": mode, "encoders": cfg.get("encoders"),
            "representation": cfg.get("representation"), "n_products": len(catalog)}

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
        report = evaluate(
            methods, queries, judge, ks=ks, out_dir=out_dir, verbose=verbose,
            meta={**meta, "llm_provider": llm.provider, "llm_model": llm.model, "query_style": style},
        )
        return Outcome(report, out_dir, [[f"P@{k}" for k in ks] + [f"mAP@{k}" for k in ks]])

    path = resolve_path(cfg["dataset"]["eval_set"])
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/01_prepare.py --config {config_name or '<config>'}")
    ev = EvalSet.load(path)
    queries_d = dict(list(ev.queries.items())[:limit]) if limit else ev.queries
    report = evaluate_qrels(
        methods, queries_d, ev.qrels, ks=ks, rel_threshold=ecfg.get("rel_threshold", ev.rel_threshold),
        out_dir=out_dir, verbose=verbose,
        meta={**meta, "llm_model": llm.model if llm else None,
              "tags": cfg.get("retrieval", {}).get("tags")},
    )
    return Outcome(report, out_dir, [[f"{m}@{k}" for k in ks] for m in ("P", "AP", "nDCG", "R")])


def format_outcome(outcome: Outcome) -> str:
    return "\n\n".join(format_table(outcome.report, keys) for keys in outcome.tables)

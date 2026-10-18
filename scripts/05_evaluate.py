"""Step 4b: compare keyword / semantic / rerank / hybrid / QAM on the generated queries.

    python scripts/05_evaluate.py [--style specific|broad] [--provider uofg] [--limit 50]

Results go to results/qam_toys (specific) or results/qam_toys_broad (broad): (summary.json, per_query.jsonl, judgments.jsonl).
Before trusting small differences, hand-check ~50 lines of judgments.jsonl.
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config
from mdsearch.eval.judge import RelevanceJudge
from mdsearch.eval.querygen import load_queries
from mdsearch.eval.runner import evaluate, format_table
from mdsearch.experiment import load_catalog, make_methods, style_paths
from mdsearch.llm import get_llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--style", default="specific", choices=["specific", "broad"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    ecfg = cfg["eval"]
    catalog = load_catalog(cfg)
    queries_path, out_dir = style_paths(cfg, args.style)
    if not queries_path.exists():
        raise SystemExit(f"{queries_path} not found. Run 04_generate_queries.py --style {args.style}")
    queries = load_queries(queries_path)
    if args.limit:
        queries = queries[: args.limit]

    llm = get_llm(args.provider)
    methods = make_methods(cfg, catalog, llm)
    judge = RelevanceJudge(
        llm, catalog, price_tol=ecfg.get("price_tol", 0.2), rating_tol=ecfg.get("rating_tol", 0.2)
    )

    ks = tuple(ecfg["ks"])
    report = evaluate(
        methods,
        queries,
        judge,
        ks=ks,
        out_dir=out_dir,
        meta={"llm_provider": llm.provider, "llm_model": llm.model, "config": args.config, "query_style": args.style,
              "encoders": cfg["encoders"], "representation": cfg["representation"]},
    )
    keys = [f"P@{k}" for k in ks] + [f"mAP@{k}" for k in ks]
    print(format_table(report, keys))
    print(f"\nparser fallbacks (no filters used): {methods['qam'].parser.n_failures}")


if __name__ == "__main__":
    main()

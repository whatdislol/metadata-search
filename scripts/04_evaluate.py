"""Step 4: compare methods on a dataset's queries (works for every dataset).

    python scripts/04_evaluate.py --config qam_toys [--style specific|broad]
    python scripts/04_evaluate.py --config trec_product
    python scripts/04_evaluate.py --config qam_toys --methods rerank,qam,qam_tags_hard,qam_tags_soft --out-suffix _tags

How relevance is obtained depends on eval.mode in the config:
  llm_judge  queries from script 03, relevance judged by an LLM (P@k, mAP@k)
  qrels      queries and human judgments from script 01 (P, AP, nDCG, recall at k); no judge

--methods defaults to eval.methods of the config. Results go to the config's eval.out_dir (add
--out-suffix to keep earlier results); summary.json lists the filter type of every method under
meta.filtering (hard / soft / none). The printed output is also saved under results/runs/.
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config
from mdsearch.eval.dispatch import default_methods, eval_mode, format_outcome, run_evaluation
from mdsearch.experiment import load_catalog, make_methods
from mdsearch.llm import get_llm
from mdsearch.runlog import describe_methods, filtering_label, record_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--style", default="specific", choices=["specific", "broad"],
                    help="query set for eval.mode llm_judge")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries")
    ap.add_argument("--out-suffix", default="", help="append to the results folder name, e.g. _tags")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    only = tuple(m.strip() for m in args.methods.split(",")) if args.methods else default_methods(cfg)
    needs_llm = eval_mode(cfg) == "llm_judge" or any(m.startswith("qam") for m in only)

    with record_run("04_evaluate", config=args.config, args=vars(args), label=args.style if eval_mode(cfg) == "llm_judge" else "") as run:
        llm = get_llm(args.provider) if needs_llm else None
        catalog = load_catalog(cfg)
        methods = make_methods(cfg, catalog, llm, only=only)
        run.set_filtering(describe_methods(methods))
        outcome = run_evaluation(
            cfg, catalog, methods, llm=llm, config_name=args.config, style=args.style,
            limit=args.limit, out_suffix=args.out_suffix,
        )
        print("\n" + format_outcome(outcome))
        print(f"\nqueries evaluated: {outcome.report['meta']['n_queries']}   products in the pool: {len(catalog)}")
        for name in ("qam", "qam_tags", "qam_tags_hard", "qam_tags_soft"):
            if name in methods:
                print(f"parser fallbacks, {name} (no filters/tags used): {methods[name].parser.n_failures}")
        print(f"\nresults saved in {outcome.out_dir}")
        print("filtering per method:")
        for name, desc in outcome.report["meta"]["filtering"].items():
            print(f"  {name:16s} {filtering_label(desc)}")


if __name__ == "__main__":
    main()

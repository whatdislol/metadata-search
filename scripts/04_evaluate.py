"""Step 4: compare methods on a dataset's queries (works for every dataset).

    python scripts/04_evaluate.py --config qam_toys [--style specific|broad]
    python scripts/04_evaluate.py --config trec_product
    python scripts/04_evaluate.py --config qam_toys --methods rerank,qam,rerank_hard,rerank_soft,qam_soft --out-suffix _tags

How relevance is obtained depends on eval.mode in the config:
  llm_judge  queries from script 03, relevance judged by an LLM (P@k, mAP@k)
  qrels      queries and human judgments from script 01 (P, AP, nDCG, recall at k); no judge

--methods defaults to eval.methods of the config. Results go to the config's eval.out_dir (add
--out-suffix to keep earlier results); --min-filled-keys N (or eval.min_filled_keys) evaluates every method
on only the queries whose `_fill` metadata has at least N keys filled; summary.json lists the filter type of every method under
meta.filtering (hard / soft / none). The printed output is also saved under results/runs/.
"""

from __future__ import annotations

import argparse
from collections import Counter

from mdsearch.config import load_experiment_config
from mdsearch.eval.dispatch import default_methods, eval_mode, format_outcome, qrels_queries, run_evaluation, select_by_fill
from mdsearch.experiment import fill_keys, fill_max, load_catalog, make_methods
from mdsearch.llm import get_llm
from mdsearch.methods import uses_llm
from mdsearch.query.tag_parser import TagParser
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
    ap.add_argument("--min-filled-keys", type=int, default=None,
                    help="only evaluate queries whose _fill metadata has at least N keys (overrides eval.min_filled_keys)")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    only = tuple(m.strip() for m in args.methods.split(",")) if args.methods else default_methods(cfg)
    min_keys = args.min_filled_keys if args.min_filled_keys is not None else cfg["eval"].get("min_filled_keys")
    needs_llm = eval_mode(cfg) == "llm_judge" or any(uses_llm(m) for m in only) or bool(min_keys)

    with record_run("04_evaluate", config=args.config, args=vars(args), label=args.style if eval_mode(cfg) == "llm_judge" else "") as run:
        llm = get_llm(args.provider) if needs_llm else None
        catalog = load_catalog(cfg)
        fill_parser = None
        if min_keys:
            keys = fill_keys(cfg)
            if not keys:
                raise SystemExit("min_filled_keys needs representation.metadata (fields / category_levels)")
            if eval_mode(cfg) != "qrels":
                raise SystemExit("min_filled_keys only works with eval.mode qrels")
            fill_parser = TagParser(llm, fill_keys=keys, fill_max=fill_max(cfg))  # the same parser (and cached replies) as the _fill methods
            # the LLM step runs BEFORE the (slow) embedding, so a failure here costs nothing; its replies are cached
            _, qs = qrels_queries(cfg, args.limit, args.config)
            kept, _ = select_by_fill(qs, fill_parser, min_keys)
            print(f"query filter: {len(kept)} of {len(qs)} queries have >= {min_keys} metadata keys filled")
            if not kept:
                raise SystemExit(f"no query has {min_keys} or more metadata keys filled")
        methods = make_methods(cfg, catalog, llm, only=only)
        run.set_filtering(describe_methods(methods))
        outcome = run_evaluation(
            cfg, catalog, methods, llm=llm, config_name=args.config, style=args.style,
            limit=args.limit, out_suffix=args.out_suffix, fill_parser=fill_parser, min_filled_keys=min_keys,
        )
        print("\n" + format_outcome(outcome))
        print(f"\nqueries evaluated: {outcome.report['meta']['n_queries']}   products in the pool: {len(catalog)}")
        qf = outcome.report["meta"].get("query_filter")
        if qf:
            dist = Counter(qf["filled_per_query"].values())
            print(f"query filter: kept {qf['kept']} of {qf['of']} queries with >= {qf['min_filled_keys']} metadata keys "
                  "filled (keys filled: " + ", ".join(f"{c}: {dist[c]}" for c in sorted(dist)) + ")")
        for name, pipe in methods.items():
            if getattr(pipe, "parser", None) is not None:
                print(f"parser fallbacks, {name} (no filters/tags used): {pipe.parser.n_failures}")
        print(f"\nresults saved in {outcome.out_dir}")
        print("filtering per method:")
        for name, desc in outcome.report["meta"]["filtering"].items():
            print(f"  {name:16s} {filtering_label(desc)}")


if __name__ == "__main__":
    main()

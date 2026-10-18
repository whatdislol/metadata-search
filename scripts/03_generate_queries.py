"""Step 3: generate evaluation queries with known ground-truth constraints (eval.mode: llm_judge).

    python scripts/03_generate_queries.py --config qam_toys [--n 200] [--style specific|broad] [--provider uofg]

  specific  queries may describe the source product in detail (known-item search)
  broad     product type + need only; many products can match (closer to the paper's queries)

Only for datasets without human judgments (the toys sample). Datasets with qrels bring their own
queries; script 01 prepares them.
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config, resolve_path
from mdsearch.data.schema import load_products
from mdsearch.eval.dispatch import eval_mode
from mdsearch.eval.querygen import generate_queries, save_queries
from mdsearch.experiment import style_paths
from mdsearch.llm import get_llm
from mdsearch.runlog import record_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--style", default="specific", choices=["specific", "broad"])
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    if eval_mode(cfg) != "llm_judge":
        raise SystemExit(f"{args.config} uses human judgments (eval.mode: {eval_mode(cfg)}); "
                         "its queries come from scripts/01_prepare.py")
    with record_run("03_generate_queries", config=args.config, args=vars(args), label=args.style) as run:
        run.set_filtering({}, print_it=False)
        products = load_products(resolve_path(cfg["dataset"]["processed"]))
        qcfg = cfg["queries"]
        n = args.n or qcfg["n_generate"]
        queries = generate_queries(
            get_llm(args.provider), products, n, seed=qcfg.get("seed", 13),
            currency=qcfg.get("currency", "£"), style=args.style,
        )
        out, _ = style_paths(cfg, args.style)
        save_queries(out, queries)
        print(f"wrote {len(queries)}/{n} queries -> {out}")
        for q in queries[:5]:
            print(f"  [{q.id}] {q.text}\n       {q.filters.active()}")


if __name__ == "__main__":
    main()

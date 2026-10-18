"""Step 4a: generate evaluation queries with known ground-truth constraints.

    python scripts/04_generate_queries.py [--n 200] [--style specific|broad] [--provider uofg]

  specific  queries may describe the source product in detail (known-item search)
  broad     product type + need only; many products can match (closer to the paper's queries)
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config, resolve_path
from mdsearch.data.schema import load_products
from mdsearch.eval.querygen import generate_queries, save_queries
from mdsearch.experiment import style_paths
from mdsearch.llm import get_llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--style", default="specific", choices=["specific", "broad"])
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    products = load_products(resolve_path(cfg["dataset"]["processed"]))
    qcfg = cfg["queries"]
    n = args.n or qcfg["n_generate"]

    llm = get_llm(args.provider)
    queries = generate_queries(
        llm, products, n, seed=qcfg.get("seed", 13), currency=qcfg.get("currency", "£"),
        style=args.style,
    )
    out, _ = style_paths(cfg, args.style)
    save_queries(out, queries)
    print(f"wrote {len(queries)}/{n} queries -> {out}")
    for q in queries[:5]:
        print(f"  [{q.id}] {q.text}\n       {q.filters.active()}")


if __name__ == "__main__":
    main()

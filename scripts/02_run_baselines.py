"""Step 2b: run the four baselines (keyword, semantic, rerank, hybrid) on a few queries.

    python scripts/02_run_baselines.py -q "LEGO set for a 6 year old under 40" [-q ...]

Needs the ML extras (pip install -e '.[ml]'); no LLM is used here.
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config
from mdsearch.experiment import load_catalog, make_methods

DEFAULT_QUERIES = [
    "educational toys from LEGO that promote creativity for children aged 5-8",
    "a top-rated board game for kids aged 9-12 under 40",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("-q", "--query", action="append", default=None)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    catalog = load_catalog(cfg)
    methods = make_methods(cfg, catalog, llm=None, only=("keyword", "semantic", "rerank", "hybrid"))

    for q in args.query or DEFAULT_QUERIES:
        print(f"\n=== {q}")
        for name, pipe in methods.items():
            print(f"-- {name}")
            for h in pipe.search(q, k=args.k).hits:
                p = catalog.get(h.id)
                print(f"   {h.score:8.3f}  {p.title[:55]:55s} brand={p.brand} price={p.price} "
                      f"rating={p.rating} min_age={p.min_age}")


if __name__ == "__main__":
    main()

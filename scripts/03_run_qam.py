"""Step 3: run QAM (parse -> hard filter -> semantic search -> cross-encoder) on a few queries.

    python scripts/03_run_qam.py -q "Looking for LEGO toys for a 6-year-old under 40" [--provider uofg]

Shows the parsed filters and how many products each filter removed.
"""

from __future__ import annotations

import argparse
import json

from mdsearch.config import load_experiment_config
from mdsearch.experiment import load_catalog, make_methods
from mdsearch.llm import get_llm

DEFAULT_QUERIES = [
    "Looking for educational toys specifically from LEGO, designed to promote creativity, "
    "suitable for children aged 5-8",
    "Locate a top-rated board game from Hasbro for kids aged 9-12 within a budget of 40",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--provider", default=None)
    ap.add_argument("-q", "--query", action="append", default=None)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    catalog = load_catalog(cfg)
    llm = get_llm(args.provider)
    methods = make_methods(cfg, catalog, llm, only=("qam",))
    qam = methods["qam"]

    for q in args.query or DEFAULT_QUERIES:
        res = qam.search(q, k=args.k)
        print(f"\n=== {q}")
        print("parsed:", json.dumps(res.parsed.model_dump() if res.parsed else None))
        if res.filter_report:
            r = res.filter_report
            print(f"filter: {r.total} -> {r.remaining} products")
            for s in r.steps:
                print(f"   {s.name}={s.value}: removed {s.removed}")
        for h in res.hits:
            p = catalog.get(h.id)
            print(f"   {h.score:8.3f}  {p.title[:55]:55s} brand={p.brand} price={p.price} "
                  f"rating={p.rating} min_age={p.min_age}")


if __name__ == "__main__":
    main()

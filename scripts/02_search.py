"""Step 2: try methods on a few queries of any dataset and see what each one did.

    python scripts/02_search.py --config qam_toys -q "LEGO set for a 6 year old under 40"
    python scripts/02_search.py --config trec_product -q "waterproof phone case" --methods rerank,rerank_hard,rerank_soft
    python scripts/02_search.py -q "creative building set" --methods rerank --brand LEGO --price-max 40 --age-max 8


Explicit filters (--brand --price-min --price-max --rating-min --age-min --age-max) add a second
run of every plain ranker (keyword / semantic / rerank / hybrid) with those hard filters, so you see
what the filter changed. They only work on datasets that have those fields (e.g. toys).

Prints the filter type of each method (hard / soft / none), the generated tags or parsed filters,
and how many products each step removed. Output is also saved under results/runs/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from mdsearch.config import load_experiment_config
from mdsearch.eval.dispatch import default_methods
from mdsearch.experiment import load_catalog, make_methods
from mdsearch.filtering import MetadataFilter
from mdsearch.llm import get_llm
from mdsearch.methods import uses_llm
from mdsearch.query.schema import Filters, ParsedQuery
from mdsearch.runlog import describe_methods, record_run


def product_line(p) -> str:
    fields = {"brand": p.brand, "price": p.price, "rating": p.rating, "min_age": p.min_age}
    shown = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    return f"{p.title[:60]:60s} {shown}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--provider", default=None)
    ap.add_argument("-q", "--query", action="append", default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--brand")
    ap.add_argument("--price-min", type=float)
    ap.add_argument("--price-max", type=float)
    ap.add_argument("--rating-min", type=float)
    ap.add_argument("--age-min", type=float)
    ap.add_argument("--age-max", type=float)
    ap.add_argument("--missing", default="exclude", choices=["exclude", "keep"],
                    help="explicit filters: what to do with products whose value is unknown")
    ap.add_argument("--price-tol", type=float, default=0.0, help="explicit filters: 0.2 = 20%% slack")
    ap.add_argument("--rating-tol", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    queries = args.query or cfg.get("search", {}).get("example_queries")
    if not queries:
        raise SystemExit("give at least one -q QUERY")
    only = tuple(m.strip() for m in args.methods.split(",")) if args.methods else default_methods(cfg)
    explicit = Filters(
        brand=args.brand, price_min=args.price_min, price_max=args.price_max,
        rating_min=args.rating_min, age_min=args.age_min, age_max=args.age_max,
    )

    with record_run("02_search", config=args.config, args=vars(args)) as run:
        llm = get_llm(args.provider) if any(uses_llm(m) for m in only) else None
        catalog = load_catalog(cfg)
        pipelines = make_methods(cfg, catalog, llm, only=only)
        explicit_names: set[str] = set()
        if not explicit.is_empty():
            print("explicit filters:", explicit.active())
            for name, pipe in list(pipelines.items()):
                if pipe.parser is None and pipe.metadata_filter is None and pipe.tag_matcher is None:
                    fname = f"{name}+filter"
                    pipelines[fname] = replace(pipe, name=fname, metadata_filter=MetadataFilter(
                        catalog, missing=args.missing, price_tol=args.price_tol, rating_tol=args.rating_tol))
                    explicit_names.add(fname)
        run.set_filtering(describe_methods(pipelines))

        for q in queries:
            print(f"\n=== {q}")
            for name, pipe in pipelines.items():
                parsed = ParsedQuery(semantic_text=q, filters=explicit) if name in explicit_names else None
                res = pipe.search(q, k=args.k, parsed=parsed)
                print(f"\n-- {name}")
                if res.parsed and res.parsed.tags:
                    print("   tags:", ", ".join(f"{t.text} [{t.kind}]" for t in res.parsed.tags))
                elif res.parsed and not res.parsed.filters.is_empty():
                    print("   parsed filters:", json.dumps(res.parsed.filters.active()))
                if res.filter_report:
                    r = res.filter_report
                    print(f"   filter: {r.total} -> {r.remaining} products")
                    for s in r.steps:
                        print(f"      {s.name}={s.value}: removed {s.removed}")
                for h in res.hits:
                    print(f"   {h.score:8.3f}  {product_line(catalog.get(h.id))}")


if __name__ == "__main__":
    main()

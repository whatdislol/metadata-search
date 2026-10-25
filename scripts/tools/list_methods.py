"""List every valid method name and what it does (no models, data or LLM needed).

    python scripts/tools/list_methods.py                      # all methods, grouped by base
    python scripts/tools/list_methods.py --base rerank        # one base method (repeat or comma-separate)
    python scripts/tools/list_methods.py --config trec_product   # also mark the ones in that config's eval.methods
    python scripts/tools/list_methods.py --name rerank_soft_expand   # one method in detail

The names are <base>[_hard|_soft][_expand]; the table is built from the same grammar the code uses
(src/mdsearch/methods.py), so it cannot go out of date.
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config
from mdsearch.eval.dispatch import default_methods
from mdsearch.methods import BASE_METHODS, describe_method, method_catalog, parse_method

COLUMNS = ("method", "llm", "filter", "first_stage", "second_stage", "datasets")
TITLES = {"method": "method", "llm": "LLM step", "filter": "filter", "first_stage": "first stage searches",
          "second_stage": "second stage", "datasets": "datasets"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="append", help=f"only these base methods: {', '.join(BASE_METHODS)}")
    ap.add_argument("--config", help="mark the methods listed in this config's eval.methods with *")
    ap.add_argument("--name", help="describe one method in detail")
    args = ap.parse_args()

    if args.name:
        for k, v in describe_method(args.name).items():
            print(f"{k:13s} {v}")
        return

    bases = {b.strip() for a in (args.base or []) for b in a.split(",")}
    bad = bases - set(BASE_METHODS)
    if bad:
        raise SystemExit(f"unknown base method(s) {sorted(bad)}; choose from {', '.join(BASE_METHODS)}")
    used = set(default_methods(load_experiment_config(args.config))) if args.config else set()
    rows = [describe_method(n) for n in method_catalog() if not bases or parse_method(n)[0] in bases]
    width = {c: max(len(TITLES[c]), *(len(r[c]) for r in rows)) for c in COLUMNS}
    print("  " + "  ".join(TITLES[c].ljust(width[c]) for c in COLUMNS))
    prev = None
    for r in rows:
        base = parse_method(r["method"])[0]
        if prev is not None and base != prev:
            print()
        prev = base
        flag = ("* " if r["method"] in used else "  ") if args.config else "  "
        print(flag + "  ".join(r[c].ljust(width[c]) for c in COLUMNS).rstrip())
    print(f"\n{len(rows)} methods. Names are <base>[_hard|_soft][_expand]; * = in eval.methods of the config.")


if __name__ == "__main__":
    main()

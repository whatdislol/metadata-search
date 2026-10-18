"""Check that a judged dataset's queries, qrels and corpus files belong together (run before 01_prepare).

    python scripts/tools/check_dataset_files.py --config trec_product
    python scripts/tools/check_dataset_files.py --config trec_product --scan-corpus     # also read the corpus

Prints the id ranges of the queries and qrels files, the grade scale, how many query ids they share,
and (with --scan-corpus, a few minutes for the 1.1 GB corpus) how many judged product ids exist in the corpus.
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config, resolve_path
from mdsearch.data.evalset import diagnose_ids, iter_jsonl, load_qrels, load_queries_tsv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="trec_product")
    ap.add_argument("--scan-corpus", action="store_true")
    args = ap.parse_args()

    d = load_experiment_config(args.config)["dataset"]
    paths = {k: resolve_path(d[k]) for k in ("queries", "qrels") if k in d}
    for k, p in paths.items():
        print(f"{k:8s} {p}  {'(found)' if p.exists() else '(MISSING)'}")
    if not all(p.exists() for p in paths.values()):
        raise SystemExit("download the missing files first (see the comments in the config)")
    queries, qrels = load_queries_tsv(paths["queries"]), load_qrels(paths["qrels"])
    print()
    print("\n".join(diagnose_ids(queries, qrels)))

    if args.scan_corpus:
        corpus = resolve_path(d["corpus"])
        judged = {pid for q in set(queries) & set(qrels) for pid in qrels[q]}
        print(f"\nscanning {corpus} for {len(judged)} judged product ids ...")
        found, n, last = 0, 0, None
        for n, rec in enumerate(iter_jsonl(corpus), start=1):
            found += str(rec["docid"]) in judged
            last = rec["docid"]
            if n % 500_000 == 0:
                print(f"  ... {n:,} records, {found} judged ids found so far", flush=True)
        print(f"corpus records: {n:,} (last docid {last})   judged ids found: {found} of {len(judged)} "
              f"({found / max(1, len(judged)):.1%})")


if __name__ == "__main__":
    main()

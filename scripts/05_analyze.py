"""Step 5: look inside the evaluation results (no LLM, no models, runs in seconds).

    python scripts/05_analyze.py --config qam_toys [--style specific|broad] [--k 5] [--ref qam] [--misses 15] [--parse-errors 15]
    python scripts/05_analyze.py --config trec_product --k 10 --ref qam_tags_hard
    python scripts/05_analyze.py --config qam_toys --out-suffix _tags --ref qam_tags_hard

eval.mode llm_judge: where non-relevant results come from (hard-constraint violations vs the subjective
  part), precision among results that satisfy the constraints, source-product hits, parser mistakes,
  paired bootstrap against --ref (details in mdsearch/eval/analysis.py). --misses / --parse-errors list examples.
eval.mode qrels: P, nDCG, recall per method, queries with nothing relevant in the top k, the size of the
  candidate pool after filtering, and a paired bootstrap of nDCG against --ref.

Output is saved under results/runs/ together with the filter type (hard / soft / none) of each method.
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config
from mdsearch.eval.analysis import run_analysis
from mdsearch.runlog import record_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--style", default="specific", choices=["specific", "broad"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ref", default="qam")
    ap.add_argument("--misses", type=int, default=0, help="list N source-product misses of --ref")
    ap.add_argument("--parse-errors", type=int, default=0,
                    help="list N queries whose constraints --ref's parser missed")
    ap.add_argument("--out-suffix", default="", help="same value that was given to 04, e.g. _tags")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    with record_run("05_analyze", config=args.config, args=vars(args), label=f"{args.ref}") as run:
        text, filtering = run_analysis(
            cfg, style=args.style, k=args.k, ref=args.ref, misses=args.misses,
            parse_errors=args.parse_errors, out_suffix=args.out_suffix,
        )
        run.set_filtering(filtering)
        print(text)


if __name__ == "__main__":
    main()

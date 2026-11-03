"""Step 1: turn a dataset's raw files into the project's format (works for every dataset).

    python scripts/01_prepare.py --config qam_toys          # Kaggle toys: parse the CSV, print coverage
    python scripts/01_prepare.py --config trec_product      # judged collection: products + queries + qrels

The dataset is chosen by `dataset.kind` in the config (see mdsearch/data/registry.py). Optional:
--source PATH  use another raw file than the config's, --limit N  only read N records (quick test).
The printed output is also saved under results/runs/.
"""

from __future__ import annotations

import argparse

from mdsearch.config import load_experiment_config
from mdsearch.data.registry import prepare_dataset
from mdsearch.runlog import record_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--source", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    with record_run("01_prepare", config=args.config, args=vars(args), label=cfg["dataset"]["kind"]) as run:
        run.set_filtering({}, print_it=False)
        prepare_dataset(cfg, limit=args.limit, source=args.source)


if __name__ == "__main__":
    main()

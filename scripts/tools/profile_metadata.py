"""Profile the native product metadata to choose `representation.metadata.fields` for the `_meta` methods.

  python scripts/tools/profile_metadata.py                      # processed products of the active config
  python scripts/tools/profile_metadata.py --corpus corpus.jsonl --limit 200000   # raw corpus file, streamed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mdsearch.config import load_experiment_config, resolve_path  # noqa: E402
from mdsearch.data.schema import load_products  # noqa: E402
from mdsearch.data.trec_product import to_product  # noqa: E402
from mdsearch.metadata_profile import format_profile, profile  # noqa: E402


def stream(path: Path, limit: int | None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                yield to_product(json.loads(line))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="trec_product")
    ap.add_argument("--corpus", help="raw corpus JSONL instead of the processed products")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-coverage", type=float, default=0.05)
    a = ap.parse_args()
    if a.corpus:
        products = stream(resolve_path(a.corpus), a.limit)
    else:
        cfg = load_experiment_config(a.config)
        products = load_products(resolve_path(cfg["dataset"]["processed"]))
        if a.limit:
            products = products[: a.limit]
    print(format_profile(profile(products, a.top, a.min_coverage)))


if __name__ == "__main__":
    main()
"""Step 2a: parse the raw Kaggle toys CSV into the project's Product format.

    python scripts/01_prepare_toys.py [--config qam_toys] [--csv PATH] [--limit N]

Read the printed coverage: if brand/price/rating/min_age/reviews are mostly missing, the
parsers in mdsearch/data/toys_kaggle.py do not match the real file and need adjusting.
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from mdsearch.config import load_experiment_config, resolve_path
from mdsearch.data.schema import save_products
from mdsearch.data.toys_kaggle import describe_coverage, load_toys_csv, parse_price


def diagnose_price(csv) -> None:
    """Split price gaps into 'cell is empty in the CSV' vs 'cell has text the parser rejects'."""
    df = pd.read_csv(csv)
    if "price" not in df.columns:
        print("\nno 'price' column; columns are:", list(df.columns))
        return
    raw = df["price"]
    empty = raw.isna() | raw.astype(str).str.strip().eq("")
    unparsed = ~empty & raw.map(parse_price).isna()
    print(f"\nprice diagnosis over {len(df)} rows:")
    print(f"  empty in the CSV:            {int(empty.sum())}  (real missing data, not a parser bug)")
    print(f"  non-empty but not parsed:    {int(unparsed.sum())}  (parser bug if > 0)")
    for v, c in Counter(raw[unparsed].astype(str)).most_common(10):
        print(f"      {c:5d} x {v!r}")
    shapes = Counter(
        pd.Series(raw[~empty].astype(str)).str.replace(r"\d+", "9", regex=True).head(2000)
    )
    print("  most common price shapes:", shapes.most_common(5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)["dataset"]
    csv = resolve_path(args.csv or cfg["raw_csv"])
    if not csv.exists():
        raise SystemExit(
            f"{csv} not found. Download 'Toy Products on Amazon' (PromptCloud) from Kaggle and "
            "place amazon_co-ecommerce_sample.csv there."
        )

    products = load_toys_csv(csv, limit=args.limit)
    out = resolve_path(cfg["processed"])
    n = save_products(out, products)
    print(f"wrote {n} products -> {out}")
    for k, v in describe_coverage(products).items():
        print(f"  {k:24s} {v:.3f}" if v <= 1 else f"  {k:24s} {v:.0f}")

    diagnose_price(csv)

    print("\nsample:")
    for p in products[:3]:
        print(f"  {p.title[:60]!r} brand={p.brand} price={p.price} rating={p.rating} "
              f"min_age={p.min_age} reviews={len(p.reviews)}")


if __name__ == "__main__":
    main()

"""Spot-check the age parser: raw text snippets next to the parsed min_age.

    python scripts/tools/check_toys_parse.py [-n 15] [--seed 1] [--conflicts]

Prints random products with the age-related text found in `product_information` and in
`description`, and what the parser returned. --conflicts only shows products where the two
sources give different ages (the airport-set situation).
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from mdsearch.config import load_experiment_config, resolve_path
from mdsearch.data.toys_kaggle import parse_min_age

SNIP = re.compile(r".{0,25}\b(?:ages?|years?|yrs?|months?)\b.{0,35}", re.I)


def snippet(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    m = SNIP.search(str(v))
    return m.group().replace("\n", " ").strip() if m else "(no age text)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qam_toys")
    ap.add_argument("-n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--conflicts", action="store_true")
    args = ap.parse_args()

    csv = resolve_path(load_experiment_config(args.config)["dataset"]["raw_csv"])
    df = pd.read_csv(csv)
    info = df.get("product_information", pd.Series([None] * len(df)))
    desc = df.get("description", pd.Series([None] * len(df)))
    df = df.assign(
        _from_info=[parse_min_age(x) for x in info],
        _from_desc=[parse_min_age(x) for x in desc],
        _final=[parse_min_age(i, d, t) for i, d, t in zip(info, desc, df["product_name"])],
    )
    print(f"{len(df)} rows | age from info: {df._from_info.notna().mean():.3f} | "
          f"from description: {df._from_desc.notna().mean():.3f} | final: {df._final.notna().mean():.3f}")
    odd = info.astype(str).str.extract(r"age:\s*(\d+)\s*-\s*(\d+)\s*years", flags=re.I).astype(float)
    n_odd = int((odd[0] > odd[1]).sum())
    print(f"impossible ranges in info ('A - B years' with A > B): {n_odd}")
    both = df._from_info.notna() & df._from_desc.notna()
    conflict = both & ((df._from_info - df._from_desc).abs() > 0.01)
    print(f"both sources present: {int(both.sum())}, disagree: {int(conflict.sum())} "
          f"({conflict.sum() / max(both.sum(), 1):.1%})\n")
    pool = df[conflict] if args.conflicts else df
    for _, r in pool.sample(min(args.n, len(pool)), random_state=args.seed).iterrows():
        print(f"{str(r['product_name'])[:60]!r}")
        print(f"   info : {snippet(info[_])!r:70s} -> {r._from_info}")
        print(f"   desc : {snippet(desc[_])!r:70s} -> {r._from_desc}")
        print(f"   FINAL min_age = {r._final}\n")


if __name__ == "__main__":
    main()

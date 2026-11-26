"""Which native metadata is worth writing into the product text: key frequencies, coverage and examples."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .data.schema import Product


def profile(products: Iterable[Product], top: int = 40, min_coverage: float = 0.05) -> dict:
    n = 0
    keys: Counter[str] = Counter()
    canon: dict[str, str] = {}
    lens: dict[str, int] = Counter()
    examples: dict[str, str] = {}
    depth: Counter[int] = Counter()
    have = Counter()
    for p in products:
        n += 1
        if p.category:
            depth[len(p.category.split(" > "))] += 1
        have["price"] += p.price is not None
        have["rating"] += p.rating is not None
        have["category"] += bool(p.category)
        seen = set()
        for table in ("attrs", "info"):
            for k, v in (p.extra.get(table) or {}).items():
                lk = k.lower()
                if lk in seen:
                    continue
                seen.add(lk)
                keys[lk] += 1
                canon.setdefault(lk, k)
                lens[lk] += len(str(v))
                examples.setdefault(lk, str(v))
        have["any attribute"] += bool(seen)
    rows = [
        {"key": canon[k], "products": c, "coverage": c / max(n, 1), "mean_chars": lens[k] / c, "example": examples[k]}
        for k, c in keys.most_common(top)
    ]
    return {
        "n": n, "have": dict(have), "category_depth": dict(sorted(depth.items())), "keys": rows,
        "suggested_fields": [r["key"] for r in rows if r["coverage"] >= min_coverage and r["mean_chars"] <= 60],
    }


def format_profile(r: dict) -> str:
    n = max(r["n"], 1)
    out = [f"{r['n']} products"]
    out += [f"  {k}: {v} ({v / n:.1%})" for k, v in r["have"].items()]
    out.append(f"  category depth (levels: products): {r['category_depth']}")
    out.append(f"\n{'key':32} {'products':>9} {'cover':>7} {'chars':>6}  example")
    for x in r["keys"]:
        out.append(f"{x['key'][:32]:32} {x['products']:>9} {x['coverage']:>7.1%} {x['mean_chars']:>6.0f}  {x['example'][:50]}")
    fields = ", ".join(f'"{f}"' for f in r["suggested_fields"])
    out.append(f"\nsuggested (coverage >= min, short values): fields: [{fields}]")
    out.append("merge synonyms yourself, e.g. \"Brand|Manufacturer\", \"Color|Colour\"; keep category_levels: 2 or 3")
    return "\n".join(out)

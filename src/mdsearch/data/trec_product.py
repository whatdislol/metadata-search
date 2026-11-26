"""TREC Product Search loader. Only what is specific to this collection lives here; query / qrels
reading, subset sampling and the saved eval set are shared (data/evalset.py).

Corpus records are {"docid": int, "title": str, "text": str, ...}. The newer corpus file also has the
listing's category path, attribute tables ("attrs", "info"), price, stars and ratings; those are kept on
the Product (category, extra, price, rating, n_reviews) for the `_meta` methods. The older file has only
title and text. The corpus has millions of products, so `prepare` keeps every judged product plus a
random sample of the rest.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import resolve_path
from .evalset import as_paths, assign_groups, build_qrels_dataset, iter_jsonl, load_qrels_files, read_queries
from .schema import Product, save_products

_LABEL = re.compile(r"Product Description")
# Some corpus texts contain passages with a space after every character and three spaces between
# words: "P r o d u c t   D e s c r i p t i o n". A run of >= 8 single characters is rebuilt.
_SPACED = re.compile(r"(?<!\S)\S(?: {1,3}\S){7,}(?!\S)")
# invisible direction marks / zero-width and non-breaking spaces around "Manufacturer : name" etc.
_INVISIBLE = {ord(c): " " for c in "\u200e\u200f\u200b\u00a0"}


def _unspace(m: re.Match) -> str:
    run = m.group(0)
    if "   " not in run and len(run) < 23:  # no word gaps and short: probably real text ("1 2 3 4 5 6 7 8")
        return run
    return " ".join(w.replace(" ", "") for w in run.split("   "))


def clean_text(text: str | None) -> str:
    """Rebuild letter-spaced passages, drop invisible characters, the 'Product Description' labels and
    the verbatim repeats the corpus contains."""
    text = _SPACED.sub(_unspace, (text or "").translate(_INVISIBLE))
    seen: set[str] = set()
    parts: list[str] = []
    for seg in _LABEL.split(text):
        seg = " ".join(seg.split())
        if seg and seg not in seen:
            seen.add(seg)
            parts.append(seg)
    return " ".join(parts)


# keys of the corpus's attribute tables that never help a search (ids, ranks, dates, review summaries)
NOISE_KEYS = {
    "asin", "best sellers rank", "customer reviews", "date first available", "is discontinued by manufacturer",
    "batteries required?", "batteries included?", "item weight", "product dimensions", "package dimensions",
    "isbn-10", "isbn-13", "publication date", "print length", "dimensions", "customer review",
}
_MAX_VALUE = 200  # characters kept per attribute value in the processed file


def clean_key(k: Any) -> str:
    return " ".join(str(k).translate(_INVISIBLE).split()).strip(" :")


def clean_value(v: Any) -> str:
    return " ".join(str(v).translate(_INVISIBLE).split())[:_MAX_VALUE]


def clean_table(d: Any) -> dict[str, str]:
    """An attribute table ({"Brand": "FYY", ...}) with cleaned keys and values, without noise keys or empty values."""
    if not isinstance(d, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in d.items():
        key, val = clean_key(k), clean_value(v)
        if key and val and key.lower() not in NOISE_KEYS:
            out.setdefault(key, val)
    return out


def _number(text: Any) -> float | None:
    m = re.search(r"\d[\d,]*\.?\d*", str(text or ""))
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def to_product(rec: dict[str, Any]) -> Product:
    """Title and text as before (so the plain baselines read the same text), plus the corpus's own metadata:
    the category path (Product.category, levels joined by ' > '), the attribute tables (extra["attrs"],
    extra["info"]), and price / star rating / number of ratings when the listing has them."""
    category = [clean_value(c) for c in (rec.get("category") or []) if clean_value(c)]
    extra: dict[str, Any] = {}
    for name, table in (("attrs", rec.get("attrs") or rec.get("attr")), ("info", rec.get("info"))):
        cleaned = clean_table(table)
        if cleaned:
            extra[name] = cleaned
    price, rating, n = _number(rec.get("price")), _number(rec.get("stars")), _number(rec.get("ratings"))
    return Product(
        id=str(rec["docid"]),
        title=" ".join(str(rec.get("title") or "").split()),
        description=clean_text(str(rec.get("text") or "")),
        price=price, rating=rating if rating is not None and rating <= 5 else None,
        n_reviews=int(n) if n is not None else None,
        category=" > ".join(category) or None,
        extra=extra,
    )


def record_id(rec: dict[str, Any]) -> str:
    return str(rec["docid"])


def prepare(cfg: dict[str, Any], *, limit: int | None = None, source: str | None = None) -> None:
    """Write the processed subset and eval set named in cfg['dataset'] (see configs/trec_product.yaml)."""
    d = cfg["dataset"]
    corpus = resolve_path(source or d["corpus"])
    paths = {
        "corpus": [corpus],
        "queries": [resolve_path(p) for p in as_paths(d["queries"])],  # one file or a list (several years)
        "qrels": [resolve_path(p) for p in as_paths(d["qrels"])],
    }
    for key, plist in paths.items():
        for p in plist:
            if not p.exists():
                raise SystemExit(f"{p} not found; set dataset.{key} in the config")

    records = iter_jsonl(corpus)
    if limit:  # quick tests: only the first N corpus records
        records = (r for i, r in zip(range(limit), records))
    qf = read_queries(paths["queries"])
    if qf.duplicates:
        shown = ", ".join(f"{dup} (= {kept})" for dup, kept in qf.duplicates[:6])
        print(f"dropped {len(qf.duplicates)} queries whose text repeats an earlier file's query: {shown}"
              + (" ..." if len(qf.duplicates) > 6 else ""))
    groups = assign_groups(qf.queries, qf.source, d.get("query_groups"))
    products, evalset = build_qrels_dataset(
        records, to_product, record_id,
        qf.queries, load_qrels_files(paths["qrels"]), cfg.get("subset", {}), groups=groups,
    )
    n = save_products(resolve_path(d["processed"]), products)
    evalset.save(resolve_path(d["eval_set"]))
    print(f"wrote {n} products -> {resolve_path(d['processed'])}")
    print(f"wrote eval set  -> {resolve_path(d['eval_set'])}")

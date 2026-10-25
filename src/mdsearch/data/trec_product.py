"""TREC Product Search loader. Only what is specific to this collection lives here; query / qrels
reading, subset sampling and the saved eval set are shared (data/evalset.py).

Corpus records are {"docid": int, "title": str, "text": str}. There are no structured price / brand /
rating fields: whatever metadata exists is words inside `title` and `text`. The corpus has millions
of products, so `prepare` keeps every judged product plus a random sample of the rest.
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


def to_product(rec: dict[str, Any]) -> Product:
    return Product(
        id=str(rec["docid"]),
        title=" ".join(str(rec.get("title") or "").split()),
        description=clean_text(str(rec.get("text") or "")),
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

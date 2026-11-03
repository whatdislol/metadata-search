"""Dataset-independent pieces for collections that come with queries and relevance judgments (qrels).

Any such dataset (TREC Product Search, WANDS, ESCI, ...) needs only a small loader that turns its
raw records into `Product`s; everything else here is shared:

    queries : {query_id: text}
    qrels   : {query_id: {product_id: grade}}      grade 0 = not relevant; higher = more relevant
    EvalSet : the chosen queries + judgments, saved as JSON next to the processed products

Assumed plain-text formats (a loader for another format only has to return the same dicts):
    queries : tab-separated  <qid>\t<query text>, or a .csv file  <qid>,<query text>  (header row optional)
    qrels   : TREC           <qid> <iteration> <docid> <grade>      (3 columns <qid> <docid> <grade> also works)
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .schema import Product


def open_text(path: str | Path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8-sig")
    return open(p, encoding="utf-8-sig")  # -sig: a byte-order mark at the start of the file is dropped


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Records of a .jsonl or .jsonl.gz file."""
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


_ID_HEADERS = ("id", "qid", "query_id", "topic")
_TEXT_HEADERS = ("query", "text", "query_text", "title", "query name", "query_name")


def _is_header(qid: str, text: str) -> bool:
    return qid.strip().lower() in _ID_HEADERS and text.strip().lower() in _TEXT_HEADERS


def _query_rows(path: Path) -> Iterator[tuple[str, str]]:
    """(id, text) pairs of a queries file: .csv (quoted fields allowed) or tab-separated."""
    names = [s_.lower() for s_ in Path(path).suffixes]
    with open_text(path) as f:
        if ".csv" in names:
            for row in csv.reader(f):
                if len(row) >= 2:
                    yield row[0], row[1]
        else:
            for line in f:
                line = line.rstrip("\r\n")
                if line.strip():
                    qid, _, text = line.partition("\t")
                    yield qid, text


def load_queries(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, (qid, text) in enumerate(_query_rows(Path(path))):
        if i == 0 and _is_header(qid, text):
            continue  # header row
        if text.strip():
            out[qid.strip()] = text.strip()
    return out


load_queries_tsv = load_queries  # the old name; .csv files are read too


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open_text(path) as f:
        for line in f:
            cols = line.split()
            if len(cols) == 4:
                qid, docid, grade = cols[0], cols[2], cols[3]
            elif len(cols) == 3:
                qid, docid, grade = cols
            else:
                continue
            try:
                qrels.setdefault(qid, {})[docid] = int(float(grade))
            except ValueError:
                continue
    return qrels


def as_paths(value: Any) -> list[Path]:
    """A config value that is one path or a list of paths."""
    return [Path(v) for v in ([value] if isinstance(value, (str, Path)) else list(value))]


def _norm_text(text: str) -> str:
    """Case, punctuation and spacing do not make a different query."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class QueryFiles:
    queries: dict[str, str]
    source: dict[str, str]  # query id -> the file (stem) it came from
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (dropped id, id with the same text it repeats)


def read_queries(paths: Iterable[str | Path], *, dedupe: bool = True) -> QueryFiles:
    """Merge several queries files (e.g. several TREC years). The same id must not mean two different
    queries. With `dedupe`, a query whose text repeats an earlier one under another id (a later year
    re-using a hard query) is dropped, so the same query is not counted twice; list files in the order
    whose judgments you want to keep."""
    out: dict[str, str] = {}
    source: dict[str, str] = {}
    seen: dict[str, str] = {}  # normalised text -> id that has it
    dups: list[tuple[str, str]] = []
    for p in paths:
        stem = Path(p).name.split(".")[0]
        for qid, text in load_queries(p).items():
            if qid in out and out[qid] != text:
                raise SystemExit(f"query id {qid} means different queries in different files ({p}); "
                                 "the files' id spaces overlap, so they cannot be combined")
            key = _norm_text(text)
            if dedupe and qid not in out and key in seen:
                dups.append((qid, seen[key]))
                continue
            out[qid] = text
            source.setdefault(qid, stem)
            seen.setdefault(key, qid)
    return QueryFiles(out, source, dups)


def load_queries_files(paths: Iterable[str | Path]) -> dict[str, str]:
    return read_queries(paths).queries


def assign_groups(
    queries: Iterable[str], source: dict[str, str], patterns: dict[str, str] | None = None
) -> dict[str, str]:
    """A group name per query id: the first `patterns` entry (name -> regex on the id) that matches,
    else the file the query came from. Used to report results per year or per kind of query."""
    rules = [(name, re.compile(rx)) for name, rx in (patterns or {}).items()]
    out: dict[str, str] = {}
    for q in queries:
        out[q] = next((name for name, rx in rules if rx.search(q)), source.get(q, ""))
    return out


def load_qrels_files(paths: Iterable[str | Path]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for p in paths:
        for qid, judged in load_qrels(p).items():
            if qid in out:
                raise SystemExit(f"query id {qid} has judgments in more than one qrels file ({p}); "
                                 "the files' id spaces overlap, so they cannot be combined")
            out[qid] = judged
    return out


def select_queries(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    max_queries: int | None,
    min_relevant: int = 1,
    rel_threshold: int = 1,
    seed: int = 13,
) -> list[str]:
    """Query ids that have text and at least `min_relevant` relevant judgments, sampled by seed."""
    ok = sorted(
        q for q in queries
        if sum(g >= rel_threshold for g in qrels.get(q, {}).values()) >= min_relevant
    )
    if max_queries and len(ok) > max_queries:
        ok = sorted(random.Random(seed).sample(ok, max_queries))
    return ok


def build_subset(
    records: Iterable[dict[str, Any]],
    to_product: Callable[[dict[str, Any]], Product],
    record_id: Callable[[dict[str, Any]], str],
    keep_ids: set[str],
    n_distractors: int,
    seed: int = 13,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[Product], int]:
    """One pass over a large corpus: every record whose id is in `keep_ids`, plus a uniform random
    sample (reservoir) of `n_distractors` of the others. Returns (products, records seen)."""
    rng = random.Random(seed)
    kept: dict[str, Product] = {}
    reservoir: list[Product] = []
    seen_other = 0
    n = 0
    for n, rec in enumerate(records, start=1):
        pid = record_id(rec)
        if pid in keep_ids:
            kept[pid] = to_product(rec)
        else:
            seen_other += 1
            if len(reservoir) < n_distractors:
                reservoir.append(to_product(rec))
            else:
                j = rng.randrange(seen_other)
                if j < n_distractors:
                    reservoir[j] = to_product(rec)
        if progress and n % 500_000 == 0:
            progress(n)
    return list(kept.values()) + reservoir, n


@dataclass
class EvalSet:
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    rel_threshold: int = 1
    notes: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)  # query id -> group (year / kind of query)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"rel_threshold": self.rel_threshold, "queries": self.queries,
                 "qrels": self.qrels, "notes": self.notes, "groups": self.groups}
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvalSet":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["queries"], d["qrels"], d.get("rel_threshold", 1), d.get("notes", {}), d.get("groups", {}))


def build_qrels_dataset(
    records: Iterable[dict[str, Any]],
    to_product: Callable[[dict[str, Any]], Product],
    record_id: Callable[[dict[str, Any]], str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    subset: dict[str, Any],
    log: Callable[[str], None] = print,
    groups: dict[str, str] | None = None,
) -> tuple[list[Product], EvalSet]:
    """Shared preparation for judged collections: choose queries, keep their judged products plus
    random distractors, and drop queries that lost every relevant product. `subset` is the config
    section: max_queries, min_relevant, rel_threshold, n_distractors, seed."""
    thr = subset.get("rel_threshold", 1)
    seed = subset.get("seed", 13)
    log(f"queries with text: {len(queries)}   with qrels: {len(qrels)}   in both: "
        f"{len(set(queries) & set(qrels))}")
    chosen = select_queries(
        queries, qrels, max_queries=subset.get("max_queries"),
        min_relevant=subset.get("min_relevant", 1), rel_threshold=thr, seed=seed,
    )
    if not chosen:
        raise SystemExit("No query has both text and relevant judgments; check the file formats "
                         "(query ids must be the same strings in both files).")
    keep = {docid for q in chosen for docid in qrels[q]}
    log(f"chosen queries: {len(chosen)}   judged products to keep: {len(keep)}")

    products, n_seen = build_subset(
        records, to_product, record_id, keep, subset.get("n_distractors", 20000), seed,
        progress=lambda n: log(f"  ... {n:,} corpus records read"),
    )
    present = {p.id for p in products}
    found = len(keep & present)
    log(f"corpus records read: {n_seen:,}   subset size: {len(products)}")
    log(f"judged products found in the corpus: {found} of {len(keep)} ({found / len(keep):.1%})")
    if found / len(keep) < 0.9:
        log("WARNING: many judged product ids are not in the corpus. The qrels may use a different "
            "id than the corpus records (e.g. an Amazon ASIN); map them in the dataset loader.")

    eq: dict[str, str] = {}
    er: dict[str, dict[str, int]] = {}
    for q in chosen:
        judged = {pid: g for pid, g in qrels[q].items() if pid in present}
        if any(g >= thr for g in judged.values()):
            eq[q], er[q] = queries[q], judged
    log(f"queries kept: {len(eq)}   dropped (no relevant product in the corpus): {len(chosen) - len(eq)}")
    kept_groups = {q: g for q, g in (groups or {}).items() if q in eq}
    return products, EvalSet(eq, er, thr, {"n_products": len(products), "n_corpus_records": n_seen}, kept_groups)


def diagnose_ids(queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> list[str]:
    """Plain-language report on whether a queries file and a qrels file belong together."""
    from collections import Counter

    def span(ids) -> str:
        nums = [int(i) for i in ids if str(i).isdigit()]
        return f"{min(nums)} .. {max(nums)}" if nums else "(not numeric)"

    def sample(ids, n=4) -> list[str]:
        return sorted(ids, key=lambda x: (len(x), x))[:n]

    shared = set(queries) & set(qrels)
    lines = [
        f"queries file : {len(queries)} queries, ids {span(queries)}, e.g. {sample(queries)}",
        f"qrels file   : {len(qrels)} queries, {sum(len(v) for v in qrels.values())} judgments, "
        f"ids {span(qrels)}, e.g. {sample(qrels)}",
        f"grades       : {dict(sorted(Counter(g for v in qrels.values() for g in v.values()).items()))}",
        f"shared query ids: {len(shared)}",
    ]
    if not shared:
        lines.append("=> NO query id is in both files: they do not belong together (different year or "
                     "different file). Use the queries file of the same track year as the qrels.")
    elif len(shared) < min(len(queries), len(qrels)):
        lines.append(f"=> only {len(shared)} of {min(len(queries), len(qrels))} ids match; the rest are "
                     "ignored by script 01.")
    return lines

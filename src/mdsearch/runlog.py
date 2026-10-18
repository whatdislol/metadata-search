"""Save what a script printed, plus how it was run, under results/runs/.

Every recorded run gets its own folder:
    results/runs/<YYYYmmdd-HHMMSS>_<script>[_<label>]/
        output.txt   everything the script printed
        run.json     script, arguments, config, date, status and the FILTERING used

`filtering.kind` says whether results were produced with a "hard" filter (products that fail are
removed), a "soft" filter (products are re-scored, none removed), both, or "none".
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .config import resolve_path

RUNS_README = """# Script runs

One folder per recorded run of scripts 01-05:

- `output.txt`  what the script printed
- `run.json`    script, arguments, config, date, status, and `filtering`

`filtering.kind` is `hard` (products failing a constraint are removed from the candidates),
`soft` (products are re-scored, none removed), `hard+soft`, or `none`.
Script 04 writes the metrics to the config's `eval.out_dir` (e.g. `results/qam_toys`); its
`summary.json` has the same `filtering` entry for every method under `meta.filtering`.
"""


def describe_filtering(pipe: Any) -> dict[str, Any]:
    """Whether (and how) a SearchPipeline filters. Works on any object with the pipeline fields."""
    parts: list[dict[str, Any]] = []
    mf = getattr(pipe, "metadata_filter", None)
    if mf is not None:
        parts.append(
            {
                "type": "hard",
                "source": "metadata fields (brand, price, rating, age)",
                "missing_values": mf.missing,
                "price_tol": mf.price_tol,
                "rating_tol": mf.rating_tol,
            }
        )
    tm = getattr(pipe, "tag_matcher", None)
    if tm is not None:
        if tm.mode == "filter":
            parts.append(
                {"type": "hard", "source": "generated tags matched by text", "top_frac": tm.top_frac,
                 "kinds": sorted(tm.kinds) if tm.kinds is not None else "all"}
            )
        else:
            parts.append(
                {"type": "soft", "source": "generated tags matched by text", "weight": tm.weight,
                 "kinds": sorted(tm.kinds) if tm.kinds is not None else "all"}
            )
    kinds = sorted({p["type"] for p in parts})
    return {
        "kind": "+".join(kinds) if kinds else "none", "components": parts,
        "query_expansion": bool(getattr(getattr(pipe, "parser", None), "expand", False)),
        # only meaningful when an LLM parser produced a shortened query
        "first_stage_query": getattr(pipe, "first_stage_query", "semantic")
        if getattr(pipe, "parser", None) is not None else "semantic",
    }


def describe_methods(methods: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: describe_filtering(p) for name, p in methods.items()}


def filtering_label(desc: Mapping[str, Any]) -> str:
    """One readable line, e.g. 'hard (metadata fields: missing=exclude)'."""
    note = "; query expansion" if desc.get("query_expansion") else ""
    if desc.get("first_stage_query") == "original":
        note += "; searches the original query"
    if desc["kind"] == "none":
        return "none" + note
    bits = []
    for c in desc["components"]:
        if c["source"].startswith("metadata"):
            bits.append(f"metadata fields: missing={c['missing_values']}")
        else:
            only = "" if c.get("kinds", "all") == "all" else f", only {'/'.join(c['kinds'])} tags"
            if c["type"] == "hard":
                bits.append(f"tags: top {c['top_frac']:.0%} of pool per tag{only}")
            else:
                bits.append(f"tags: score boost, weight {c['weight']}{only}")
    return f"{desc['kind']} ({'; '.join(bits)}{note})"


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            if not getattr(st, "closed", False):  # output.txt is closed when libraries write at exit
                st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            if not getattr(st, "closed", False):
                st.flush()

    def __getattr__(self, name: str):  # encoding, isatty, ... from the real stdout
        return getattr(self.streams[0], name)


class Run:
    def __init__(self, directory: Path, info: dict[str, Any]) -> None:
        self.dir = directory
        self.info = info

    def set_filtering(self, filtering: Mapping[str, Any], *, print_it: bool = True) -> None:
        """`filtering` maps a method name to describe_filtering(...). Printed so it is in output.txt."""
        self.info["filtering"] = dict(filtering)
        if print_it:
            for name, desc in filtering.items():
                print(f"filtering [{name}]: {filtering_label(desc)}")


@contextmanager
def record_run(
    script: str,
    *,
    config: str | None = None,
    args: Mapping[str, Any] | None = None,
    label: str = "",
    root: str | Path = "results/runs",
) -> Iterator[Run]:
    base = resolve_path(root)
    base.mkdir(parents=True, exist_ok=True)
    readme = base / "README.md"
    if not readme.exists():
        readme.write_text(RUNS_README, encoding="utf-8")

    now = dt.datetime.now()
    name = f"{now:%Y%m%d-%H%M%S}_{script}" + (f"_{label}" if label else "")
    directory = base / name
    directory.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "script": script,
        "config": config,
        "args": dict(args or {}),
        "date": now.isoformat(timespec="seconds"),
        "status": "running",
        "filtering": {},
    }
    run = Run(directory, info)
    real_stdout = sys.stdout
    with open(directory / "output.txt", "w", encoding="utf-8") as out:
        sys.stdout = _Tee(real_stdout, out)
        try:
            yield run
            info["status"] = "ok"
        except BaseException as e:  # keep the record, then re-raise
            info["status"] = "failed"
            info["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
            raise
        finally:
            sys.stdout = real_stdout
            (directory / "run.json").write_text(
                json.dumps(info, indent=2, default=str), encoding="utf-8"
            )
            print(f"\nrecorded: {directory}")

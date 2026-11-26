"""Paths, YAML config loading and .env handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# src/mdsearch/config.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return Path(os.environ.get("MDSEARCH_CONFIG_DIR", PROJECT_ROOT / "configs"))


def resolve_path(p: str | os.PathLike[str]) -> Path:
    """Relative paths are relative to the repo root, so scripts work from any cwd."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Nested dicts are merged key by key; anything else (lists, numbers, strings) in `over` replaces."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_experiment_config(name_or_path: str, _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Load e.g. 'qam_toys' (from configs/) or an explicit path.

    A config may start from another one with `extends: <name>` and then only list what differs
    (e.g. a small pilot config that extends the main one); nested sections are merged, lists are replaced."""
    p = Path(name_or_path)
    if not p.exists():
        stem = name_or_path if name_or_path.endswith(".yaml") else name_or_path + ".yaml"
        p = config_dir() / stem
    cfg = load_yaml(p)
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    if str(p) in _seen:
        raise ValueError(f"config 'extends' loop at {p}")
    return _merge(load_experiment_config(parent, _seen + (str(p),)), cfg)


def load_env() -> None:
    """Load a .env file if python-dotenv is available. Existing env vars win."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

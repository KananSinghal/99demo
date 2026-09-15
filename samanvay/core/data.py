"""Loads and caches the knowledge files in data/. These are the only place domain
knowledge lives - adding a material class or a standards equivalence is a data
change, never a code change."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from .. import config

_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}


def load(name: str) -> dict:
    """Load data/<name>.json, cached. Raises a clear error if a file is missing."""
    with _LOCK:
        if name in _CACHE:
            return _CACHE[name]
        path = Path(config.DATA_DIR) / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"knowledge file not found: {path}\n"
                f"Set SAMANVAY_DATA_DIR or run from the repository root."
            )
        with path.open("r", encoding="utf-8") as fh:
            _CACHE[name] = json.load(fh)
        return _CACHE[name]


def save(name: str, payload: dict) -> Path:
    """Write a derived knowledge file (e.g. the mined abbreviation lexicon)."""
    path = Path(config.DATA_DIR) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)
    with _LOCK:
        _CACHE[name] = payload
    return path


def exists(name: str) -> bool:
    return (Path(config.DATA_DIR) / f"{name}.json").exists()


def invalidate(name: str | None = None) -> None:
    with _LOCK:
        if name is None:
            _CACHE.clear()
        else:
            _CACHE.pop(name, None)


# ------------------------------------------------------------------ convenience

def standards_kg() -> dict:
    return load("standards_kg")


def class_dictionary() -> dict:
    return load("class_dictionary")


def uom_table() -> dict:
    return load("uom")


def abbreviations() -> dict:
    """Seed lexicon merged with the mined one, mined winning on conflict."""
    base = dict(load("abbreviations").get("seed", {}))
    if exists("mined_abbreviations"):
        base.update(load("mined_abbreviations").get("mined", {}))
    return base


def noise_tokens() -> set:
    return set(load("abbreviations").get("noise_tokens", []))


def transliteration() -> dict:
    tbl = dict(load("abbreviations").get("transliteration", {}))
    tbl.pop("_comment", None)
    return tbl

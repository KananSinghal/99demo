"""Material class assignment and classification facets.

Classify first, extract second. Once a record is known to be a hexagon-head bolt
you are no longer doing open-ended information extraction - you are filling a fixed
schema from a controlled vocabulary.

The first decision is not which class, but whether the item is standardisable at
all. A bespoke fabricated skid is a population of one; harmonising it produces
noise. Knowing what NOT to harmonise is part of the specification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from . import data, text

GENERIC = "GENERIC"


@dataclass
class Classification:
    class_code: str
    confidence: float
    item_name: str
    nsc: str
    unspsc: str
    hsn: str
    eclass: str
    standardisable: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_KEYWORDS: Optional[List[Tuple[str, str, int]]] = None   # (keyword, class_code, weight)
_KEYWORDS_GLUED: Optional[List[Tuple[str, str, int]]] = None  # (space-stripped keyword, class_code, weight)

# A space-stripped substring match is only attempted for multi-word keywords at
# or above this stripped length. "GATE VALVE" -> "GATEVALVE" (9 chars) is safe;
# a single short word matched as a bare substring would risk the same false
# positive kg._SHORT_ALIAS exists to prevent (e.g. "CAP" inside an unrelated word).
_GLUE_MIN_LEN = 5


def _build() -> None:
    global _KEYWORDS, _KEYWORDS_GLUED
    if _KEYWORDS is not None:
        return
    rows: List[Tuple[str, str, int]] = []
    glued: List[Tuple[str, str, int]] = []
    for code, spec in data.class_dictionary()["classes"].items():
        for kw in spec.get("keywords", []):
            basic_kw = text.basic(kw)
            weight = len(kw.split())
            rows.append((basic_kw, code, weight))
            stripped = basic_kw.replace(" ", "")
            if weight >= 2 and len(stripped) >= _GLUE_MIN_LEN:
                glued.append((stripped, code, weight))
    # Longest keyword first so "GATE VALVE" beats "VALVE" and "BALL BEARING" beats "BEARING".
    rows.sort(key=lambda r: (-len(r[0]), r[0]))
    glued.sort(key=lambda r: (-len(r[0]), r[0]))
    _KEYWORDS = rows
    _KEYWORDS_GLUED = glued


def class_spec(class_code: str) -> dict:
    classes = data.class_dictionary()["classes"]
    return classes.get(class_code, classes[GENERIC])


def attribute_specs(class_code: str) -> List[dict]:
    return class_spec(class_code).get("attributes", [])


def attribute_spec(class_code: str, key: str) -> dict:
    for spec in attribute_specs(class_code):
        if spec["key"] == key:
            return spec
    return {"key": key, "label": key.upper()}


def mandatory_keys(class_code: str) -> List[str]:
    return [a["key"] for a in attribute_specs(class_code) if a.get("mandatory")]


def blocking_keys(class_code: str) -> List[str]:
    return [a["key"] for a in attribute_specs(class_code) if a.get("blocking")]


def all_classes() -> List[str]:
    return list(data.class_dictionary()["classes"].keys())


def is_engineered(description: str) -> Tuple[bool, str]:
    """Engineered-to-order items are excluded by design, not by accident."""
    hay = text.basic(description)
    for marker in data.class_dictionary().get("engineered_markers", []):
        if text.basic(marker) in hay:
            return True, f"engineered-to-order marker present: '{marker}'"
    return False, ""


def classify(description: str, extra: str = "") -> Classification:
    """Assign a material class. Keyword-driven from the class dictionary, so adding
    a class is a data change. Ambiguity resolves to GENERIC rather than a guess."""
    _build()
    hay = " " + text.clean(f"{description} {extra}") + " "
    expanded = " " + text.normalise(f"{description} {extra}") + " "

    scores: Dict[str, int] = {}
    for kw, code, weight in _KEYWORDS or []:
        needle = f" {kw} "
        if needle in hay or needle in expanded:
            scores[code] = scores.get(code, 0) + weight * 10 + len(kw)

    if not scores:
        # Nothing matched with word boundaries. Before giving up to GENERIC, try
        # multi-word keywords with their internal space removed - "GATEVALVE" for
        # "GATE VALVE", "CHECKVALVE" for "CHECK VALVE". This is real, common ERP
        # data-entry noise, not a hypothetical: a misclassification here sends the
        # record to GENERIC, which has almost no blocking attributes, which is
        # how items of very different sizes were able to auto-merge.
        hay_glued = hay.replace(" ", "")
        expanded_glued = expanded.replace(" ", "")
        for kw, code, weight in _KEYWORDS_GLUED or []:
            if kw in hay_glued or kw in expanded_glued:
                scores[code] = scores.get(code, 0) + weight * 10 + len(kw)

    engineered, reason = is_engineered(description)

    if not scores:
        spec = class_spec(GENERIC)
        return Classification(
            class_code=GENERIC, confidence=0.25, item_name=spec["item_name"],
            nsc=spec["nsc"], unspsc=spec["unspsc"], hsn=spec["hsn"], eclass=spec["eclass"],
            standardisable=not engineered,
            reason=reason or "no class keyword matched; classified as generic",
        )

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    margin = (best_score - runner) / best_score if best_score else 0.0
    confidence = round(min(0.55 + 0.45 * margin, 0.99), 4)

    spec = class_spec(best)
    return Classification(
        class_code=best, confidence=confidence, item_name=spec["item_name"],
        nsc=spec["nsc"], unspsc=spec["unspsc"], hsn=spec["hsn"], eclass=spec["eclass"],
        standardisable=not engineered,
        reason=reason or f"matched class keywords for {best}",
    )


def facets(class_code: str) -> dict:
    """The versioned classification facets attached to an NMC. Identity is permanent;
    classification is a view, which is why the code never rots."""
    spec = class_spec(class_code)
    return {
        "nsc": spec["nsc"],
        "unspsc": spec["unspsc"],
        "hsn": spec["hsn"],
        "eclass": spec["eclass"],
        "facet_version": data.class_dictionary().get("version", "1.0.0"),
    }

"""Pair features.

The reranker sees structure, not a sentence. Every feature here is something you
can say out loud in an evidence card: "seven of eight mandatory attributes matched
exactly, one matched through ISO 3506, none conflicted".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import blocking, classify, embed, kg, text

FEATURE_NAMES: Tuple[str, ...] = (
    "bias",
    "identity_key",            # same MPN or same vendor material number
    "attr_exact_ratio",        # share of shared attributes matching verbatim
    "attr_kg_ratio",           # share matching only through the standards graph
    "attr_conflict_count",     # hard conflicts (negative weight)
    "attr_missing_ratio",      # attributes present on one record only
    "mandatory_covered",       # share of the class's mandatory attributes on both
    "blocking_all_equal",      # every blocking attribute agrees
    "numeric_match",           # numeric attributes equal within tolerance
    "numeric_mismatch",        # numeric attributes differ (negative weight)
    "cosine",                  # bi-encoder similarity
    "token_jaccard",
    "trigram",
    "same_class",
    "service_unknown",         # a service qualifier on one side only (negative)
    "standard_conflict",       # different governing standards (negative)
)


def _attr_map(record: dict) -> Dict[str, str]:
    return {k: str(v) for k, v in (record.get("attributes") or {}).items() if v not in (None, "")}


def _numeric_map(record: dict) -> Dict[str, Optional[float]]:
    return {k: v for k, v in (record.get("numerics") or {}).items() if v is not None}


def compare_attributes(a: dict, b: dict) -> dict:
    """Attribute-by-attribute comparison, the raw material for both the feature
    vector and the evidence card. Nothing here is a black box."""
    cls = a.get("class_code") or b.get("class_code") or "GENERIC"
    av, bv = _attr_map(a), _attr_map(b)
    an, bn = _numeric_map(a), _numeric_map(b)

    rows: List[dict] = []
    for spec in classify.attribute_specs(cls):
        key = spec["key"]
        va, vb = av.get(key), bv.get(key)
        row = {
            "key": key,
            "label": spec.get("label", key.upper()),
            "a": va,
            "b": vb,
            "mandatory": bool(spec.get("mandatory")),
            "blocking": bool(spec.get("blocking")),
            "status": "missing",
            "citation": "",
            "caveat": "",
            "direction": "",
        }
        # Service qualification is evaluated even when one side is absent: absence
        # means UNKNOWN, never "not required". This check must come before the
        # generic missing-value branch or a NACE-qualified valve silently matches
        # an unqualified one.
        if spec.get("service"):
            verdict = kg.compare_service(va or "", vb or "")
            row["status"] = {"IDENTICAL": "exact", "UNKNOWN": "unknown"}.get(verdict.relation, "conflict")
            row["citation"], row["caveat"] = verdict.citation, verdict.caveat
            if va is None and vb is None:
                row["status"] = "absent_both"
        elif va is None and vb is None:
            row["status"] = "absent_both"
        elif va is None or vb is None:
            row["status"] = "missing"
            row["caveat"] = "stated on one record only"
        elif spec.get("standard"):
            verdict = kg.compare_standard(va, vb)
            row["status"] = {"IDENTICAL": "exact", "CONFLICT": "conflict"}.get(verdict.relation, "unknown")
            row["citation"], row["caveat"] = verdict.citation, verdict.caveat
        elif spec.get("numeric"):
            na, nb = an.get(key), bn.get(key)
            tol = float(spec.get("tolerance", 0.0))
            if na is None or nb is None:
                row["status"] = "exact" if text.basic(va) == text.basic(vb) else "conflict"
            elif abs(na - nb) <= tol:
                row["status"] = "exact"
            else:
                row["status"] = "conflict"
                row["caveat"] = f"{na:g} vs {nb:g} {spec.get('unit','')}".strip()
        elif spec.get("ordered"):
            verdict = kg.compare_ordered(spec["ordered"], va, vb)
            row["status"] = {
                "IDENTICAL": "exact", "EQUIVALENT": "kg_equiv",
                "SUBSTITUTABLE": "directed", "CONFLICT": "conflict", "UNKNOWN": "unknown",
            }[verdict.relation]
            row["citation"], row["caveat"], row["direction"] = verdict.citation, verdict.caveat, verdict.direction
            row["requires_equal"] = kg.order_requires_equal(spec["ordered"])
        elif spec.get("kg"):
            verdict = kg.compare_material(va, vb)
            row["status"] = {
                "IDENTICAL": "exact", "EQUIVALENT": "kg_equiv",
                "CONFLICT": "conflict", "UNKNOWN": "unknown",
            }[verdict.relation]
            row["citation"], row["caveat"] = verdict.citation, verdict.caveat
            row["partial"] = verdict.partial
            if row["status"] == "exact" and verdict.path and verdict.path[0] != va:
                row["resolved"] = verdict.path[0]
        else:
            row["status"] = "exact" if text.basic(va) == text.basic(vb) else "conflict"
        rows.append(row)

    present = [r for r in rows if r["status"] not in ("absent_both",)]
    conflicts = [r for r in rows if r["status"] == "conflict"]
    blocking_conflicts = [r for r in conflicts if r["blocking"]]
    return {
        "class_code": cls,
        "rows": rows,
        "present": len(present),
        "exact": len([r for r in rows if r["status"] == "exact"]),
        "kg_equiv": len([r for r in rows if r["status"] == "kg_equiv"]),
        "directed": [r for r in rows if r["status"] == "directed"],
        "unknown": len([r for r in rows if r["status"] == "unknown"]),
        "missing": len([r for r in rows if r["status"] == "missing"]),
        "conflicts": conflicts,
        "blocking_conflicts": blocking_conflicts,
    }


def identity_match(a: dict, b: dict) -> Tuple[bool, str]:
    ka = set(blocking.identity_keys(a))
    kb = set(blocking.identity_keys(b))
    shared = ka & kb
    if not shared:
        return False, ""
    key = sorted(shared)[0]
    if key.startswith("vmn:"):
        return True, (
            "both records carry a purchasing info record against the same vendor with "
            "the same vendor material number - the vendor has already resolved this entity"
        )
    return True, "both records carry the same manufacturer part number"


def vector(a: dict, b: dict, cosine: Optional[float] = None, cmp: Optional[dict] = None) -> List[float]:
    """The feature vector handed to the scorer."""
    cmp = cmp or compare_attributes(a, b)
    rows = cmp["rows"]
    cls = cmp["class_code"]

    comparable = [r for r in rows if r["status"] not in ("absent_both",)]
    n_cmp = max(len(comparable), 1)
    exact = cmp["exact"]
    kg_equiv = cmp["kg_equiv"]
    conflicts = len(cmp["conflicts"])
    missing = cmp["missing"]

    mandatory = classify.mandatory_keys(cls)
    mand_rows = [r for r in rows if r["key"] in mandatory]
    mand_covered = (
        len([r for r in mand_rows if r["status"] in ("exact", "kg_equiv", "directed")]) / len(mand_rows)
        if mand_rows else 0.0
    )

    blocking_rows = [r for r in rows if r["blocking"] and r["status"] != "absent_both"]
    blocking_all_equal = 1.0 if blocking_rows and all(
        r["status"] in ("exact", "kg_equiv") for r in blocking_rows
    ) else 0.0

    numeric_keys = {s["key"] for s in classify.attribute_specs(cls) if s.get("numeric")}
    num_rows = [r for r in rows if r["key"] in numeric_keys and r["status"] != "absent_both"]
    numeric_match = len([r for r in num_rows if r["status"] == "exact"]) / max(len(num_rows), 1)
    numeric_mismatch = len([r for r in num_rows if r["status"] == "conflict"]) / max(len(num_rows), 1)

    service_unknown = 1.0 if any(
        r["status"] == "unknown" for r in rows
        if r["key"] in {s["key"] for s in classify.attribute_specs(cls) if s.get("service")}
    ) else 0.0
    standard_conflict = 1.0 if any(
        r["status"] == "conflict" for r in rows
        if r["key"] in {s["key"] for s in classify.attribute_specs(cls) if s.get("standard")}
    ) else 0.0

    id_hit, _ = identity_match(a, b)
    cos = cosine if cosine is not None else 0.0
    da, db = a.get("description", ""), b.get("description", "")

    return [
        1.0,
        1.0 if id_hit else 0.0,
        exact / n_cmp,
        kg_equiv / n_cmp,
        float(conflicts),
        missing / n_cmp,
        mand_covered,
        blocking_all_equal,
        numeric_match,
        numeric_mismatch,
        cos,
        text.token_jaccard(da, db),
        text.trigram_similarity(da, db),
        1.0 if a.get("class_code") == b.get("class_code") else 0.0,
        service_unknown,
        standard_conflict,
    ]


def explain(weights: Sequence[float], feats: Sequence[float], top: int = 6) -> List[dict]:
    """Per-feature contribution, so a score is never an unexplained number."""
    contrib = [
        {"feature": name, "value": round(float(v), 4), "weight": round(float(w), 4),
         "contribution": round(float(v) * float(w), 4)}
        for name, v, w in zip(FEATURE_NAMES, feats, weights)
    ]
    contrib.sort(key=lambda c: -abs(c["contribution"]))
    return contrib[:top]

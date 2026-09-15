"""The evidence card.

The UI the domain expert on the jury will judge you by. Not a similarity score - a
structured argument they can disagree with.

Three things make it better than a SHAP plot: it names the standard, it separates
MISSING from CONFLICTING, and it offers a counterfactual. An engineer can argue with
it. That is the point.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import classify, constraints, features, render, uom

_STATUS_LABEL = {
    "exact": "exact",
    "kg_equiv": "std. equiv",
    "directed": "directional",
    "conflict": "conflict",
    "missing": "missing",
    "unknown": "unknown",
    "absent_both": "not stated",
}
_STATUS_TONE = {
    "exact": "ok",
    "kg_equiv": "warn",
    "directed": "acc",
    "conflict": "stop",
    "missing": "muted",
    "unknown": "warn",
    "absent_both": "muted",
}


def _record_view(rec: dict, redact: bool = False) -> dict:
    attrs = rec.get("attributes") or {}
    view = {
        "id": rec.get("id"),
        "cpse_code": rec.get("cpse_code"),
        "cpse_name": rec.get("cpse_name"),
        "plant": rec.get("plant"),
        "matnr": rec.get("matnr"),
        "description": rec.get("description"),
        "class_code": rec.get("class_code"),
        "golden_description": render.golden_description(rec.get("class_code", "GENERIC"), attrs),
        "uom": rec.get("uom"),
    }
    if not redact:
        view["unit_price"] = rec.get("unit_price")
        view["unit_price_base"] = rec.get("unit_price_base")
        view["stock_qty"] = rec.get("stock_qty")
    return view


def build(
    a: dict,
    b: dict,
    score: float,
    relation: Optional[constraints.Relation] = None,
    decision=None,
    cmp: Optional[dict] = None,
    weights: Optional[List[float]] = None,
    feature_vector: Optional[List[float]] = None,
    redact_commercial: bool = False,
) -> dict:
    """Assemble the full evidence card for one proposed pair."""
    cmp = cmp or features.compare_attributes(a, b)
    relation = relation or constraints.solve(a, b, cmp)

    rows: List[dict] = []
    for row in cmp["rows"]:
        if row["status"] == "absent_both":
            continue
        rows.append({
            "key": row["key"],
            "label": row["label"],
            "a": render.display_value(str(row["a"])) if row["a"] else None,
            "b": render.display_value(str(row["b"])) if row["b"] else None,
            "status": row["status"],
            "status_label": _STATUS_LABEL.get(row["status"], row["status"]),
            "tone": _STATUS_TONE.get(row["status"], "muted"),
            "mandatory": row["mandatory"],
            "blocking": row["blocking"],
            "citation": row.get("citation") or "",
            "caveat": row.get("caveat") or "",
            "direction": row.get("direction") or "",
        })

    # Unit of measure gets its own row: it is not an attribute of the item, it is a
    # property of how the CPSE issues it, and reconciling it is what makes the prices
    # comparable at all.
    uom_row = None
    if a.get("uom") or b.get("uom"):
        cmpu = uom.anomaly_ratio(
            a.get("unit_price") or 0, a.get("uom") or "", a.get("description") or "",
            b.get("unit_price") or 0, b.get("uom") or "", b.get("description") or "",
        )
        ra, rb = cmpu["a"], cmpu["b"]
        gap = cmpu["normalised_ratio"]
        uom_row = {
            "key": "uom",
            "label": "UOM",
            "a": f"{a.get('uom')} -> {ra['code']}",
            "b": f"{b.get('uom')} -> {rb['code']}",
            "status": "exact" if cmpu["comparable"] else "conflict",
            "status_label": "reconciled" if cmpu["comparable"] else "incomparable",
            "tone": "ok" if cmpu["comparable"] else "stop",
            "mandatory": False,
            "blocking": False,
            "citation": "UN/CEFACT Recommendation 20",
            "caveat": (
                (f"unit price gap {abs(gap-1)*100:.1f}% after normalisation" if gap else "")
                + ((" | " + ra["note"]) if ra.get("note") and not redact_commercial else "")
                + ((" | " + rb["note"]) if rb.get("note") and not redact_commercial else "")
            ).strip(" |"),
            "direction": "",
            "raw_ratio": None if redact_commercial else cmpu["raw_ratio"],
            "normalised_ratio": None if redact_commercial else cmpu["normalised_ratio"],
            "explained_by_uom": cmpu["explained_by_uom"],
        }
        rows.append(uom_row)

    why_not_identical = ""
    if relation.relation == constraints.EQUIVALENT:
        why_not_identical = "; ".join(relation.reasons) or "part-level identity is not proven"

    card = {
        "relation": relation.relation,
        "direction": relation.direction,
        "score": round(float(score), 4),
        "proof_strength": relation.proof_strength,
        "identity_key": relation.identity_key,
        "class_code": cmp["class_code"],
        "item_name": classify.class_spec(cmp["class_code"]).get("item_name"),
        "a": _record_view(a, redact_commercial),
        "b": _record_view(b, redact_commercial),
        "attributes": rows,
        "reasons": relation.reasons,
        "caveats": relation.caveats,
        "citations": relation.citations,
        "conflicts": [
            {"label": c["label"], "a": c["a"], "b": c["b"], "caveat": c.get("caveat", "")}
            for c in relation.conflicts
        ],
        "why_not_identical": why_not_identical,
        "counterfactual": constraints.counterfactual(a, b, cmp),
        "summary": {
            "exact": cmp["exact"],
            "std_equivalent": cmp["kg_equiv"],
            "directional": len(cmp["directed"]),
            "missing": cmp["missing"],
            "unknown": cmp["unknown"],
            "conflicts": len(cmp["conflicts"]),
        },
    }

    if decision is not None:
        card["decision"] = decision.to_dict() if hasattr(decision, "to_dict") else decision

    if weights and feature_vector:
        card["contributions"] = features.explain(weights, feature_vector)

    return card


def narrative(card: dict) -> str:
    """One-paragraph plain-language version, for a steward who wants the gist and for
    the ledger's human-readable reason field."""
    rel = card["relation"]
    s = card["summary"]
    bits = [
        f"{s['exact']} attribute(s) match exactly",
        f"{s['std_equivalent']} match through the standards graph" if s["std_equivalent"] else "",
        f"{s['directional']} differ directionally" if s["directional"] else "",
        f"{s['missing']} stated on one record only" if s["missing"] else "",
        f"{s['conflicts']} conflict" if s["conflicts"] else "",
    ]
    body = ", ".join(b for b in bits if b)
    head = f"Typed {rel}"
    if card.get("direction"):
        head += f" ({card['direction']})"
    tail = ""
    if card.get("why_not_identical"):
        tail = f" Not identical because: {card['why_not_identical']}."
    elif card.get("counterfactual"):
        tail = f" {card['counterfactual'][0].upper()}{card['counterfactual'][1:]}."
    return f"{head} at score {card['score']:.3f}. {body}.{tail}"

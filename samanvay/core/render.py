"""Golden description rendering.

If a language model writes the description that lands in a national material master,
the first hallucination is a safety incident and the project is over. So it doesn't.

The model extracts attributes; a DETERMINISTIC template renders the text, in the
noun-modifier form the codification world already uses (an approved item name
followed by an ordered characteristic list, per an ISO 22745-style identification
guide for the class).

Four properties fall out of doing it this way:
  deterministic          the same attribute set always renders the same string, so
                         re-running the pipeline never churns the master data
  diffable               a change is a change to one attribute, visible in the ledger
  language-independent   swap the template for Hindi; the attribute record is unchanged
  auditable              every token traces to an attribute, a source and a confidence
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import classify, kg

# Values stored as knowledge-graph family ids are rendered through their human label.
_FAMILY_DISPLAY = {
    "SS316": "STAINLESS STEEL 316",
    "SS316L": "STAINLESS STEEL 316L",
    "SS316TI": "STAINLESS STEEL 316TI",
    "SS304": "STAINLESS STEEL 304",
    "SS304L": "STAINLESS STEEL 304L",
    "CS_A105": "ASTM A105",
    "CS_A106B": "ASTM A106 GR.B",
    "CS_GENERIC": "CARBON STEEL",
    "IS2062_E250": "IS 2062 E250",
    "FC_A4": "ISO 3506 CLASS A4",
    "FC_A2": "ISO 3506 CLASS A2",
    "FC_A4_70": "A4-70",
    "FC_A4_80": "A4-80",
    "FC_A2_70": "A2-70",
    "PC_8_8": "PROPERTY CLASS 8.8",
    "PC_10_9": "PROPERTY CLASS 10.9",
    "PC_12_9": "PROPERTY CLASS 12.9",
    "B7": "ASTM A193 B7",
    "B7M": "ASTM A193 B7M",
    "WCB": "ASTM A216 WCB",
    "CF8M": "ASTM A351 CF8M",
    "ARMOUR_GENERIC": "ARMOURED",
    "ARMOUR_SWA": "STEEL WIRE ARMOURED",
    "ARMOUR_STA": "STEEL STRIP ARMOURED",
    "ARMOUR_NONE": "UNARMOURED",
    "NACE_MR0175": "NACE MR0175",
    "XLPE": "XLPE",
    "PVC": "PVC",
    "CU": "COPPER",
    "AL": "ALUMINIUM",
    "PTFE": "PTFE",
    "GRAPHITE": "GRAPHITE",
    "NBR": "NBR",
    "EPDM": "EPDM",
}


def display_value(value: str) -> str:
    if value in _FAMILY_DISPLAY:
        return _FAMILY_DISPLAY[value]
    if "+" in value:
        return " + ".join(_FAMILY_DISPLAY.get(p, p) for p in value.split("+"))
    return value


def golden_description(class_code: str, attributes: Dict[str, str]) -> str:
    """Render the standardised description. Pure function of (class, attributes)."""
    spec = classify.class_spec(class_code)
    item_name = spec.get("item_name", "ITEM")
    parts: List[str] = []
    for attr in spec.get("attributes", []):
        key = attr["key"]
        value = attributes.get(key)
        if value in (None, ""):
            continue
        label = attr.get("label", key.upper())
        unit = attr.get("unit")
        shown = display_value(str(value))
        if unit and not any(ch.isalpha() for ch in str(value)):
            shown = f"{shown} {unit}"
        parts.append(f"{label}={shown}")
    if not parts:
        return item_name
    return f"{item_name}: " + "; ".join(parts)


def short_description(class_code: str, attributes: Dict[str, str], max_attrs: int = 4) -> str:
    """A compact form for list views and dashboards."""
    spec = classify.class_spec(class_code)
    keys = [a["key"] for a in spec.get("attributes", []) if a.get("mandatory")] or \
           [a["key"] for a in spec.get("attributes", [])]
    bits = [display_value(str(attributes[k])) for k in keys[:max_attrs] if attributes.get(k)]
    return f"{spec.get('item_name','ITEM')} " + " ".join(bits) if bits else spec.get("item_name", "ITEM")


def trace(class_code: str, extraction_attributes: Dict[str, dict]) -> List[dict]:
    """Every token in the golden description traced back to an attribute, its source
    and its extraction confidence. This is the auditability claim, made concrete."""
    spec = classify.class_spec(class_code)
    rows: List[dict] = []
    for attr in spec.get("attributes", []):
        key = attr["key"]
        payload = extraction_attributes.get(key)
        if not payload:
            continue
        rows.append({
            "label": attr.get("label", key.upper()),
            "key": key,
            "rendered": display_value(str(payload.get("value", ""))),
            "raw": payload.get("value"),
            "source": payload.get("source"),
            "evidence": payload.get("evidence"),
            "confidence": payload.get("confidence"),
        })
    return rows


def render_in(language: str, class_code: str, attributes: Dict[str, str]) -> str:
    """Language is a property of the template, not of the data. Only the labels
    change; the attribute record behind them is identical."""
    labels_hi = {
        "THREAD": "धारा", "LENGTH": "लंबाई",
        "MATERIAL": "सामग्री", "GRADE": "ग्रेड",
        "SIZE": "आकार", "CLASS": "वर्ग",
        "STD": "मानक", "TYPE": "प्रकार",
    }
    if language.lower() not in ("hi", "hin", "hindi"):
        return golden_description(class_code, attributes)
    spec = classify.class_spec(class_code)
    parts: List[str] = []
    for attr in spec.get("attributes", []):
        value = attributes.get(attr["key"])
        if value in (None, ""):
            continue
        label = attr.get("label", attr["key"].upper())
        parts.append(f"{labels_hi.get(label, label)}={display_value(str(value))}")
    return f"{spec.get('item_name','ITEM')}: " + "; ".join(parts) if parts else spec.get("item_name", "ITEM")

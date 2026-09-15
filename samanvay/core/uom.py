"""Unit-of-measure algebra.

The slide nobody else has. Two CPSEs buy the same cable: one in M at Rs 520, the
other in DRUM at Rs 2,55,000. Raw, that is an apparent 490x price gap that corrupts
every spend chart built on it. Normalised through a real unit algebra it is 2%.

Demand aggregation is arithmetically impossible without this step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

from . import data, text


@dataclass
class UomResolution:
    raw: str
    code: str                 # UN/CEFACT Rec 20 code of the resolved base unit
    dimension: str            # length | mass | count | volume | area | time
    qty_per_uom: float        # how many base units one issue-unit contains
    base_code: str            # base unit code for the dimension
    is_pack: bool
    pack_source: str          # "explicit" | "default" | "" - where qty_per_uom came from
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_INDEX: Optional[Dict[str, Tuple[str, dict]]] = None
_PACK_INDEX: Optional[Dict[str, Tuple[str, dict]]] = None
_PATTERNS = None


def _build() -> None:
    global _INDEX, _PACK_INDEX, _PATTERNS
    if _INDEX is not None:
        return
    table = data.uom_table()
    idx: Dict[str, Tuple[str, dict]] = {}
    for code, spec in table["units"].items():
        for alias in spec["aliases"] + [code]:
            idx[text.basic(alias)] = (code, spec)
    pidx: Dict[str, Tuple[str, dict]] = {}
    for name, spec in table["packs"].items():
        for alias in spec["aliases"] + [name]:
            pidx[text.basic(alias)] = (name, spec)
    _INDEX, _PACK_INDEX = idx, pidx
    _PATTERNS = [re.compile(p) for p in table.get("pack_size_patterns", [])]


def bases() -> Dict[str, str]:
    return data.uom_table()["bases"]


def _find_pack_qty(description: str) -> Optional[Tuple[float, str]]:
    """Infer the content of a pack from the description: 'DRUM OF 500 M', '100 MTR/COIL',
    'BOX OF 100 NOS'. Returns (qty, unit_alias)."""
    _build()
    hay = text.clean(description)
    for pat in _PATTERNS or []:
        m = pat.search(hay)
        if m:
            groups = [g for g in m.groups() if g]
            if not groups:
                continue
            try:
                qty = float(groups[0])
            except (TypeError, ValueError):
                continue
            unit = groups[1] if len(groups) > 1 else ""
            if qty > 0:
                return qty, unit
    return None


def resolve(uom_raw: str, description: str = "") -> UomResolution:
    """Resolve an issue unit of measure to a base unit and a quantity per issue unit."""
    _build()
    raw = text.basic(uom_raw or "")
    tbl = data.uom_table()
    base_map = tbl["bases"]

    # 1. A plain unit: M, KG, NOS, LTR...
    if raw in (_INDEX or {}):
        code, spec = _INDEX[raw]
        dim = spec["dimension"]
        return UomResolution(
            raw=uom_raw or "",
            code=code,
            dimension=dim,
            qty_per_uom=float(spec["factor"]),
            base_code=base_map[dim],
            is_pack=False,
            pack_source="",
        )

    # 2. A pack: DRUM, COIL, BOX, PKT... content inferred from the description.
    if raw in (_PACK_INDEX or {}):
        name, spec = _PACK_INDEX[raw]
        found = _find_pack_qty(description)
        if found:
            qty, unit_alias = found
            unit_code, unit_spec = (_INDEX or {}).get(
                text.basic(unit_alias), (spec["default_unit"], None)
            )
            factor = float(unit_spec["factor"]) if unit_spec else 1.0
            dim = unit_spec["dimension"] if unit_spec else spec["dimension_hint"]
            return UomResolution(
                raw=uom_raw or "",
                code=unit_code,
                dimension=dim,
                qty_per_uom=qty * factor,
                base_code=base_map[dim],
                is_pack=True,
                pack_source="explicit",
                note=f"{name} content read from the description: {qty:g} {unit_alias or unit_code}",
            )
        dim = spec["dimension_hint"]
        default_unit = spec["default_unit"]
        unit_spec = tbl["units"].get(default_unit, {"factor": 1.0})
        return UomResolution(
            raw=uom_raw or "",
            code=default_unit,
            dimension=dim,
            qty_per_uom=float(spec["default_qty"]) * float(unit_spec.get("factor", 1.0)),
            base_code=base_map[dim],
            is_pack=True,
            pack_source="default",
            note=(
                f"{name} content not stated in the description; assumed "
                f"{spec['default_qty']:g} {default_unit}. Flag for steward confirmation."
            ),
        )

    # 3. Unknown. Treat as one countable unit but say so loudly.
    return UomResolution(
        raw=uom_raw or "",
        code="C62",
        dimension="count",
        qty_per_uom=1.0,
        base_code=base_map["count"],
        is_pack=False,
        pack_source="",
        note=f"unit of measure '{uom_raw}' is not in the table; treated as one unit",
    )


def base_unit_price(price: float | None, uom_raw: str, description: str = "") -> Tuple[Optional[float], UomResolution]:
    """Convert an issue-unit price into a price per base unit. This is the number that
    can legitimately be compared and aggregated across CPSEs."""
    res = resolve(uom_raw, description)
    if price is None:
        return None, res
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None, res
    if res.qty_per_uom <= 0:
        return None, res
    return round(price / res.qty_per_uom, 6), res


def comparable(a: UomResolution, b: UomResolution) -> bool:
    """Two records can only have their prices or quantities compared when their
    issue units reduce to the same physical dimension."""
    return a.dimension == b.dimension


def convert(qty: float, from_uom: str, to_uom: str, description: str = "") -> Optional[float]:
    a = resolve(from_uom, description)
    b = resolve(to_uom, description)
    if not comparable(a, b) or b.qty_per_uom == 0:
        return None
    return qty * a.qty_per_uom / b.qty_per_uom


def anomaly_ratio(price_a: float, uom_a: str, desc_a: str, price_b: float, uom_b: str, desc_b: str) -> dict:
    """Report the raw apparent price gap and the true normalised gap. Used by the
    analytics screen to surface UoM-driven false anomalies."""
    ba, ra = base_unit_price(price_a, uom_a, desc_a)
    bb, rb = base_unit_price(price_b, uom_b, desc_b)
    raw_ratio = None
    if price_a and price_b and min(price_a, price_b) > 0:
        raw_ratio = max(price_a, price_b) / min(price_a, price_b)
    norm_ratio = None
    if ba and bb and min(ba, bb) > 0 and comparable(ra, rb):
        norm_ratio = max(ba, bb) / min(ba, bb)
    return {
        "raw_ratio": round(raw_ratio, 3) if raw_ratio else None,
        "normalised_ratio": round(norm_ratio, 3) if norm_ratio else None,
        "a": ra.to_dict() | {"unit_price_base": ba},
        "b": rb.to_dict() | {"unit_price_base": bb},
        "comparable": comparable(ra, rb),
        "explained_by_uom": bool(raw_ratio and norm_ratio and raw_ratio > 3 and norm_ratio < 1.5),
    }

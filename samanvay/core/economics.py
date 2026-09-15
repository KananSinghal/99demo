"""The value model.

Half the field puts "Rs 5,000 crore saved" on a slide. The jury has discounted that
slide for a decade. This module does the opposite: it ships a transparent model with
every assumption exposed as a named parameter, and it reports what the corpus
actually measured.

Levers, in the order they should be presented:

  1. surplus redeployment   non-moving stock at A matching live demand at B through
                            the DIRECTED substitutable edge. Needs no assumptions at
                            all - the stock and the demand are both in the data. Lead
                            with this.
  2. inventory pooling      the square-root law. For an item independently stocked at
                            k locations, pooled safety stock scales as 1/sqrt(k), a
                            theoretical reduction of (1 - 1/sqrt(k)) - 50% at k=4.
                            Multiplied by a stated realisability factor, because the
                            theoretical figure is an upper bound.
  3. demand aggregation     report the VOLUME that becomes aggregatable, which is a
                            hard number. Never claim a fixed discount percentage.
  4. master-data upkeep     duplicate codes x annual cost per code. Smallest lever,
                            easiest to evidence. Good for credibility, bad as a headline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, List, Optional, Sequence

from .. import config


@dataclass
class Assumptions:
    carrying_rate: float = config.INVENTORY_CARRYING_RATE
    pooling_realisability: float = config.POOLING_REALISABILITY
    code_upkeep_per_year: float = config.CODE_UPKEEP_COST_PER_YEAR
    non_moving_days: int = config.NON_MOVING_DAYS
    service_z: float = 1.65               # ~95% service level
    currency: str = "INR"

    def to_dict(self) -> dict:
        return asdict(self)


def pooling_reduction(k: int) -> float:
    """Square-root law: theoretical safety-stock reduction from pooling k independent
    stocking points. 1 - 1/sqrt(k). k=2 -> 29.3%, k=4 -> 50%, k=9 -> 66.7%."""
    if k <= 1:
        return 0.0
    return 1.0 - 1.0 / math.sqrt(k)


def pooled_safety_stock(individual_values: Sequence[float]) -> dict:
    """Show the formula, because the materials people in the room will recognise it."""
    k = len([v for v in individual_values if v and v > 0])
    total = float(sum(v for v in individual_values if v))
    if k <= 1:
        return {"locations": k, "current": round(total, 2), "pooled": round(total, 2),
                "reduction_pct": 0.0, "saving": 0.0, "formula": "single location - nothing to pool"}
    reduction = pooling_reduction(k)
    pooled = total * (1.0 - reduction)
    return {
        "locations": k,
        "current": round(total, 2),
        "pooled": round(pooled, 2),
        "reduction_pct": round(reduction * 100, 2),
        "saving": round(total - pooled, 2),
        "formula": f"1 - 1/sqrt({k}) = {reduction:.4f}  (square-root law)",
    }


@dataclass
class Lever:
    name: str
    basis: str
    measured: bool
    value: float
    unit: str
    detail: dict = field(default_factory=dict)
    caveat: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def redeployment(nmc_groups: Sequence[dict], assumptions: Optional[Assumptions] = None) -> Lever:
    """Lever 1. Non-moving stock at one CPSE that could satisfy live demand at another.

    Directly measurable. No assumptions. This is the most concrete rupee figure the
    system can honestly produce, which is why it goes first.

    nmc_groups: [{nmc, class_code, members: [{cpse_code, stock_qty, unit_price_base,
                  last_movement_days, annual_demand}]}]
    """
    assumptions = assumptions or Assumptions()
    total = 0.0
    matches: List[dict] = []
    for group in nmc_groups:
        members = group.get("members") or []
        surplus = [
            m for m in members
            if (m.get("stock_qty") or 0) > 0
            and (m.get("last_movement_days") or 0) >= assumptions.non_moving_days
        ]
        demand = [m for m in members if (m.get("annual_demand") or 0) > 0]
        if not surplus or not demand:
            continue
        for src in surplus:
            for dst in demand:
                if src.get("cpse_code") == dst.get("cpse_code"):
                    continue
                qty = min(float(src.get("stock_qty") or 0), float(dst.get("annual_demand") or 0))
                price = float(dst.get("unit_price_base") or src.get("unit_price_base") or 0)
                value = qty * price
                if qty <= 0 or value <= 0:
                    continue
                total += value
                matches.append({
                    "nmc": group.get("nmc"),
                    "class_code": group.get("class_code"),
                    "from_cpse": src.get("cpse_code"),
                    "to_cpse": dst.get("cpse_code"),
                    "qty": round(qty, 3),
                    "unit_price_base": round(price, 4),
                    "value": round(value, 2),
                    "idle_days": src.get("last_movement_days"),
                    "relation": src.get("relation", "IDENTICAL"),
                })
    matches.sort(key=lambda m: -m["value"])
    return Lever(
        name="surplus redeployment",
        basis="non-moving stock at one CPSE matching live demand at another, through identical, equivalent or directed-substitutable relations",
        measured=True,
        value=round(total, 2),
        unit=assumptions.currency,
        detail={"opportunities": len(matches), "top": matches[:50]},
        caveat="measured from the corpus; realisation depends on transfer logistics and inter-CPSE settlement",
    )


def pooling(nmc_groups: Sequence[dict], assumptions: Optional[Assumptions] = None) -> Lever:
    """Lever 2. Safety-stock pooling under the square-root law."""
    assumptions = assumptions or Assumptions()
    theoretical = 0.0
    rows: List[dict] = []
    for group in nmc_groups:
        members = [m for m in (group.get("members") or []) if (m.get("stock_qty") or 0) > 0]
        by_cpse: Dict[str, float] = {}
        for m in members:
            value = float(m.get("stock_qty") or 0) * float(m.get("unit_price_base") or 0)
            by_cpse[m.get("cpse_code") or "?"] = by_cpse.get(m.get("cpse_code") or "?", 0.0) + value
        if len(by_cpse) < 2:
            continue
        calc = pooled_safety_stock(list(by_cpse.values()))
        theoretical += calc["saving"]
        rows.append({"nmc": group.get("nmc"), "class_code": group.get("class_code"), **calc})
    rows.sort(key=lambda r: -r["saving"])
    realisable = theoretical * assumptions.pooling_realisability
    return Lever(
        name="inventory pooling",
        basis="square-root law: pooled safety stock scales as 1/sqrt(k) across k independent stocking points",
        measured=False,
        value=round(realisable, 2),
        unit=assumptions.currency,
        detail={
            "theoretical_upper_bound": round(theoretical, 2),
            "realisability_factor": assumptions.pooling_realisability,
            "poolable_groups": len(rows),
            "top": rows[:50],
        },
        caveat=(
            "the square-root figure is a theoretical upper bound. Applied only to the "
            "genuinely poolable subset and multiplied by a stated realisability factor. "
            "Adjust the factor and the number moves - that is the point of shipping the model."
        ),
    )


def aggregation(nmc_groups: Sequence[dict], assumptions: Optional[Assumptions] = None) -> Lever:
    """Lever 3. Report the aggregatable VOLUME - a hard number - not a discount rate."""
    assumptions = assumptions or Assumptions()
    volume = 0.0
    rows: List[dict] = []
    for group in nmc_groups:
        members = group.get("members") or []
        cpses = {m.get("cpse_code") for m in members if (m.get("annual_demand") or 0) > 0}
        if len(cpses) < 2:
            continue
        spend = sum(
            float(m.get("annual_demand") or 0) * float(m.get("unit_price_base") or 0)
            for m in members
        )
        prices = [float(m.get("unit_price_base") or 0) for m in members if m.get("unit_price_base")]
        spread = (max(prices) / min(prices)) if prices and min(prices) > 0 else None
        volume += spend
        rows.append({
            "nmc": group.get("nmc"), "class_code": group.get("class_code"),
            "cpses": len(cpses), "annual_spend": round(spend, 2),
            "price_spread": round(spread, 3) if spread else None,
        })
    rows.sort(key=lambda r: -r["annual_spend"])
    return Lever(
        name="demand aggregation",
        basis="annual spend on items now demonstrably bought by two or more CPSEs under one national code",
        measured=True,
        value=round(volume, 2),
        unit=assumptions.currency,
        detail={"aggregatable_groups": len(rows), "top": rows[:50]},
        caveat=(
            "this is the VOLUME that becomes tenderable together, not a saving. "
            "We do not claim a discount percentage we cannot evidence."
        ),
    )


def upkeep(duplicate_codes: int, assumptions: Optional[Assumptions] = None) -> Lever:
    assumptions = assumptions or Assumptions()
    return Lever(
        name="master-data upkeep",
        basis="duplicate codes eliminated x annual cost of maintaining one material code",
        measured=False,
        value=round(duplicate_codes * assumptions.code_upkeep_per_year, 2),
        unit=assumptions.currency,
        detail={"duplicate_codes": duplicate_codes,
                "cost_per_code_per_year": assumptions.code_upkeep_per_year},
        caveat="smallest lever and the easiest to evidence; good for credibility, poor as a headline",
    )


def model(nmc_groups: Sequence[dict], duplicate_codes: int,
          assumptions: Optional[Assumptions] = None) -> dict:
    """The whole value model, with every assumption exposed."""
    assumptions = assumptions or Assumptions()
    levers = [
        redeployment(nmc_groups, assumptions),
        pooling(nmc_groups, assumptions),
        aggregation(nmc_groups, assumptions),
        upkeep(duplicate_codes, assumptions),
    ]
    measured = sum(l.value for l in levers if l.measured and l.name != "demand aggregation")
    modelled = sum(l.value for l in levers if not l.measured)
    return {
        "assumptions": assumptions.to_dict(),
        "levers": [l.to_dict() for l in levers],
        "measured_total": round(measured, 2),
        "modelled_total": round(modelled, 2),
        "statement": (
            "We are not quoting a national savings figure, because we cannot evidence one. "
            "This is the model with every assumption exposed, and the numbers our pilot "
            "corpus actually measured. Plug in real CPSE data and it tells you the truth - "
            "including if the truth is disappointing."
        ),
    }

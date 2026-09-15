"""CPSE -> sector registry.

The same bearing sits in a refinery store and a power-plant store under two different
codes. Cross-CPSE matching only pays off at national scale when it also crosses
SECTORS, so every analytics view needs to know which sector a CPSE belongs to.

Sector is a property of the organisation, not of a material record, so it lives here
rather than on source_material: no schema change, and a CPSE ingested through any
connector picks up its sector automatically. Unknown codes report as UNASSIGNED
rather than being guessed.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

OIL_GAS = "Oil & Gas"
POWER = "Power"
STEEL = "Steel"
MINING = "Mining"
HEAVY_ENG = "Heavy Engineering"
UNASSIGNED = "Unassigned"

SECTORS = (OIL_GAS, POWER, STEEL, MINING, HEAVY_ENG)

CPSE_SECTOR: Dict[str, str] = {
    "ONGC": OIL_GAS, "IOCL": OIL_GAS, "BPCL": OIL_GAS, "HPCL": OIL_GAS,
    "GAIL": OIL_GAS, "OIL": OIL_GAS, "CPCL": OIL_GAS,
    "NTPC": POWER, "NHPC": POWER, "PGCIL": POWER,
    "SAIL": STEEL, "RINL": STEEL,
    "CIL": MINING, "NMDC": MINING,
    "BHEL": HEAVY_ENG, "HEC": HEAVY_ENG,
}


def sector_of(cpse_code: str) -> str:
    return CPSE_SECTOR.get((cpse_code or "").strip().upper(), UNASSIGNED)


def sectors_of(cpse_codes: Iterable[str]) -> List[str]:
    return sorted({sector_of(c) for c in cpse_codes})

"""Non-SAP connectors.

A national registry that only works for SAP shops is not national. Oracle EBS and
Fusion, IFS, Ramco, and the spreadsheet-era entities all land on the same canonical
schema, so nothing downstream ever learns which ERP a record came from.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

CANONICAL_FIELDS = (
    "cpse_code", "cpse_name", "plant", "matnr", "description", "long_text", "uom",
    "unit_price", "currency", "stock_qty", "annual_demand", "last_movement_days",
    "mfr_name", "mfr_part_no", "vendor_id", "vendor_name", "vendor_matl_no",
)

# Column aliases seen in the wild. Extend this map, never the engine.
ALIASES: Dict[str, str] = {
    "material": "matnr", "material_code": "matnr", "item_code": "matnr", "code": "matnr",
    "part_number": "matnr", "stock_code": "matnr", "inventory_id": "matnr",
    "material_description": "description", "item_description": "description",
    "desc": "description", "short_text": "description", "name": "description",
    "unit": "uom", "uom_code": "uom", "base_uom": "uom", "issue_unit": "uom",
    "price": "unit_price", "rate": "unit_price", "moving_avg_price": "unit_price",
    "std_price": "unit_price", "valuation_price": "unit_price",
    "qty": "stock_qty", "quantity": "stock_qty", "on_hand": "stock_qty", "soh": "stock_qty",
    "demand": "annual_demand", "annual_consumption": "annual_demand", "consumption": "annual_demand",
    "manufacturer": "mfr_name", "make": "mfr_name", "brand": "mfr_name",
    "mpn": "mfr_part_no", "manufacturer_part": "mfr_part_no", "oem_part_no": "mfr_part_no",
    "supplier": "vendor_name", "supplier_id": "vendor_id", "vendor": "vendor_name",
    "supplier_part_no": "vendor_matl_no", "vendor_part": "vendor_matl_no",
    "org": "cpse_code", "organisation": "cpse_name", "organization": "cpse_name",
    "location": "plant", "site": "plant", "depot": "plant", "warehouse": "plant",
    "last_issue_days": "last_movement_days", "days_since_movement": "last_movement_days",
}

NUMERIC = {"unit_price", "stock_qty", "annual_demand", "last_movement_days"}


def normalise_row(row: Dict[str, object], cpse_code: str = "") -> dict:
    out: dict = {}
    for key, value in row.items():
        if value in (None, ""):
            continue
        k = str(key).strip().lower().replace(" ", "_").replace("-", "_")
        target = k if k in CANONICAL_FIELDS else ALIASES.get(k)
        if not target:
            continue
        out[target] = value
    if cpse_code:
        out.setdefault("cpse_code", cpse_code)
    for field in NUMERIC:
        if field in out:
            try:
                out[field] = float(str(out[field]).replace(",", "").strip())
            except (ValueError, TypeError):
                out.pop(field, None)
    return out


class CsvConnector:
    """Any CSV with any column names, mapped onto the canonical schema."""

    def __init__(self, path: str, cpse_code: str = "", cpse_name: str = ""):
        self.path = Path(path)
        self.cpse_code = cpse_code
        self.cpse_name = cpse_name

    def read(self, limit: Optional[int] = None) -> List[dict]:
        if not self.path.exists():
            raise FileNotFoundError(f"no such file: {self.path}")
        out: List[dict] = []
        with self.path.open("r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(8192)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            for row in csv.DictReader(fh, dialect=dialect):
                rec = normalise_row(row, self.cpse_code)
                if self.cpse_name:
                    rec.setdefault("cpse_name", self.cpse_name)
                if rec.get("description"):
                    out.append(rec)
                if limit and len(out) >= limit:
                    break
        return out

    def unmapped_columns(self) -> List[str]:
        """What we could not place. Surfaced rather than silently dropped, because a
        column nobody mapped is usually the one holding the specification."""
        with self.path.open("r", encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh), [])
        out = []
        for col in header:
            k = str(col).strip().lower().replace(" ", "_").replace("-", "_")
            if k not in CANONICAL_FIELDS and k not in ALIASES:
                out.append(col)
        return out


class JsonConnector:
    """A REST payload or a JSON export. Accepts a list, or an object with a 'records',
    'items', 'value' or 'd.results' envelope (the last is OData's)."""

    def __init__(self, payload, cpse_code: str = "", cpse_name: str = ""):
        self.payload = payload
        self.cpse_code = cpse_code
        self.cpse_name = cpse_name

    @staticmethod
    def from_file(path: str, cpse_code: str = "", cpse_name: str = "") -> "JsonConnector":
        return JsonConnector(json.loads(Path(path).read_text(encoding="utf-8")),
                             cpse_code, cpse_name)

    def _rows(self) -> Iterable[dict]:
        p = self.payload
        if isinstance(p, list):
            return p
        if isinstance(p, dict):
            for key in ("records", "items", "value", "data", "rows"):
                if isinstance(p.get(key), list):
                    return p[key]
            d = p.get("d")
            if isinstance(d, dict) and isinstance(d.get("results"), list):
                return d["results"]
        return []

    def read(self, limit: Optional[int] = None) -> List[dict]:
        out: List[dict] = []
        for row in self._rows():
            if not isinstance(row, dict):
                continue
            rec = normalise_row(row, self.cpse_code)
            if self.cpse_name:
                rec.setdefault("cpse_name", self.cpse_name)
            if rec.get("description"):
                out.append(rec)
            if limit and len(out) >= limit:
                break
        return out

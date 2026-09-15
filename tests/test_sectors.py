"""Sector registry and the cross-sector analytics rollup."""

import unittest

from . import _boot  # noqa: F401

from samanvay.api.service import cross_sector
from samanvay.core import sectors


class SectorRegistry(unittest.TestCase):
    def test_known_and_unknown(self):
        self.assertEqual(sectors.sector_of("ntpc"), sectors.POWER)
        self.assertEqual(sectors.sector_of("CPCL"), sectors.OIL_GAS)
        self.assertEqual(sectors.sector_of("XYZ"), sectors.UNASSIGNED)


class CrossSector(unittest.TestCase):
    def test_only_multi_sector_groups_count(self):
        by_cpse = [{"cpse_code": "IOCL", "sector": sectors.OIL_GAS, "materials": 10},
                   {"cpse_code": "BPCL", "sector": sectors.OIL_GAS, "materials": 5},
                   {"cpse_code": "NTPC", "sector": sectors.POWER, "materials": 7}]
        groups = [
            {"members": [{"cpse_code": "IOCL", "annual_demand": 10, "unit_price": 5},
                         {"cpse_code": "NTPC", "annual_demand": 2, "unit_price": 5}]},
            {"members": [{"cpse_code": "IOCL"}, {"cpse_code": "BPCL"}]},
        ]
        out = cross_sector(groups, by_cpse)
        self.assertEqual(out["cross_sector_codes"], 1)
        self.assertEqual(out["cross_sector_spend"], 60.0)
        self.assertEqual(out["pairs"], [{"sectors": [sectors.OIL_GAS, sectors.POWER], "codes": 1}])
        oil = next(s for s in out["by_sector"] if s["sector"] == sectors.OIL_GAS)
        self.assertEqual((oil["materials"], oil["shared_codes"]), (15, 1))


if __name__ == "__main__":
    unittest.main()

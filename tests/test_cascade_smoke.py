"""A small, fast, fully in-memory run of the nine-stage cascade over a
hand-built corpus of ten records (no database, no corpus file). This is not a
substitute for scripts/benchmark.py's precision/recall numbers against the
full synthetic ground truth -- it is a fast smoke test that the pipeline runs
end to end without exceptions and gets the obviously-right answers on a
handful of cases chosen to cover a duplicate, a near-duplicate, a class
mismatch, a quarantine case, and an exclusion case in one pass.
"""
import unittest

from . import _boot  # noqa: F401
from samanvay.core import cascade, conformal, constraints


RECORDS = [
    {"id": 1, "cpse_code": "AAA", "description": "HEX BOLT M12X60 SS316 FULL THD",
     "uom": "NO", "unit_price": 22.0},
    {"id": 2, "cpse_code": "BBB", "description": "BOLT,HEXAGONAL,M12 X 60MM,SS316,ISO4017",
     "uom": "NO", "unit_price": 23.0},
    {"id": 3, "cpse_code": "CCC", "description": "HEX BOLT M16X60 SS316 FULL THD",
     "uom": "NO", "unit_price": 28.0},
    {"id": 4, "cpse_code": "AAA", "description": "BEARING 6205-2RS",
     "uom": "NO", "unit_price": 240.0},
    {"id": 5, "cpse_code": "BBB", "description": "BRG 6205 2RS",
     "uom": "NO", "unit_price": 235.0},
    {"id": 6, "cpse_code": "CCC", "description": "BEARING 6206-2RS",
     "uom": "NO", "unit_price": 310.0},
    {"id": 7, "cpse_code": "AAA", "description": "PUMP CASING AS PER DRAWING NO 4471-B",
     "uom": "NO", "unit_price": 185000.0},
    {"id": 8, "cpse_code": "BBB", "description": "MISC",
     "uom": "NO", "unit_price": 50.0},
    {"id": 9, "cpse_code": "AAA", "description": 'GATE VALVE 6" 150# WCB FLGD RF HW NACE MR0175',
     "uom": "NO", "unit_price": 15000.0},
    {"id": 10, "cpse_code": "BBB", "description": 'GATE VALVE 6" 150# WCB FLGD RF HW',
     "uom": "NO", "unit_price": 14800.0},
]


class TestCascadeSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = cascade.run(RECORDS, train=True, mine_lexicon=False)

    def test_runs_without_exception_and_returns_stats(self):
        self.assertIn("funnel", self.result.stats)
        self.assertIn("tiers", self.result.stats)
        self.assertIn("relations", self.result.stats)

    def test_engineered_item_is_excluded_not_scored(self):
        excluded_ids = {row["id"] for row in self.result.excluded}
        self.assertIn(7, excluded_ids)

    def test_low_information_item_is_quarantined_not_guessed_at(self):
        quarantined_ids = {row["id"] for row in self.result.quarantined}
        self.assertIn(8, quarantined_ids)

    def test_canonical_pool_excludes_quarantined_and_excluded(self):
        canonical_ids = {c["id"] for c in self.result.canonical}
        self.assertNotIn(7, canonical_ids)
        self.assertNotIn(8, canonical_ids)
        self.assertIn(1, canonical_ids)

    def test_true_duplicate_pair_is_found_and_typed_mergeable(self):
        pair = self._find_pair(1, 2)
        self.assertIsNotNone(pair, "records 1 and 2 (same bolt) must reach the scorer")
        self.assertIn(pair["relation"], constraints.MERGEABLE)

    def test_near_duplicate_bearing_pair_is_found_and_typed_mergeable(self):
        pair = self._find_pair(4, 5)
        self.assertIsNotNone(pair, "records 4 and 5 (same bearing) must reach the scorer")
        self.assertIn(pair["relation"], constraints.MERGEABLE)

    def test_different_bore_bearing_pair_is_distinct(self):
        pair = self._find_pair(4, 6)
        if pair is not None:
            self.assertEqual(pair["relation"], constraints.DISTINCT)

    def test_nace_vs_unqualified_valve_is_undetermined_and_reviewed(self):
        pair = self._find_pair(9, 10)
        self.assertIsNotNone(pair, "the NACE pair must reach the scorer despite differing text")
        self.assertEqual(pair["relation"], constraints.UNDETERMINED)
        self.assertEqual(pair["tier"], conformal.REVIEW)

    def test_no_pair_across_bolt_and_bearing_classes(self):
        self.assertIsNone(self._find_pair(1, 4))

    def _find_pair(self, a_id, b_id):
        for p in self.result.proposals:
            ids = {int(p["a_id"]), int(p["b_id"])}
            if ids == {a_id, b_id}:
                return p
        return None


if __name__ == "__main__":
    unittest.main()

"""Unit tests for samanvay.core.constraints.solve() -- the typed-relation
solver. Every case here is a named adversarial pair from the design blueprint
and the judge Q&A bank, run through the real canonicalise() -> solve()
pipeline (not hand-built feature dicts), each one a regression test for a bug
that was actually caught and fixed while building this:

  * M12 vs 1/2"-13UNC:  metric and unified threads of near-identical diameter
                         must be DISTINCT, not a same-family size VARIANT_OF.
  * 6205 vs 6206:        one bore-size digit apart, must be DISTINCT.
  * SWA vs ARMOURED:     a specific construction of a generic material must be
                         EQUIVALENT through the standards graph.
  * SWA vs UNARMOURED:   armoured and unarmoured must never match.
  * NACE vs unqualified: sour-service qualification is non-inferable -- absence
                         means UNKNOWN, never "not required".
"""
import unittest

from . import _boot  # noqa: F401
from samanvay.core import constraints, features
from samanvay.core.cascade import canonicalise


def rec(rid, description, **kw):
    payload = {"id": rid, "cpse_code": "X", "description": description,
               "uom": "NO", "unit_price": 100.0}
    payload.update(kw)
    return canonicalise(payload)


def solve(desc_a, desc_b):
    a, b = rec(1, desc_a), rec(2, desc_b)
    cmp = features.compare_attributes(a, b)
    return constraints.solve(a, b, cmp)


class TestThreadSeries(unittest.TestCase):
    def test_metric_vs_unified_thread_is_distinct(self):
        rel = solve(
            "HEX BOLT M12X60 SS316 FULL THD",
            'HEX BOLT 1/2"-13UNC X 2" SS316 FULL THD',
        )
        self.assertEqual(rel.relation, constraints.DISTINCT)
        self.assertEqual(rel.blocked_by, "thread")

    def test_same_metric_diameter_different_house_style_is_identical(self):
        rel = solve(
            "HEX BOLT M12X60 SS316 FULL THD",
            "BOLT,HEXAGONAL,M12 X 60MM,SS316,ISO4017",
        )
        self.assertEqual(rel.relation, constraints.IDENTICAL)

    def test_same_series_different_diameter_is_variant_not_distinct(self):
        # A single blocking-attribute conflict (diameter), both sides on the
        # SAME thread series, with everything else matching: "same family,
        # different size" -- rolled up for spend, never auto-merged. This is
        # the counterpart to the metric-vs-unified case above, which is a
        # different SERIES and must stay DISTINCT.
        rel = solve("HEX BOLT M12X60 SS316 FULL THD", "HEX BOLT M16X60 SS316 FULL THD")
        self.assertEqual(rel.relation, constraints.VARIANT_OF)


class TestBearingBore(unittest.TestCase):
    def test_one_designation_digit_apart_is_distinct(self):
        rel = solve("BEARING 6205-2RS", "BEARING 6206-2RS")
        self.assertEqual(rel.relation, constraints.DISTINCT)

    def test_same_designation_different_notation_is_identical(self):
        rel = solve("BEARING 6205-2RS", "BRG 6205 2RS")
        self.assertEqual(rel.relation, constraints.IDENTICAL)


class TestCableArmour(unittest.TestCase):
    def test_swa_equivalent_to_generic_armoured(self):
        rel = solve(
            "CABLE XLPE 1.1KV 3C X 95 SQMM CU SWA",
            "CABLE XLPE 1.1KV 3C X 95 SQMM CU ARMOURED",
        )
        self.assertEqual(rel.relation, constraints.EQUIVALENT)

    def test_swa_vs_unarmoured_is_distinct(self):
        rel = solve(
            "CABLE XLPE 1.1KV 3C X 95 SQMM CU SWA",
            "CABLE XLPE 1.1KV 3C X 95 SQMM CU UNARMOURED",
        )
        self.assertEqual(rel.relation, constraints.DISTINCT)
        self.assertEqual(rel.blocked_by, "armour")


class TestNonInferableService(unittest.TestCase):
    def test_nace_stated_on_only_one_side_is_undetermined(self):
        rel = solve(
            'GATE VALVE 6" 150# WCB FLGD RF HW NACE MR0175',
            'GATE VALVE 6" 150# WCB FLGD RF HW',
        )
        self.assertEqual(rel.relation, constraints.UNDETERMINED)
        self.assertEqual(rel.blocked_by, "service_qualifier")

    def test_nace_stated_on_both_sides_matches_normally(self):
        rel = solve(
            'GATE VALVE 6" 150# WCB FLGD RF HW NACE MR0175',
            'GATE VALVE 6" 150# WCB FLGD RF HW NACE MR0175',
        )
        self.assertEqual(rel.relation, constraints.IDENTICAL)

    def test_neither_side_qualified_matches_normally(self):
        rel = solve(
            'GATE VALVE 6" 150# WCB FLGD RF HW',
            'GATE VALVE 6" 150# WCB FLGD RF HW',
        )
        self.assertEqual(rel.relation, constraints.IDENTICAL)


class TestDirectedSubstitutability(unittest.TestCase):
    def test_higher_schedule_substitutes_lower_directed(self):
        a = rec(1, 'PIPE 6" SCH 80 A106B SMLS BW', uom="M")
        b = rec(2, 'PIPE 6" SCH 40 A106B SMLS BW', uom="M")
        rel = constraints.solve(a, b)
        self.assertEqual(rel.relation, constraints.SUBSTITUTABLE)
        self.assertEqual(rel.direction, "a->b")

    def test_direction_flips_when_argument_order_flips(self):
        a = rec(1, 'PIPE 6" SCH 40 A106B SMLS BW', uom="M")
        b = rec(2, 'PIPE 6" SCH 80 A106B SMLS BW', uom="M")
        rel = constraints.solve(a, b)
        self.assertEqual(rel.relation, constraints.SUBSTITUTABLE)
        self.assertEqual(rel.direction, "b->a")

    def test_different_nominal_size_is_never_substitutable(self):
        a = rec(1, 'PIPE 6" SCH 40 A106B SMLS BW', uom="M")
        b = rec(2, 'PIPE 8" SCH 40 A106B SMLS BW', uom="M")
        rel = constraints.solve(a, b)
        self.assertNotEqual(rel.relation, constraints.SUBSTITUTABLE)


class TestClassMismatch(unittest.TestCase):
    def test_different_classes_are_distinct_and_never_scored(self):
        rel = solve("HEX BOLT M12X60 SS316 FULL THD", "BEARING 6205-2RS")
        self.assertEqual(rel.relation, constraints.DISTINCT)
        self.assertEqual(rel.blocked_by, "class")


if __name__ == "__main__":
    unittest.main()

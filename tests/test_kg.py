"""Unit tests for samanvay.core.kg -- the standards knowledge graph.

test_short_alias_is_whole_token_only is a named regression test: an earlier
version matched alias strings as substrings, so "BALL BEARING" spuriously
matched the "AL" (aluminium) alias. The fix (_SHORT_ALIAS) restricts aliases
of length <= 3 to whole-token matches only.
"""
import unittest

from . import _boot  # noqa: F401
from samanvay.core import kg


class TestFamilyResolution(unittest.TestCase):
    def test_short_alias_is_whole_token_only(self):
        # "AL" (aluminium) must never match inside "BALL BEARING".
        self.assertIsNone(kg.family_of("BALL BEARING 6205-2RS DEEP GROOVE"))

    def test_short_alias_matches_as_a_real_token(self):
        # But a genuine standalone "AL" token should still resolve.
        fam = kg.family_of("SLEEVE AL 25MM")
        self.assertIsNotNone(fam)

    def test_longer_alias_matches_as_substring(self):
        # "A106 GR.B" and "A106B" are both long enough to be matched inside a
        # full free-text description, not just as a whole token.
        self.assertEqual(kg.family_of("PIPE ASTM A106 GR.B SMLS BW"), kg.family_of("A106B"))

    def test_unknown_designation_is_none(self):
        self.assertIsNone(kg.family_of("XYZNOTAMATERIAL123"))


class TestCompareMaterial(unittest.TestCase):
    def test_verbatim_match_is_identical(self):
        v = kg.compare_material("SS316", "SS316")
        self.assertEqual(v.relation, kg.IDENTICAL)

    def test_same_family_different_alias_is_identical(self):
        # "UNS S31600" is the unified numbering system designation for the
        # same SS316 family node -- a different alias, the same steel.
        v = kg.compare_material("SS316", "UNS S31600")
        self.assertEqual(v.relation, kg.IDENTICAL)

    def test_fastener_property_class_is_equivalent_not_identical_to_base_grade(self):
        # A4-70 (ISO 3506 stainless fastener property class) gradeSatisfiedBy
        # SS316: a directed, one-to-many mapping. Real equivalence, but it does
        # not prove part-level identity, so this must be EQUIVALENT with a
        # partial caveat, never silently upgraded to IDENTICAL.
        v = kg.compare_material("SS316", "A4-70")
        self.assertEqual(v.relation, kg.EQUIVALENT)
        self.assertTrue(v.partial)

    def test_unrelated_materials_conflict(self):
        v = kg.compare_material("SS316", "CS_A105")
        self.assertIn(v.relation, (kg.CONFLICT, kg.UNKNOWN))

    def test_missing_on_one_side_is_unknown(self):
        v = kg.compare_material("SS316", "")
        self.assertEqual(v.relation, kg.UNKNOWN)


class TestOrderedAttributes(unittest.TestCase):
    def test_higher_schedule_substitutes_lower(self):
        v = kg.compare_ordered("pipe_schedule", "SCH80", "SCH40")
        self.assertEqual(v.relation, kg.SUBSTITUTABLE)
        self.assertEqual(v.direction, "a->b")

    def test_direction_flips_with_argument_order(self):
        v = kg.compare_ordered("pipe_schedule", "SCH40", "SCH80")
        self.assertEqual(v.relation, kg.SUBSTITUTABLE)
        self.assertEqual(v.direction, "b->a")

    def test_same_position_is_equivalent(self):
        v = kg.compare_ordered("pipe_schedule", "SCH40", "SCH40")
        self.assertEqual(v.relation, kg.IDENTICAL)

    def test_value_off_scale_is_unknown(self):
        v = kg.compare_ordered("pipe_schedule", "SCH40", "NOT_A_SCHEDULE")
        self.assertEqual(v.relation, kg.UNKNOWN)


class TestServiceQualifiers(unittest.TestCase):
    def test_both_qualified_same_way_is_identical(self):
        v = kg.compare_service("NACE MR0175", "SOUR SERVICE")
        self.assertEqual(v.relation, kg.IDENTICAL)

    def test_neither_qualified_is_identical(self):
        v = kg.compare_service("", "")
        self.assertEqual(v.relation, kg.IDENTICAL)

    def test_qualified_on_only_one_side_is_unknown_not_conflict(self):
        # This is the non-inferable rule: absence means unknown, never "not required".
        v = kg.compare_service("NACE MR0175", "")
        self.assertEqual(v.relation, kg.UNKNOWN)


class TestStandards(unittest.TestCase):
    def test_superseded_standard_maps_to_successor(self):
        canon, note = kg.canonical_standard("DIN 933")
        self.assertNotEqual(canon, "DIN 933")
        self.assertTrue(note)

    def test_compare_standard_after_supersession_matches(self):
        v = kg.compare_standard("DIN 933", "ISO 4017")
        self.assertEqual(v.relation, kg.IDENTICAL)


if __name__ == "__main__":
    unittest.main()

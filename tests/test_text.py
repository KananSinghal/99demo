"""Unit tests for samanvay.core.text -- the normalisation primitives everything
else is built on. Two of these lock in real bugs found while building the
corpus: `/` being stripped (which silently turned every imperial fraction like
1/2" into 2") and glued cable-core notation ("3CX95") not being split so the
blocker and the embedder see the same tokens as the spaced-out house styles.
"""
import unittest

from . import _boot  # noqa: F401
from samanvay.core import text


class TestClean(unittest.TestCase):
    def test_preserves_imperial_fraction(self):
        cleaned = text.clean('HEX BOLT 1/2"-13UNC X 2" SS316')
        self.assertIn("1/2", cleaned)
        self.assertNotIn(" 2\"-13UNC", cleaned, "the leading '1' must survive")

    def test_preserves_decimal_thread_pitch(self):
        self.assertIn("1.75", text.clean("M12X1.75 SS316"))

    def test_splits_glued_cable_core_notation(self):
        cleaned = text.clean("CABLE 3CX95 SQMM CU SWA")
        self.assertIn("3C X 95", cleaned)

    def test_separates_glued_units(self):
        cleaned = text.clean("HOSE 25MM PN16")
        self.assertIn("25 MM", cleaned)

    def test_collapses_separators_and_case(self):
        self.assertEqual(text.clean("bolt,  hex;head"), "BOLT HEX HEAD")

    def test_idempotent(self):
        once = text.clean('GATE VALVE 6" 150# WCB FLGD RF HW')
        twice = text.clean(once)
        self.assertEqual(once, twice)


class TestSimilarity(unittest.TestCase):
    def test_token_jaccard_identical_is_one(self):
        self.assertEqual(text.token_jaccard("HEX BOLT M12", "HEX BOLT M12"), 1.0)

    def test_token_jaccard_disjoint_is_zero(self):
        self.assertEqual(text.token_jaccard("HEX BOLT M12", "BALL BEARING 6205"), 0.0)

    def test_trigram_similarity_rewards_near_misses(self):
        near = text.trigram_similarity("HEXAGONAL BOLT M12X60", "HEX BOLT M12X60")
        far = text.trigram_similarity("HEXAGONAL BOLT M12X60", "BALL BEARING 6205")
        self.assertGreater(near, far)

    def test_levenshtein_basic(self):
        self.assertEqual(text.levenshtein("6205", "6206"), 1)
        self.assertEqual(text.levenshtein("ABC", "ABC"), 0)
        self.assertEqual(text.levenshtein("", "ABC"), 3)


class TestCompleteness(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(text.completeness(""), 0.0)

    def test_richer_description_scores_higher(self):
        low = text.completeness("ITEM")
        high = text.completeness("HEX BOLT M12X60 SS316 FULL THD ISO4017 A4-70")
        self.assertGreater(high, low)

    def test_bounded_zero_to_one(self):
        score = text.completeness(
            "HEX BOLT M12X60 SS316 FULL THD ISO4017 A4-70 SPARE PARTS QTY 500 LOT A"
        )
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()

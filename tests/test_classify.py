"""Unit tests for samanvay.core.classify -- material class assignment.

test_glued_compound_keyword_still_classifies is a named regression test for a
real bug the benchmark script caught: "GATEVALVE" (no space -- ordinary ERP
data-entry noise) did not match the word-boundary keyword "GATE VALVE" or the
bare "VALVE" keyword, so it fell through to GENERIC, which has almost no
blocking attributes. That silently let items of completely different sizes
and pressure classes auto-merge, because GENERIC never compared their size or
class attributes. Fixed by adding a space-stripped fallback match for
multi-word keywords, used only when word-boundary matching finds nothing.
"""
import unittest

from . import _boot  # noqa: F401
from samanvay.core import classify


class TestGluedKeywords(unittest.TestCase):
    def test_glued_compound_keyword_still_classifies(self):
        for glued in ("GATEVALVE", "CHECKVALVE", "BALLVALVE", "GLOBEVALVE"):
            with self.subTest(glued=glued):
                cls = classify.classify(f'{glued} 6" CL300 BODY A216 WCB FLGD RF HAND WHEEL')
                self.assertEqual(cls.class_code, "VALVE")

    def test_spaced_keyword_still_classifies_as_before(self):
        cls = classify.classify('GATE VALVE 6" 150# WCB FLGD RF HW')
        self.assertEqual(cls.class_code, "VALVE")

    def test_short_single_word_keyword_is_not_glue_matched(self):
        # "NUT" is a single-word keyword; the glued fallback only applies to
        # multi-word keywords ("HEX NUT" etc.), so an unrelated word that
        # merely contains "NUT" must not misclassify as FASTENER_NUT.
        cls = classify.classify("COCONUT FIBRE MATTING ROLL")
        self.assertNotEqual(cls.class_code, "FASTENER_NUT")


class TestClassifyBasics(unittest.TestCase):
    def test_unmatched_text_is_generic(self):
        cls = classify.classify("ZZZ QQQ NOT A REAL DESCRIPTION AT ALL")
        self.assertEqual(cls.class_code, classify.GENERIC)
        self.assertEqual(cls.confidence, 0.25)

    def test_longer_keyword_wins_over_shorter_substring_keyword(self):
        # "GATE VALVE" (weight 2) must outrank the bare "VALVE" (weight 1).
        cls = classify.classify('GATE VALVE 6" 150# WCB FLGD RF HW')
        self.assertEqual(cls.class_code, "VALVE")
        self.assertGreater(cls.confidence, 0.5)

    def test_engineered_to_order_is_not_standardisable(self):
        cls = classify.classify("PUMP CASING AS PER DRAWING NO 4471-B")
        self.assertFalse(cls.standardisable)

    def test_mandatory_and_blocking_keys_are_subsets_of_attribute_specs(self):
        specs = {s["key"] for s in classify.attribute_specs("VALVE")}
        self.assertTrue(set(classify.mandatory_keys("VALVE")) <= specs)
        self.assertTrue(set(classify.blocking_keys("VALVE")) <= specs)


if __name__ == "__main__":
    unittest.main()

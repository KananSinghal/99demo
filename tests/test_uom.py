"""Unit tests for samanvay.core.uom -- unit-of-measure algebra and the
UoM-driven price-anomaly detector. test_drum_vs_metre_anomaly_is_fully_explained
is the flagship demo case from the blueprint: two records that look like a 500x
price outlier are, once the pack size is read out of the free text, the same
price per metre.
"""
import unittest

from . import _boot  # noqa: F401
from samanvay.core import uom


class TestResolve(unittest.TestCase):
    def test_plain_unit_resolves(self):
        r = uom.resolve("M", "")
        self.assertEqual(r.dimension, "length")
        self.assertEqual(r.qty_per_uom, 1.0)

    def test_pack_size_read_from_description(self):
        r = uom.resolve("DRUM", "CABLE XLPE 1.1KV 3C X 95 SQMM CU SWA DRUM OF 500 M")
        self.assertTrue(r.is_pack)
        self.assertEqual(r.qty_per_uom, 500.0)
        self.assertEqual(r.dimension, "length")

    def test_unknown_unit_falls_back_to_count(self):
        r = uom.resolve("ZZZ", "SOME ITEM")
        self.assertEqual(r.dimension, "count")
        self.assertFalse(r.is_pack)


class TestComparableAndConvert(unittest.TestCase):
    def test_same_dimension_is_comparable(self):
        a = uom.resolve("M", "")
        b = uom.resolve("CMT", "")
        self.assertTrue(uom.comparable(a, b))

    def test_different_dimension_is_not_comparable(self):
        a = uom.resolve("M", "")
        b = uom.resolve("KG", "")
        self.assertFalse(uom.comparable(a, b))

    def test_convert_metres_to_centimetres(self):
        out = uom.convert(2, "M", "CMT")
        self.assertAlmostEqual(out, 200.0, places=3)


class TestAnomalyRatio(unittest.TestCase):
    def test_drum_vs_metre_anomaly_is_fully_explained(self):
        r = uom.anomaly_ratio(
            21000, "DRUM", "CABLE XLPE 1.1KV 3C X 95 SQMM CU SWA DRUM OF 500 M",
            42, "M", "CABLE XLPE 1.1KV 3C X 95 SQMM CU SWA",
        )
        self.assertAlmostEqual(r["raw_ratio"], 500.0, places=1)
        self.assertAlmostEqual(r["normalised_ratio"], 1.0, places=1)
        self.assertTrue(r["explained_by_uom"])
        self.assertTrue(r["comparable"])

    def test_genuinely_different_prices_are_not_explained_away(self):
        r = uom.anomaly_ratio(
            100, "NO", "HEX BOLT M12X60 SS316",
            5000, "NO", "HEX BOLT M12X60 SS316",
        )
        self.assertFalse(r["explained_by_uom"])
        self.assertAlmostEqual(r["normalised_ratio"], 50.0, places=1)


if __name__ == "__main__":
    unittest.main()

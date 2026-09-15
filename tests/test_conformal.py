"""Unit tests for samanvay.core.conformal -- calibrated accept/review/reject
decisions. These pin down the tier logic that the auto-accept guarantee
depends on: a hard conflict always rejects, a directional or size-variant
relation always goes to review (never merges on its own), and a
pressure-boundary class never auto-accepts on text similarity alone, only
with a real identity key.
"""
import random
import unittest

from . import _boot  # noqa: F401
from samanvay import config
from samanvay.core import conformal


def _fitted_calibrator(class_code="VALVE", n=40, seed=7):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        rows.append((class_code, rng.uniform(0.75, 0.99), 1))
        rows.append((class_code, rng.uniform(0.01, 0.30), 0))
    cal = conformal.Calibrator()
    cal.fit(rows)
    return cal


class TestBlockedAndUndetermined(unittest.TestCase):
    def setUp(self):
        self.cal = _fitted_calibrator()

    def test_blocked_pair_always_auto_rejects(self):
        d = self.cal.decide("VALVE", 0.99, "DISTINCT", blocked=True)
        self.assertEqual(d.tier, conformal.AUTO_REJECT)

    def test_undetermined_relation_always_goes_to_review(self):
        d = self.cal.decide("VALVE", 0.99, "UNDETERMINED")
        self.assertEqual(d.tier, conformal.REVIEW)

    def test_substitutable_never_auto_merges(self):
        d = self.cal.decide("VALVE", 0.99, "SUBSTITUTABLE", has_identity_key=True, has_graph_path=True)
        self.assertEqual(d.tier, conformal.REVIEW)

    def test_variant_of_never_auto_merges(self):
        d = self.cal.decide("VALVE", 0.99, "VARIANT_OF", has_identity_key=True, has_graph_path=True)
        self.assertEqual(d.tier, conformal.REVIEW)


class TestIdentityKeyGate(unittest.TestCase):
    def setUp(self):
        self.cal = _fitted_calibrator("VALVE")
        self.accept = self.cal.calibration_for("VALVE").accept_threshold

    def test_pressure_boundary_class_requires_identity_key_to_auto_accept(self):
        self.assertIn("VALVE", config.IDENTITY_KEY_REQUIRED)
        d = self.cal.decide("VALVE", self.accept + 0.001, "EQUIVALENT",
                             has_identity_key=False, has_graph_path=True)
        self.assertEqual(d.tier, conformal.REVIEW)

    def test_pressure_boundary_class_auto_accepts_with_identity_key(self):
        d = self.cal.decide("VALVE", self.accept + 0.001, "EQUIVALENT",
                             has_identity_key=True, has_graph_path=True)
        self.assertEqual(d.tier, conformal.AUTO_ACCEPT)

    def test_high_score_without_any_proof_still_goes_to_review(self):
        # No identity key AND no standards-graph path: text similarity alone
        # is never enough, whatever the class.
        d = self.cal.decide("GENERIC", 0.999, "EQUIVALENT",
                             has_identity_key=False, has_graph_path=False)
        self.assertEqual(d.tier, conformal.REVIEW)


class TestScoreThresholds(unittest.TestCase):
    def setUp(self):
        self.cal = _fitted_calibrator("VALVE")
        self.c = self.cal.calibration_for("VALVE")

    def test_low_score_auto_rejects(self):
        d = self.cal.decide("VALVE", self.c.reject_threshold - 0.001, "EQUIVALENT",
                             has_identity_key=True)
        self.assertEqual(d.tier, conformal.AUTO_REJECT)

    def test_middle_score_goes_to_review(self):
        mid = (self.c.accept_threshold + self.c.reject_threshold) / 2
        d = self.cal.decide("VALVE", mid, "EQUIVALENT", has_identity_key=True, has_graph_path=True)
        self.assertEqual(d.tier, conformal.REVIEW)

    def test_accept_threshold_is_never_below_the_safety_floor(self):
        self.assertGreaterEqual(self.c.accept_threshold, 0.55)

    def test_reject_threshold_is_never_above_the_safety_ceiling(self):
        self.assertLessEqual(self.c.reject_threshold, 0.5)
        self.assertLess(self.c.reject_threshold, self.c.accept_threshold)


class TestPerClassAlpha(unittest.TestCase):
    def test_pressure_boundary_classes_get_tighter_alpha_than_generic(self):
        self.assertLess(config.alpha_for("VALVE"), config.alpha_for("GENERIC"))

    def test_unknown_class_falls_back_to_default_alpha(self):
        self.assertEqual(config.alpha_for("NOT_A_REAL_CLASS"), config.DEFAULT_ALPHA)


class TestPersistence(unittest.TestCase):
    def test_round_trips_through_to_dict_from_dict(self):
        cal = _fitted_calibrator("VALVE")
        restored = conformal.Calibrator.from_dict(cal.to_dict())
        a = cal.calibration_for("VALVE")
        b = restored.calibration_for("VALVE")
        self.assertEqual(a.accept_threshold, b.accept_threshold)
        self.assertEqual(a.reject_threshold, b.reject_threshold)


if __name__ == "__main__":
    unittest.main()

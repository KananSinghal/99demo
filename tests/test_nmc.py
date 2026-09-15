"""Unit tests for samanvay.core.nmc -- the National Material Code and its
Verhoeff check digit (the same check-digit scheme Aadhaar uses). These are
property-based on purpose (round trip, tamper detection) rather than pinned to
one hand-computed digit, so they stay correct if the serial or NSC inputs
change.
"""
import unittest

from . import _boot  # noqa: F401
from samanvay.core import nmc


class TestVerhoeff(unittest.TestCase):
    def test_check_digit_is_a_single_digit(self):
        d = nmc.verhoeff_check_digit("5306720147723")
        self.assertIn(d, range(10))

    def test_validate_true_for_correct_check_digit(self):
        d = nmc.verhoeff_check_digit("5306720147723")
        self.assertTrue(nmc.verhoeff_validate(f"5306720147723{d}"))

    def test_validate_false_for_wrong_check_digit(self):
        d = nmc.verhoeff_check_digit("5306720147723")
        wrong = (d + 1) % 10
        self.assertFalse(nmc.verhoeff_validate(f"5306720147723{wrong}"))

    def test_transposition_is_detected(self):
        # Verhoeff's whole point over a simple mod-10 checksum: it catches
        # adjacent-digit transpositions, the commonest keyboard-entry error.
        base = "5306720147723"
        d = nmc.verhoeff_check_digit(base)
        transposed = base[:5] + base[6] + base[5] + base[7:]
        if transposed == base:
            self.skipTest("no adjacent-digit swap available in this fixture")
        self.assertFalse(nmc.verhoeff_validate(f"{transposed}{d}"))


class TestMintParseValidate(unittest.TestCase):
    def test_mint_produces_a_valid_code(self):
        m = nmc.mint(147723, "5306", "72")
        self.assertTrue(nmc.validate(m.code)["valid"])

    def test_serial_is_non_significant_but_deterministic(self):
        a = nmc.mint(147723, "5306", "72")
        b = nmc.mint(147723, "5306", "72")
        self.assertEqual(a.code, b.code)

    def test_different_serials_mint_different_codes(self):
        a = nmc.mint(147723, "5306", "72")
        b = nmc.mint(147724, "5306", "72")
        self.assertNotEqual(a.code, b.code)

    def test_tampered_check_digit_fails_validation(self):
        m = nmc.mint(147723, "5306", "72")
        last = int(m.code[-1])
        tampered = m.code[:-1] + str((last + 1) % 10)
        result = nmc.validate(tampered)
        self.assertFalse(result["valid"])
        self.assertIn("expected_check_digit", result)

    def test_parse_accepts_bare_nsn(self):
        m = nmc.mint(147723, "5306", "72")
        parsed = nmc.parse(m.nsn)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.code, m.code)

    def test_parse_accepts_formatted_code_with_spaces(self):
        m = nmc.mint(147723, "5306", "72")
        spaced = m.code.replace("-", " - ")
        parsed = nmc.parse(spaced)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.nsn, m.nsn)

    def test_parse_rejects_garbage(self):
        self.assertIsNone(nmc.parse("NOT-A-CODE"))

    def test_mint_for_class_uses_class_nsc(self):
        m = nmc.mint_for_class(1, "FASTENER_BOLT")
        self.assertTrue(nmc.validate(m.code)["valid"])


if __name__ == "__main__":
    unittest.main()

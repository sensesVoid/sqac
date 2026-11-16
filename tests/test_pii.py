"""Tests for PII detection."""

import unittest

from sqac.pii import detect_pii, _luhn_valid


class TestPii(unittest.TestCase):
    def test_email(self):
        findings = detect_pii("contact me at user@example.com please")
        self.assertTrue(any(f["type"] == "email" for f in findings))

    def test_phone(self):
        findings = detect_pii("call me at +1-555-123-4567")
        self.assertTrue(any(f["type"] == "phone" for f in findings))

    def test_ssn(self):
        findings = detect_pii("SSN: 123-45-6789")
        self.assertTrue(any(f["type"] == "ssn" for f in findings))

    def test_credit_card_luhn(self):
        # Valid Luhn number
        findings = detect_pii("card: 4111 1111 1111 1111")
        self.assertTrue(any(f["type"] == "credit_card" for f in findings))

    def test_credit_card_invalid_luhn(self):
        findings = detect_pii("card: 4111 1111 1111 1112")
        self.assertFalse(any(f["type"] == "credit_card" for f in findings))

    def test_ip(self):
        findings = detect_pii("server at 192.168.1.1")
        self.assertTrue(any(f["type"] == "ip" for f in findings))

    def test_iban(self):
        findings = detect_pii("IBAN: GB82WEST12345698765432")
        self.assertTrue(any(f["type"] == "iban" for f in findings))

    def test_no_pii(self):
        findings = detect_pii("hello world, no secrets here")
        self.assertEqual(findings, [])

    def test_luhn_valid(self):
        self.assertTrue(_luhn_valid("4111111111111111"))
        self.assertFalse(_luhn_valid("4111111111111112"))

    def test_multiple_findings(self):
        findings = detect_pii("email user@test.com and ip 10.0.0.1")
        types = {f["type"] for f in findings}
        self.assertIn("email", types)
        self.assertIn("ip", types)


if __name__ == "__main__":
    unittest.main()

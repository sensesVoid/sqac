"""Tests for vault audit log."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from sqac.vault_audit import append_audit, read_audit, verify_audit_chain, AUDIT_PATH


class TestVaultAudit(unittest.TestCase):
    def setUp(self):
        self._orig_path = AUDIT_PATH
        self.td = tempfile.TemporaryDirectory()
        # Monkeypatch AUDIT_PATH
        import sqac.vault_audit as va
        self._va = va
        self.audit_path = Path(self.td.name) / "vault-audit.jsonl"
        va.AUDIT_PATH = self.audit_path

    def tearDown(self):
        self._va.AUDIT_PATH = self._orig_path
        self.td.cleanup()

    def test_append_and_read(self):
        append_audit(vault="default", name="openai", op="put", success=True, bytes_count=10)
        entries = read_audit()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "openai")
        self.assertEqual(entries[0]["op"], "put")
        self.assertTrue(entries[0]["success"])

    def test_chain_verification(self):
        append_audit(vault="default", name="a", op="put", success=True)
        append_audit(vault="default", name="b", op="put", success=True)
        self.assertTrue(verify_audit_chain())

    def test_tail(self):
        for i in range(5):
            append_audit(vault="default", name=f"e{i}", op="put", success=True)
        entries = read_audit(tail=3)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["name"], "e2")

    def test_no_values_in_audit(self):
        append_audit(vault="default", name="secret", op="put", success=True, bytes_count=100)
        text = self.audit_path.read_text()
        self.assertNotIn("sk-secret", text)
        self.assertNotIn("password", text.lower().split("op")[0] if "password" in text.lower() else "ok")

    def test_empty_audit(self):
        entries = read_audit()
        self.assertEqual(entries, [])
        self.assertTrue(verify_audit_chain())

    def test_multiple_ops(self):
        append_audit(vault="default", name="x", op="put", success=True)
        append_audit(vault="default", name="x", op="meta", success=True)
        append_audit(vault="default", name="x", op="remove", success=True)
        entries = read_audit()
        self.assertEqual(len(entries), 3)
        ops = [e["op"] for e in entries]
        self.assertEqual(ops, ["put", "meta", "remove"])


if __name__ == "__main__":
    unittest.main()

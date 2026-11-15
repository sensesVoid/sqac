"""Tests for vault MCP tools (reference-only, no plaintext exfiltration)."""

import os
import tempfile
import unittest
from unittest.mock import patch

from sqac.vault import Vault


class TestVaultMcpTools(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.vault_path = os.path.join(self.td.name, "default.sqac")
        os.environ["SQAC_VAULT_PASSPHRASE"] = "testpass"
        # Create a vault with a secret
        v = Vault.create(self.vault_path, mode="portable", passphrase="testpass")
        v.put("openai", "sk-secret-value-123", label="OpenAI key", policy={"allowed_hosts": ["api.openai.com"], "auth": "bearer"})
        # Point DEFAULT_VAULT_PATH to our temp vault
        import sqac.vault as vm
        self._orig_path = vm.DEFAULT_VAULT_PATH
        vm.DEFAULT_VAULT_PATH = __import__("pathlib").Path(self.vault_path)

    def tearDown(self):
        import sqac.vault as vm
        vm.DEFAULT_VAULT_PATH = self._orig_path
        self.td.cleanup()
        os.environ.pop("SQAC_VAULT_PASSPHRASE", None)

    def test_vault_list_returns_metadata_only(self):
        from sqac.vault_mcp import vault_list
        result = vault_list()
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["entries"]), 1)
        entry = result["entries"][0]
        self.assertEqual(entry["name"], "openai")
        # No value/ct/nonce in metadata
        self.assertNotIn("value", entry)
        self.assertNotIn("ct", entry)
        self.assertNotIn("nonce", entry)

    def test_vault_meta_returns_no_value(self):
        from sqac.vault_mcp import vault_meta
        result = vault_meta("openai")
        self.assertTrue(result["ok"])
        meta = result["meta"]
        self.assertEqual(meta["label"], "OpenAI key")
        self.assertNotIn("value", meta)
        self.assertNotIn("ct", meta)
        self.assertNotIn("nonce", meta)

    def test_vault_meta_not_found(self):
        from sqac.vault_mcp import vault_meta
        result = vault_meta("nonexistent")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "not_found")

    def test_vault_audit_returns_entries(self):
        from sqac.vault_mcp import vault_audit
        result = vault_audit(limit=10)
        self.assertTrue(result["ok"])
        # Should have at least the put audit entry
        self.assertGreaterEqual(len(result["entries"]), 1)

    def test_no_plaintext_in_list(self):
        from sqac.vault_mcp import vault_list
        result = vault_list()
        text = str(result)
        self.assertNotIn("sk-secret-value-123", text)

    def test_no_plaintext_in_meta(self):
        from sqac.vault_mcp import vault_meta
        result = vault_meta("openai")
        text = str(result)
        self.assertNotIn("sk-secret-value-123", text)

    def test_no_plaintext_in_audit(self):
        from sqac.vault_mcp import vault_audit
        result = vault_audit(limit=10)
        text = str(result)
        self.assertNotIn("sk-secret-value-123", text)

    def test_vault_not_found_returns_structured(self):
        import sqac.vault as vm
        orig = vm.DEFAULT_VAULT_PATH
        vm.DEFAULT_VAULT_PATH = __import__("pathlib").Path("/nonexistent/path.sqac")
        try:
            from sqac.vault_mcp import vault_list
            result = vault_list()
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "vault_not_found")
        finally:
            vm.DEFAULT_VAULT_PATH = orig

    def test_locked_vault_returns_structured_error(self):
        # Open vault without passphrase in an env without SQAC_VAULT_PASSPHRASE
        import sqac.vault as vm
        os.environ.pop("SQAC_VAULT_PASSPHRASE", None)
        # Force keychain unavailable
        with patch("sqac.vault_crypto.keychain_get", side_effect=Exception("no keyring")):
            from sqac.vault_mcp import vault_meta
            result = vault_meta("openai")
            # Should return locked or not_found, not crash
            self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()

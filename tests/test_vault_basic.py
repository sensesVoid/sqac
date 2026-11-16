"""Basic vault smoke test."""

import os
import tempfile
import unittest
from unittest.mock import patch

from sqac.vault import Vault, create_vault, open_vault


class TestVaultBasic(unittest.TestCase):
    def test_create_and_open_keychain_mode(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "default.sqac")
            master = os.urandom(32)
            with patch("sqac.vault_crypto.keychain_set"), patch("sqac.vault_crypto.keychain_get", return_value=master.hex()):
                v = Vault.create(path, mode="keychain", master_key=master)
                v.put("openai", "sk-test-123", label="OpenAI", policy={"allowed_hosts": ["api.openai.com"], "auth": "bearer"})
                self.assertEqual(len(v.list()), 1)
                meta = v.get_meta("openai")
                self.assertEqual(meta["label"], "OpenAI")
                self.assertNotIn("value", meta)
                # resolve internally
                self.assertEqual(v.resolve("openai"), "sk-test-123")
                # remove
                self.assertTrue(v.remove("openai"))
                self.assertEqual(v.list(), [])

    def test_portable_mode_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "default.sqac")
            v = Vault.create(path, mode="portable", passphrase="testpass")
            v.put("secret1", "my-secret-value", entry_type="secret")
            self.assertEqual(v.resolve("secret1"), "my-secret-value")
            # Reopen with passphrase
            v2 = Vault.open(path, passphrase="testpass")
            self.assertEqual(v2.resolve("secret1"), "my-secret-value")
            # Wrong passphrase fails
            with self.assertRaises(Exception):
                Vault.open(path, passphrase="wrongpass")

    def test_wrong_passphrase_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "default.sqac")
            Vault.create(path, mode="portable", passphrase="correct")
            with self.assertRaises(Exception):
                Vault.open(path, passphrase="incorrect")

    def test_value_cap(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "default.sqac")
            v = Vault.create(path, mode="portable", passphrase="test")
            with self.assertRaises(ValueError):
                v.put("big", "x" * 70000)

    def test_parse_ref(self):
        from sqac.vault import parse_vault_ref, canonical_ref
        self.assertEqual(parse_vault_ref("vault:openai"), "openai")
        self.assertEqual(parse_vault_ref("${vault:openai}"), "openai")
        self.assertEqual(parse_vault_ref("vault://openai"), "openai")
        self.assertIsNone(parse_vault_ref("not-a-ref"))
        self.assertEqual(canonical_ref("openai"), "vault:openai")


if __name__ == "__main__":
    unittest.main()

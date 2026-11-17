"""Tests for vault HTTP proxy."""

import os
import tempfile
import unittest

from sqac.vault import Vault


class TestVaultHttp(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.vault_path = os.path.join(self.td.name, "default.sqac")
        os.environ["SQAC_VAULT_PASSPHRASE"] = "testpass"
        v = Vault.create(self.vault_path, mode="portable", passphrase="testpass")
        v.put("api_key", "secret-token-123", label="API key", policy={"allowed_hosts": ["127.0.0.1"], "auth": "bearer"})

        import sqac.vault as vm
        self._orig_path = vm.DEFAULT_VAULT_PATH
        vm.DEFAULT_VAULT_PATH = __import__("pathlib").Path(self.vault_path)

    def tearDown(self):
        import sqac.vault as vm
        vm.DEFAULT_VAULT_PATH = self._orig_path
        self.td.cleanup()
        os.environ.pop("SQAC_VAULT_PASSPHRASE", None)

    def test_http_rejects_non_https(self):
        from sqac.vault import open_vault
        v = open_vault()
        with self.assertRaises(ValueError):
            v.http("api_key", "GET", "http://127.0.0.1:8080/test")

    def test_http_rejects_disallowed_host(self):
        from sqac.vault import open_vault
        v = open_vault()
        with self.assertRaises(ValueError):
            v.http("api_key", "GET", "https://evil.com/test")

    def test_http_requires_allowed_hosts(self):
        # Create entry without allowed_hosts
        from sqac.vault import open_vault
        v = open_vault()
        v.put("nohost", "secret", policy={"allowed_hosts": [], "auth": "bearer"})
        with self.assertRaises(ValueError):
            v.http("nohost", "GET", "https://example.com/test")

    def test_vault_http_mcp_no_plaintext(self):
        from sqac.vault_mcp import vault_http
        # Will fail policy but should not leak secret in error
        result = vault_http("api_key", "GET", "https://evil.com/test")
        text = str(result)
        self.assertNotIn("secret-token-123", text)


if __name__ == "__main__":
    unittest.main()

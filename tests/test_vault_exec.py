"""Tests for vault exec (env injection + redaction)."""

import os
import tempfile
import unittest

from sqac.vault import Vault


class TestVaultExec(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.vault_path = os.path.join(self.td.name, "default.sqac")
        os.environ["SQAC_VAULT_PASSPHRASE"] = "testpass"
        v = Vault.create(self.vault_path, mode="portable", passphrase="testpass")
        v.put("mykey", "secret-value-42", label="test key")

        import sqac.vault as vm
        self._orig_path = vm.DEFAULT_VAULT_PATH
        vm.DEFAULT_VAULT_PATH = __import__("pathlib").Path(self.vault_path)

    def tearDown(self):
        import sqac.vault as vm
        vm.DEFAULT_VAULT_PATH = self._orig_path
        self.td.cleanup()
        os.environ.pop("SQAC_VAULT_PASSPHRASE", None)

    def test_exec_injects_env(self):
        from sqac.vault import open_vault
        v = open_vault()
        result = v.exec_env({"MYVAR": "vault:mykey"}, ["python", "-c", "import os; print(os.environ.get('MYVAR', 'MISSING'))"], redact=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("secret-value-42", result.stdout)

    def test_exec_redacts_output(self):
        from sqac.vault import open_vault
        v = open_vault()
        result = v.exec_env({"MYVAR": "vault:mykey"}, ["python", "-c", "import os; print('val=' + os.environ.get('MYVAR', 'MISSING'))"], redact=True)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("secret-value-42", result.stdout)
        self.assertIn("[REDACTED:vault:MYVAR]", result.stdout)

    def test_exec_strips_sqac_vault_env(self):
        from sqac.vault import open_vault
        v = open_vault()
        os.environ["SQAC_VAULT_PASSPHRASE"] = "testpass"
        result = v.exec_env({"MYVAR": "vault:mykey"}, ["python", "-c", "import os; print('SQAC_VAULT_PASSPHRASE' in os.environ)"], redact=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("False", result.stdout)
        os.environ.pop("SQAC_VAULT_PASSPHRASE", None)

    def test_exec_invalid_ref_raises(self):
        from sqac.vault import open_vault
        v = open_vault()
        with self.assertRaises(ValueError):
            v.exec_env({"MYVAR": "not-a-ref"}, ["echo", "hi"])

    def test_exec_no_redact_flag(self):
        from sqac.vault import open_vault
        v = open_vault()
        result = v.exec_env({"MYVAR": "vault:mykey"}, ["python", "-c", "import os; print(os.environ.get('MYVAR'))"], redact=False)
        self.assertIn("secret-value-42", result.stdout)


if __name__ == "__main__":
    unittest.main()

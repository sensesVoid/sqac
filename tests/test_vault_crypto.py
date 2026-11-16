"""Tests for vault crypto primitives."""

import os
import unittest

from sqac.vault_crypto import (
    generate_dek,
    generate_nonce,
    aad,
    encrypt_value,
    decrypt_value,
    VaultCryptoError,
    wrap_dek,
    unwrap_dek,
    derive_kek_argon2id,
)


class TestVaultCrypto(unittest.TestCase):
    def test_roundtrip_and_aad_binding(self):
        dek = generate_dek()
        n, ct = encrypt_value(dek, b"secret", vault="default", name="openai", entry_type="secret")
        self.assertEqual(decrypt_value(dek, n, ct, vault="default", name="openai", entry_type="secret"), b"secret")
        with self.assertRaises(VaultCryptoError):
            decrypt_value(dek, n, ct, vault="default", name="other", entry_type="secret")

    def test_wrap_unwrap_and_argon2id(self):
        dek = generate_dek()
        salt = os.urandom(16)
        kek = derive_kek_argon2id("pass", salt)
        wn, wct = wrap_dek(dek, kek)
        self.assertEqual(unwrap_dek(wn, wct, kek), dek)

    def test_wrong_passphrase_fails(self):
        dek = generate_dek()
        salt = os.urandom(16)
        wn, wct = wrap_dek(dek, derive_kek_argon2id("a", salt))
        with self.assertRaises(VaultCryptoError):
            unwrap_dek(wn, wct, derive_kek_argon2id("b", salt))

    def test_tampered_ciphertext_fails(self):
        dek = generate_dek()
        n, ct = encrypt_value(dek, b"secret", vault="default", name="openai", entry_type="secret")
        import base64
        raw = base64.b64decode(ct.encode())
        tampered = bytearray(raw)
        tampered[0] ^= 1
        with self.assertRaises(VaultCryptoError):
            decrypt_value(dek, n, base64.b64encode(bytes(tampered)).decode(), vault="default", name="openai", entry_type="secret")

    def test_aad_format(self):
        self.assertEqual(aad("default", "openai", "secret"), b"sqac-vault-v1|default|openai|secret")

    def test_nonce_length(self):
        self.assertEqual(len(generate_nonce()), 12)
        self.assertEqual(len(generate_dek()), 32)


if __name__ == "__main__":
    unittest.main()

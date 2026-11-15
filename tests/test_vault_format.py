"""Tests for v5 vault format support."""

import json
import tempfile
import unittest
from pathlib import Path

from sqac.format import (
    CartridgeHeader,
    Entry,
    write_cartridge,
    read_cartridge,
    VERSION,
    WRITE_VERSION,
    VAULT_WRITE_VERSION,
    FLAG_ENCRYPTED,
)


class TestVaultFormat(unittest.TestCase):
    def test_version_constants(self):
        self.assertEqual(WRITE_VERSION, 4)
        self.assertEqual(VAULT_WRITE_VERSION, 5)
        self.assertGreaterEqual(VERSION, 5)
        self.assertEqual(FLAG_ENCRYPTED, 0x0008)

    def test_cartridge_type_property(self):
        h = CartridgeHeader(version=5, extra={"cartridge_type": "vault"})
        self.assertEqual(h.cartridge_type, "vault")
        h2 = CartridgeHeader(version=4)
        self.assertEqual(h2.cartridge_type, "memory")

    def test_vault_header_roundtrip_v5(self):
        hdr = CartridgeHeader(version=5, dims=1024, extra={
            "cartridge_type": "vault",
            "vault": {"schema": 1, "kdf": "keychain", "wrap_nonce": "AA==", "wrapped_dek": "AQ==", "keychain_service": "sqac-vault", "created": "2026-01-01T00:00:00Z"},
        })
        payload = {"name": "k", "entry_type": "secret", "nonce": "AA==", "ct": "AQ=="}
        e = Entry(key_bits=b"k" + b"\x00" * 127, payload=payload, kind=0, encrypted=True)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "default.sqac"
            write_cartridge(str(p), hdr, {}, [e])
            cart = read_cartridge(str(p))
            self.assertEqual(cart.header.version, 5)
            self.assertEqual(cart.header.extra["cartridge_type"], "vault")
            self.assertTrue(cart.entries[0].encrypted)
            self.assertEqual(cart.entries[0].payload["ct"], "AQ==")

    def test_memory_still_writes_v4(self):
        hdr = CartridgeHeader(version=WRITE_VERSION, dims=1024)
        e = Entry(key_bits=b"k" + b"\x00" * 127, payload={"content": "hello"}, kind=0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mem.sqac"
            write_cartridge(str(p), hdr, {}, [e])
            cart = read_cartridge(str(p))
        self.assertEqual(cart.header.version, 4)


if __name__ == "__main__":
    unittest.main()

"""Tests for vault/agent cartridge isolation from rack and search."""

import os
import tempfile
import unittest

from sqac.format import CartridgeHeader, Entry, write_cartridge
from sqac.rack import CartridgeRack
from sqac.store import SqacStore


class TestVaultIsolation(unittest.TestCase):
    def test_auto_load_skips_vault_type(self):
        with tempfile.TemporaryDirectory() as td:
            # Create a vault cartridge (type=vault)
            hdr = CartridgeHeader(version=5, dims=1024, extra={"cartridge_type": "vault"})
            e = Entry(key_bits=b"x" + b"\x00" * 127, payload={"name": "secret", "ct": "abc"}, encrypted=True)
            write_cartridge(os.path.join(td, "vault.sqac"), hdr, {}, [e])
            # Create a normal memory cartridge named "vault" (type=memory)
            st = SqacStore()
            st.add("normal memory named vault", key="vault", kind="fact")
            st.save(os.path.join(td, "vault_mem.sqac"))

            rack = CartridgeRack(directory=td, auto_load=True)
            # vault.sqac (type=vault) should NOT be loaded
            self.assertNotIn("vault", rack.names())
            # vault_mem.sqac should be loaded
            self.assertIn("vault_mem", rack.names())

    def test_search_excludes_vault_entries(self):
        with tempfile.TemporaryDirectory() as td:
            # Create a vault cartridge
            hdr = CartridgeHeader(version=5, dims=1024, extra={"cartridge_type": "vault"})
            e = Entry(key_bits=b"x" + b"\x00" * 127, payload={"name": "secret", "ct": "abc", "content": "secret value"}, encrypted=True)
            write_cartridge(os.path.join(td, "myvault.sqac"), hdr, {}, [e])
            # Create normal memory
            st = SqacStore()
            st.add("normal fact", key="normal", kind="fact")
            st.save(os.path.join(td, "memory.sqac"))

            rack = CartridgeRack(directory=td, auto_load=True)
            # Even if we manually register the vault, search should skip it
            if "myvault" not in rack.names():
                rack.register("myvault")  # may or may not work
            hits = rack.search("secret")
            for h in hits:
                self.assertNotEqual(h.meta.get("cartridge"), "myvault")

    def test_normal_cartridge_named_vault_is_loaded(self):
        with tempfile.TemporaryDirectory() as td:
            st = SqacStore()
            st.add("a normal cartridge named vault", key="v", kind="fact")
            st.save(os.path.join(td, "vault.sqac"))
            rack = CartridgeRack(directory=td, auto_load=True)
            self.assertIn("vault", rack.names())


if __name__ == "__main__":
    unittest.main()

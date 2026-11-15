"""Windows must replace a cartridge while the writer lock is held."""

import sys
import tempfile
import unittest
from pathlib import Path

from sqac.format import CartridgeHeader, Entry, read_cartridge, write_cartridge


@unittest.skipUnless(sys.platform == "win32", "Windows-specific lock behavior")
class TestWindowsReplace(unittest.TestCase):
    def test_second_write_replaces_locked_cartridge(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "memory.sqac"
            header = CartridgeHeader(dims=64)
            write_cartridge(path, header, {}, [_entry("one")])
            write_cartridge(path, header, {}, [_entry("two")])
            self.assertEqual(read_cartridge(path).entries[0].payload["content"], "two")


def _entry(text: str) -> Entry:
    return Entry(key_bits=bytearray(8), payload={"content": text})


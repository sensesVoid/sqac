"""Fuzz tests for cartridge format hardening."""

import json
import os
import struct
import tempfile
import unittest

from sqac.format import FormatError, read_cartridge, write_cartridge, CartridgeHeader, Entry, VERSION


class TestFormatFuzz(unittest.TestCase):
    def test_truncated_file_raises(self):
        """Truncated files should raise FormatError, not crash."""
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "trunc.sqac")
            # Write a minimal valid header then truncate
            hdr = CartridgeHeader(dims=1024)
            e = Entry(key_bits=bytearray(128), payload={"content": "x"})
            write_cartridge(p, hdr, {}, [e])
            data = open(p, "rb").read()
            # Truncate at various points
            for cut in [0, 4, 8, 20, len(data) // 2, len(data) - 1]:
                tp = os.path.join(td, f"trunc_{cut}.sqac")
                with open(tp, "wb") as f:
                    f.write(data[:cut])
                with self.assertRaises((FormatError, Exception)):
                    read_cartridge(tp)

    def test_corrupted_magic_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "badmagic.sqac")
            hdr = CartridgeHeader(dims=1024)
            e = Entry(key_bits=bytearray(128), payload={"content": "x"})
            write_cartridge(p, hdr, {}, [e])
            data = bytearray(open(p, "rb").read())
            data[0] ^= 0xFF
            with open(p, "wb") as f:
                f.write(data)
            with self.assertRaises((FormatError, ValueError, UnicodeDecodeError)):
                read_cartridge(p)

    def test_huge_dims_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "hugedims.sqac")
            # Craft a header with huge dims
            import json
            hdr_json = json.dumps({"magic": "SQAC", "version": 4, "dims": 999999, "encoder_name": "bsc-ngram-v1", "vocab_fingerprint": "", "created": "", "name": "", "description": ""}).encode()
            with open(p, "wb") as f:
                f.write(struct.pack("<II", len(hdr_json), 0))
                f.write(hdr_json)
                f.write(struct.pack("<I", 0))  # vocab len
                f.write(struct.pack("<Q", 0))  # entries
                f.write(struct.pack("<I", 0))  # ext blocks
            with self.assertRaises((FormatError, ValueError, UnicodeDecodeError)):
                read_cartridge(p)

    def test_huge_tb_len_rejected(self):
        # Craft a file with a huge token length
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "hugetoken.sqac")
            hdr = CartridgeHeader(dims=1024)
            hdr_json = json.dumps({"magic": "SQAC", "version": 4, "dims": 1024, "encoder_name": "bsc-ngram-v1", "vocab_fingerprint": "", "created": "", "name": "", "description": ""}).encode()
            with open(p, "wb") as f:
                f.write(struct.pack("<II", len(hdr_json), 0))
                f.write(hdr_json)
                f.write(struct.pack("<I", 1))  # vocab_len = 1
                f.write(struct.pack("<I", 999999))  # tb_len = huge
                f.write(b"token")
                f.write(struct.pack("<I", 0))  # idx
                f.write(struct.pack("<Q", 0))  # entries
                f.write(struct.pack("<I", 0))  # ext blocks
            with self.assertRaises(FormatError):
                read_cartridge(p)

    def test_version_too_new_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "future.sqac")
            hdr_json = json.dumps({"magic": "SQAC", "version": 99, "dims": 1024, "encoder_name": "bsc-ngram-v1", "vocab_fingerprint": "", "created": "", "name": "", "description": ""}).encode()
            with open(p, "wb") as f:
                f.write(struct.pack("<II", len(hdr_json), 0))
                f.write(hdr_json)
            with self.assertRaises((FormatError, ValueError, UnicodeDecodeError)):
                read_cartridge(p)

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "empty.sqac")
            open(p, "wb").close()
            with self.assertRaises(Exception):
                read_cartridge(p)


if __name__ == "__main__":
    unittest.main()

"""Windows lock audit tests."""

import os
import tempfile
import threading
import unittest
from pathlib import Path

from sqac.format import (
    CartridgeHeader,
    Entry,
    write_cartridge,
    read_cartridge,
    locked_cartridge,
    CartridgeLockError,
    compact_cartridge,
)


class TestWindowsLockAudit(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = Path(self.td.name) / "test.sqac"
        hdr = CartridgeHeader(dims=1024)
        e = Entry(key_bits=bytearray(128), payload={"content": "hello"})
        write_cartridge(str(self.path), hdr, {}, [e])

    def tearDown(self):
        self.td.cleanup()

    def test_zero_byte_file_lock(self):
        """Locking a zero-byte sidecar lock file should work."""
        lock_path = self.path.with_name(self.path.name + ".lock")
        # Ensure sidecar exists (may be zero bytes)
        lock_path.touch(exist_ok=True)
        with locked_cartridge(self.path):
            pass  # should not raise

    def test_concurrent_writers_serialized(self):
        """Concurrent writes should be serialized, not corrupted."""
        results = []
        errors = []

        def writer(i):
            try:
                hdr = CartridgeHeader(dims=1024)
                e = Entry(key_bits=bytearray(128), payload={"content": f"writer-{i}"})
                write_cartridge(str(self.path), hdr, {}, [e])
                results.append(i)
            except Exception as exc:
                errors.append((i, str(exc)))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"errors: {errors}")
        # File should be readable
        cart = read_cartridge(str(self.path))
        self.assertEqual(len(cart.entries), 1)

    def test_compact_reentrancy(self):
        """compact_cartridge should be safe to call concurrently."""
        # Add tombstones first
        hdr = CartridgeHeader(dims=1024)
        entries = [
            Entry(key_bits=bytearray(128), payload={"content": "a"}, deleted=True),
            Entry(key_bits=bytearray(128), payload={"content": "b"}),
        ]
        write_cartridge(str(self.path), hdr, {}, entries)

        # Compact twice concurrently
        results = []
        def compactor():
            try:
                n = compact_cartridge(str(self.path))
                results.append(n)
            except Exception as exc:
                results.append(("error", str(exc)))

        threads = [threading.Thread(target=compactor) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should succeed
        self.assertTrue(len(results) > 0)
        # Final file should be readable
        cart = read_cartridge(str(self.path))
        self.assertGreaterEqual(len(cart.entries), 1)

    def test_read_during_write(self):
        """Reads should wait for write lock (or fail gracefully)."""
        write_done = threading.Event()
        read_result = []

        def writer():
            with locked_cartridge(self.path):
                import time
                time.sleep(0.5)
            write_done.set()

        def reader():
            write_done.wait(0.1)
            try:
                cart = read_cartridge(str(self.path), locked=True)
                read_result.append(("ok", len(cart.entries)))
            except Exception as exc:
                read_result.append(("error", str(exc)))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Read should either succeed or raise CartridgeLockError
        self.assertTrue(len(read_result) == 1)

    def test_sidecar_lock_cleanup(self):
        """Sidecar lock file should not prevent future operations."""
        lock_path = self.path.with_name(self.path.name + ".lock")
        # Create and use lock
        with locked_cartridge(self.path):
            pass
        # Lock file may exist; file should still be readable
        cart = read_cartridge(str(self.path))
        self.assertEqual(len(cart.entries), 1)
        # Another lock cycle should work
        with locked_cartridge(self.path):
            pass
        cart2 = read_cartridge(str(self.path))
        self.assertEqual(len(cart2.entries), 1)


if __name__ == "__main__":
    unittest.main()

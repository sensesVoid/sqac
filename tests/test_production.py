"""Tests for production features: locking, compact, cache, sanitization.

Run: python -m pytest tests/test_production.py -q
"""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from sqac.format import (
    CartridgeLockError,
    compact_cartridge,
    locked_cartridge,
    read_cartridge,
    write_cartridge,
    CartridgeHeader,
    Entry,
    FLAG_DELETED,
)
from sqac.store import (
    SqacStore,
    sanitize_content,
    sanitize_key,
    resolve_kind,
)


class TestSanitization(unittest.TestCase):
    def test_strip_control_chars(self):
        dirty = "hello\x00\x01\x02world\x7f"
        clean = sanitize_content(dirty)
        self.assertEqual(clean, "helloworld")

    def test_strip_newlines_preserved(self):
        text = "line1\nline2\nline3"
        clean = sanitize_content(text)
        self.assertEqual(clean, text)

    def test_strip_tabs_preserved(self):
        text = "col1\tcol2"
        clean = sanitize_content(text)
        self.assertEqual(clean, text)

    def test_truncation(self):
        long = "x" * 60_000
        clean = sanitize_content(long)
        self.assertLess(len(clean), 60_000)
        self.assertIn("truncated", clean)

    def test_key_sanitization(self):
        key = sanitize_key("  hello\x00world  ")
        self.assertEqual(key, "helloworld")

    def test_key_length_limit(self):
        long_key = "x" * 2000
        key = sanitize_key(long_key)
        self.assertEqual(len(key), 1024)


class TestFileLocking(unittest.TestCase):
    def test_write_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.sqac"
            header = CartridgeHeader(dims=256, name="test")
            entries = [Entry(key_bits=bytearray(b"\xaa" * 32), payload={"content": "hello"})]
            write_cartridge(path, header, {}, entries)
            cart = read_cartridge(path, locked=True)
            self.assertEqual(cart.entries[0].payload["content"], "hello")

    def test_locked_write_blocks_concurrent_write(self):
        """Two threads writing to the same file: second should wait."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.sqac"
            header = CartridgeHeader(dims=256, name="test")
            entries = [Entry(key_bits=bytearray(b"\xaa" * 32), payload={"content": "v1"})]
            write_cartridge(path, header, {}, entries, locked=True)

            results = []

            def writer():
                with locked_cartridge(path):
                    time.sleep(0.1)  # hold the lock
                    # read to verify we have it
                    cart = read_cartridge(path)
                    results.append(cart.entries[0].payload["content"])

            t1 = threading.Thread(target=writer)
            t2 = threading.Thread(target=writer)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)
            # Both should complete (not deadlock)
            self.assertEqual(len(results), 2)

    def test_read_lock_allows_concurrent_read(self):
        """Multiple readers can hold the lock simultaneously."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.sqac"
            header = CartridgeHeader(dims=256, name="test")
            entries = [Entry(key_bits=bytearray(b"\xaa" * 32), payload={"content": "hello"})]
            write_cartridge(path, header, {}, entries, locked=True)

            results = []

            def reader():
                cart = read_cartridge(path, locked=True)
                results.append(cart.entries[0].payload["content"])

            threads = [threading.Thread(target=reader) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            self.assertEqual(len(results), 5)
            self.assertTrue(all(r == "hello" for r in results))


class TestCompact(unittest.TestCase):
    def test_compact_removes_tombstones(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.sqac"
            header = CartridgeHeader(dims=256, name="test")
            entries = [
                Entry(key_bits=bytearray(b"\xaa" * 32), payload={"content": "keep me"}),
                Entry(key_bits=bytearray(b"\xbb" * 32), payload={"content": "delete me"}, deleted=True),
                Entry(key_bits=bytearray(b"\xcc" * 32), payload={"content": "keep me too"}),
            ]
            write_cartridge(path, header, {}, entries)

            report = compact_cartridge(path)
            self.assertTrue(report["compacted"])
            self.assertEqual(report["original"], 3)
            self.assertEqual(report["alive"], 2)
            self.assertEqual(report["removed"], 1)

            # Verify the compacted file is valid
            cart = read_cartridge(path)
            self.assertEqual(len(cart.entries), 2)
            self.assertFalse(any(e.deleted for e in cart.entries))

    def test_compact_noop_when_clean(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.sqac"
            header = CartridgeHeader(dims=256, name="test")
            entries = [Entry(key_bits=bytearray(b"\xaa" * 32), payload={"content": "alive"})]
            write_cartridge(path, header, {}, entries)

            report = compact_cartridge(path)
            self.assertFalse(report["compacted"])
            self.assertEqual(report["removed"], 0)

    def test_compact_to_separate_file(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.sqac"
            dst = Path(td) / "dst.sqac"
            header = CartridgeHeader(dims=256, name="test")
            entries = [
                Entry(key_bits=bytearray(b"\xaa" * 32), payload={"content": "keep"}),
                Entry(key_bits=bytearray(b"\xbb" * 32), payload={"content": "gone"}, deleted=True),
            ]
            write_cartridge(src, header, {}, entries)

            report = compact_cartridge(src, out=dst)
            self.assertTrue(report["compacted"])
            self.assertTrue(dst.exists())
            # Original unchanged
            cart_src = read_cartridge(src)
            self.assertEqual(len(cart_src.entries), 2)
            # Destination compacted
            cart_dst = read_cartridge(dst)
            self.assertEqual(len(cart_dst.entries), 1)


class TestQueryCache(unittest.TestCase):
    def test_cache_returns_same_result(self):
        store = SqacStore()
        store.add("Use pytest for testing", key="testing framework")
        store.add("Deploy to ARM64 only", key="deployment target")

        h1 = store.search("testing framework", use_cache=True)
        h2 = store.search("testing framework", use_cache=True)
        self.assertEqual(h1[0].content, h2[0].content)
        self.assertEqual(h1[0].confidence, h2[0].confidence)

    def test_cache_invalidated_on_write(self):
        store = SqacStore()
        store.add("Use pytest", key="testing")
        h1 = store.search("testing", use_cache=True)
        self.assertTrue(h1)

        # Add new entry — cache should be invalidated
        store.add("Deploy to ARM64", key="deployment")
        h2 = store.search("testing", use_cache=True)
        # Should still work (cache rebuilt)
        self.assertTrue(h2)

    def test_cache_disabled_bypasses(self):
        store = SqacStore()
        store.add("Use pytest", key="testing")
        h1 = store.search("testing", use_cache=False)
        h2 = store.search("testing", use_cache=False)
        self.assertEqual(h1[0].content, h2[0].content)


class TestStoreCompact(unittest.TestCase):
    def test_in_place_compact(self):
        store = SqacStore()
        idx1 = store.add("fact one", key="f1")
        idx2 = store.add("fact two", key="f2")
        idx3 = store.add("fact three", key="f3")
        store.delete(idx2)
        self.assertEqual(len(store), 2)

        report = store.compact()
        self.assertEqual(report["before"], 3)
        self.assertEqual(report["after"], 2)
        self.assertEqual(report["removed"], 1)
        self.assertEqual(len(store), 2)

    def test_compact_preserves_search(self):
        store = SqacStore()
        store.add("Use pytest for testing", key="testing framework")
        store.add("Deploy to ARM64 only", key="deployment target")
        idx3 = store.add("This will be deleted", key="temp fact")
        store.delete(idx3)
        store.compact()

        hits = store.search("testing framework")
        self.assertTrue(hits)
        self.assertIn("pytest", hits[0].content)


class TestStoreInputValidation(unittest.TestCase):
    def test_reject_empty_content(self):
        store = SqacStore()
        with self.assertRaises(ValueError):
            store.add("")

    def test_reject_whitespace_only(self):
        store = SqacStore()
        with self.assertRaises(ValueError):
            store.add(chr(32)*3 + chr(10) + chr(32)*2 + chr(9) + chr(32)*2)

    def test_strips_control_chars(self):
        store = SqacStore()
        idx = store.add("hello" + chr(0) + "world", key="test")
        hits = store.search("test")
        self.assertTrue(hits)
        self.assertNotIn(chr(0), hits[0].content)


class TestCompactCLI(unittest.TestCase):
    def test_compact_command(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.sqac"
            # Build a store with a tombstone
            store = SqacStore()
            store.add("alive", key="alive")
            idx = store.add("dead", key="dead")
            store.delete(idx)
            store.save(db)

            # Verify tombstone exists
            cart = read_cartridge(db)
            self.assertEqual(len(cart.entries), 2)
            self.assertTrue(cart.entries[1].deleted)

            # Run compact via CLI
            from sqac.cli import main
            ret = main(["compact", "--db", str(db)])
            self.assertEqual(ret, 0)

            # Verify tombstone removed
            cart = read_cartridge(db)
            self.assertEqual(len(cart.entries), 1)
            self.assertFalse(cart.entries[0].deleted)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the SQAC cartridge system.

Run: python -m pytest tests/test_sqac.py -q   (or python -m unittest)
"""

import os
import tempfile
import unittest

from sqac.encoder import BSCEncoder
from sqac.format import FormatError, read_cartridge, write_cartridge, CartridgeHeader, Entry
from sqac.store import SqacStore


class TestEncoder(unittest.TestCase):
    def test_deterministic_across_instances(self):
        a = BSCEncoder().encode_bits("hello world")
        b = BSCEncoder().encode_bits("hello world")
        self.assertEqual(a, b)

    def test_similarity_structure(self):
        enc = BSCEncoder()
        ident = enc.similarity(enc.encode_bits("auth middleware"), enc.encode_bits("auth middleware"))
        near = enc.similarity(enc.encode_bits("auth middleware"), enc.encode_bits("auth midleware"))
        far = enc.similarity(enc.encode_bits("auth middleware"), enc.encode_bits("banana pancake"))
        self.assertEqual(ident, 1.0)
        self.assertGreater(near, far, "typo variant should be more similar than unrelated text")
        self.assertGreater(far, 0.35, "unrelated should hover near the 0.5 noise floor, not 0")
        self.assertLess(far, 0.65)

    def test_dims_validation(self):
        with self.assertRaises(ValueError):
            BSCEncoder(dims=100)  # not multiple of 8
        with self.assertRaises(ValueError):
            BSCEncoder(dims=32)  # too small


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = SqacStore()
        self.store.add(
            "Use require_scopes() middleware for all authenticated endpoints",
            key="how to handle auth in the api",
            source="team-handbook",
        )
        self.store.add(
            "Always use repository pattern, never raw SQL in controllers",
            key="database access rule",
            source="team-handbook",
        )
        self.store.add(
            "Deploy only to ARM64; AMD64 images are not supported",
            key="deployment target",
            source="ops",
        )

    def test_exact_hit(self):
        hits = self.store.search("how to handle auth in the api")
        self.assertEqual(hits[0].mode, "exact")
        self.assertEqual(hits[0].confidence, 1.0)
        self.assertIn("require_scopes", hits[0].content)

    def test_fuzzy_by_content_overlap(self):
        hits = self.store.search("auth middleware for endpoints")
        self.assertTrue(hits, "expected fuzzy hit")
        self.assertEqual(hits[0].mode, "fuzzy")
        self.assertIn("require_scopes", hits[0].content)

    def test_fuzzy_with_typos(self):
        hits = self.store.search("authenticaed endpoint midleware")
        self.assertTrue(hits)
        self.assertIn("require_scopes", hits[0].content)

    def test_semantic_gap_fails_safe(self):
        # Synonym-only queries are below the lexical tier (thesis: MiniLM tier).
        # The store must return [] rather than a confident wrong answer.
        hits = self.store.search("favorite color of the ceo")
        self.assertEqual(hits, [])

    def test_teach_roundtrip_persists(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.sqac")
            self.store.save(path, name="t")
            loaded = SqacStore.load(path)
            self.assertEqual(len(loaded), len(self.store))
            h1 = self.store.search("auth middleware for endpoints")
            h2 = loaded.search("auth middleware for endpoints")
            self.assertEqual(h1[0].content, h2[0].content)
            self.assertAlmostEqual(h1[0].confidence, h2[0].confidence, places=6)

    def test_fingerprint_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.sqac")
            self.store.save(path)
            cart = read_cartridge(path)
            # Tamper at the byte level: write_cartridge would recompute the
            # fingerprint, so patch the raw bytes instead.
            fp = cart.header.vocab_fingerprint.encode()
            blob = open(path, "rb").read().replace(fp, b"0" * 32)
            tampered = os.path.join(td, "tampered.sqac")
            with open(tampered, "wb") as f:
                f.write(blob)
            with self.assertRaises(FormatError):
                SqacStore.load(tampered)

    def test_delete_is_tombstone(self):
        idx = self.store.add("temp fact", key="temp fact")
        self.assertEqual(len(self.store), 4)
        self.assertTrue(self.store.delete(idx))
        self.assertEqual(len(self.store), 3)
        self.assertEqual(self.store.search("temp fact"), [])


class TestFormat(unittest.TestCase):
    def test_roundtrip_preserves_everything(self):
        header = CartridgeHeader(dims=256, name="x", description="y")
        entries = [
            Entry(key_bits=bytearray(b"\xaa" * 32), payload={"content": "hello", "n": 1}),
            Entry(key_bits=bytearray(b"\x55" * 32), payload={"content": "world"}, deleted=True),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "c.sqac")
            write_cartridge(path, header, {"atom": 3}, entries, ext_blocks=[("probe", b"\x01\x02")])
            cart = read_cartridge(path)
        self.assertEqual(cart.header.name, "x")
        self.assertEqual(cart.header.dims, 256)
        self.assertEqual(cart.vocab, {"atom": 3})
        self.assertEqual(len(cart.entries), 2)
        self.assertEqual(cart.entries[0].payload["content"], "hello")
        self.assertTrue(cart.entries[1].deleted)
        self.assertEqual(cart.ext_blocks["probe"], b"\x01\x02")

    def test_truncated_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "c.sqac")
            write_cartridge(
                path,
                CartridgeHeader(dims=256),
                {},
                [Entry(key_bits=bytearray(32), payload={})],
            )
            blob = open(path, "rb").read()
            bad = os.path.join(td, "bad.sqac")
            with open(bad, "wb") as f:
                f.write(blob[: len(blob) // 2])
            with self.assertRaises(FormatError):
                read_cartridge(bad)

if __name__ == "__main__":
    unittest.main()

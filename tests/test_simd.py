"""Parity tests for the bundled Rust SIMD backend."""

import random
import unittest

import numpy as np

from sqac.simd import HAS_SIMD, backend


def _vectors(n=32, key_len=128):
    rng = random.Random(1234)
    query = bytes(rng.randrange(256) for _ in range(key_len))
    keys = [bytes(rng.randrange(256) for _ in range(key_len)) for _ in range(n)]
    ckeys = [bytes(rng.randrange(256) for _ in range(key_len)) for _ in range(n)]
    return query, keys, ckeys


class TestSimdFlag(unittest.TestCase):
    def test_flag_matches_store_stats(self):
        from sqac.store import SqacStore

        self.assertEqual(SqacStore().stats()["simd"], HAS_SIMD)


@unittest.skipUnless(HAS_SIMD, "Rust extension not built")
class TestSimdParity(unittest.TestCase):
    def test_bulk_similarity_matches_numpy(self):
        query, keys, _ = _vectors()
        dims = len(query) * 8
        got = backend.bulk_similarity(query, b"".join(keys), len(keys), dims)
        q = np.unpackbits(np.frombuffer(query, dtype=np.uint8))
        for i, key in enumerate(keys):
            v = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
            expected = 1.0 - float(np.count_nonzero(q != v)) / dims
            self.assertAlmostEqual(got[i], expected, places=12)

    def test_pack_and_similarity_matches_numpy(self):
        query, keys, ckeys = _vectors()
        dims = len(query) * 8
        live = list(range(len(keys)))
        sims_k, sims_c = backend.pack_and_similarity(
            query, keys, ckeys, live, len(query), dims
        )
        q = np.unpackbits(np.frombuffer(query, dtype=np.uint8))
        for i, (key, ckey) in enumerate(zip(keys, ckeys)):
            for src, got in ((key, sims_k[i]), (ckey, sims_c[i])):
                v = np.unpackbits(np.frombuffer(src, dtype=np.uint8))
                expected = 1.0 - float(np.count_nonzero(q != v)) / dims
                self.assertAlmostEqual(got, expected, places=12)

    def test_identical_and_opposite(self):
        dims = 1024
        zero = bytes(dims // 8)
        ones = b"\xff" * (dims // 8)
        sims = backend.bulk_similarity(zero, zero + ones, 2, dims)
        self.assertAlmostEqual(sims[0], 1.0, places=12)
        self.assertAlmostEqual(sims[1], 0.0, places=12)

#!/usr/bin/env python3
"""Tests for the SQAC pre-encoding normalization hook (deliverable 2)."""
import os
import unittest

from sqac import textnorm
from sqac.encoder import BSCEncoder
from sqac.store import SqacStore


class TestTextnormSteps(unittest.TestCase):
    def test_unicode_fold(self):
        self.assertEqual(textnorm.unicode_fold("naïve café résumé"), "naive cafe resume")

    def test_stopwords_en(self):
        out = textnorm.stopwords("en")("the quick brown fox and the lazy dogs")
        self.assertEqual(out, "quick brown fox lazy dogs")

    def test_stopwords_file(self):
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as f:
            f.write("balki\ndan\n")
            path = f.name
        try:
            out = textnorm.stopwords(f"file={path}")("balki ako dan pa")
            self.assertEqual(out, "ako pa")
        finally:
            os.unlink(path)

    def test_affix_en(self):
        out = textnorm.affix("en")("running tested connects quickly")
        self.assertEqual(out, "runn test connect quick")

    def test_affix_respects_min_stem(self):
        # "is"/"as" — too short to strip anything unsafe
        out = textnorm.affix("en")("bus gas")
        self.assertEqual(out, "bus gas")


class TestTextnormResolve(unittest.TestCase):
    def test_pipeline_order(self):
        norm = textnorm.resolve("unicode_fold,stopwords:en,affix:en")
        self.assertEqual(norm("Naïvely were running quickly"),
                         "naive runn quick")

    def test_from_env(self):
        os.environ["SQAC_NORMALIZE"] = "unicode_fold"
        try:
            norm = textnorm.from_env()
            self.assertIsNotNone(norm)
            self.assertEqual(norm("café"), "cafe")
        finally:
            os.environ.pop("SQAC_NORMALIZE", None)
        self.assertIsNone(textnorm.from_env())

    def test_unknown_step_raises(self):
        with self.assertRaises(ValueError):
            textnorm.resolve("bogus_step")


class TestEncoderHook(unittest.TestCase):
    def test_default_encoder_unchanged(self):
        enc = BSCEncoder(dims=1024)
        same = enc.encode_bits("running tests") == enc.encode_bits("running tests")
        self.assertTrue(same)
        self.assertNotEqual(enc.encode_bits("running tests"), enc.encode_bits("run tests"))

    def test_normalizer_applied_before_encoding(self):
        plain = BSCEncoder(dims=1024)
        norm = BSCEncoder(dims=1024, normalizer=textnorm.resolve("affix:en"))
        self.assertEqual(norm.encode_bits("running tests"), plain.encode_bits("runn test"))


class TestStoreIntegration(unittest.TestCase):
    def test_exact_tier_uses_instance_normalizer(self):
        store = SqacStore(semantic=False,
                          normalizer=textnorm.resolve("unicode_fold"))
        store.add("deploy target asia", key="naïve-deploy", kind="fact")
        hits = store.search("naive-deploy", top_k=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0].mode, "exact")

    def test_roundtrip_across_normalizers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "facts.sqac")
            store = SqacStore(semantic=False,
                              normalizer=textnorm.resolve("unicode_fold"))
            store.add("café memory", key="cafe-key", kind="fact")
            store.save(path, name="facts")
            # loading WITHOUT the normalizer must not break (fingerprint-stable)
            loaded = SqacStore.load(path)
            self.assertEqual(len(loaded), 1)


class TestServerWireup(unittest.TestCase):
    def test_make_rack_picks_up_env_normalizer(self):
        import tempfile, shutil
        import sqac.server as srv
        os.environ["SQAC_NORMALIZE"] = "unicode_fold"
        try:
            with tempfile.TemporaryDirectory() as d:
                rack = srv._make_rack(d)
                self.assertIsNotNone(rack._normalizer)
                self.assertEqual(rack._normalizer("café"), "cafe")
        finally:
            os.environ.pop("SQAC_NORMALIZE", None)


if __name__ == "__main__":
    unittest.main()

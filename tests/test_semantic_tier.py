"""Semantic tier (MiniLM SimHash) tests.

Skipped automatically when torch/transformers are unavailable.
Run: python -m unittest tests.test_semantic_tier
"""

import os
import tempfile
import unittest

from sqac.encoder import MiniLMSimHashEncoder
from sqac.store import SqacStore

SEMANTIC_AVAILABLE = MiniLMSimHashEncoder.available()


@unittest.skipUnless(SEMANTIC_AVAILABLE, "torch/transformers not installed")
class TestSemanticTier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enc = MiniLMSimHashEncoder()

    def test_simhash_calibration_structure(self):
        """Floor ~0.5, identical 1.0, paraphrase well above floor."""
        enc = self.enc
        ident = enc.similarity(enc.encode_bits("deploy the service"), enc.encode_bits("deploy the service"))
        para = enc.similarity(
            enc.encode_bits("how do I handle auth"),
            enc.encode_bits("what is the way to manage authentication"),
        )
        floor = enc.similarity(
            enc.encode_bits("favorite color of the ceo"),
            enc.encode_bits("Deploy only to ARM64; AMD64 images are not supported"),
        )
        self.assertEqual(ident, 1.0)
        self.assertGreater(para, 0.65)
        self.assertLess(floor, 0.55)
        self.assertGreater(floor, 0.40)

    def test_synonym_recall_that_lexical_missed(self):
        """The canonical case: x86 query must find the ARM64/AMD64 rule."""
        store = SqacStore(semantic=True)
        store.add(
            "Deploy only to ARM64; AMD64 images are not supported",
            key="deployment target",
            source="ops",
        )
        hits = store.search("can we deploy on x86?", top_k=1)
        self.assertTrue(hits, "semantic tier must catch x86 -> ARM64")
        self.assertEqual(hits[0].mode, "semantic")
        self.assertIn("ARM64", hits[0].content)

    def test_unrelated_query_fails_safe(self):
        store = SqacStore(semantic=True)
        store.add("Deploy only to ARM64; AMD64 images are not supported", key="deployment target")
        hits = store.search("favorite color of the ceo", top_k=3)
        self.assertEqual(hits, [], "unrelated query must not return confident garbage")

    def test_semantic_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sem.sqac")
            store = SqacStore(semantic=True)
            store.add("Always use repository pattern, never raw SQL", key="database access rule")
            store.save(path, name="sem")
            loaded = SqacStore.load(path)
            self.assertTrue(loaded.semantic)
            hits = loaded.search("can we deploy on x86?", top_k=1)  # unrelated, but tier must be armed
            # main assertion: reload produced working semantic vectors
            self.assertTrue(loaded._sem_keys[0] is not None)


if __name__ == "__main__":
    unittest.main()

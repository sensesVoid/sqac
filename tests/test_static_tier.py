"""Static quantized semantic tier (potion-base-8M, pure numpy).

Validates the light tier on SQAC's own calibration tasks and checks
cartridge compatibility: the encoder identity rides in the header
(|sem-v1:static-potion-8m-int8) and must roundtrip through save/load.

These tests are skipped when the model files are absent (CI, fresh clone):
    python -c "from huggingface_hub import hf_hub_download; ..."
See sqac/static_encoder.py docstring for the one-command download.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.encoder import MiniLMSimHashEncoder
from sqac.static_encoder import StaticSimHashEncoder
from sqac.store import SqacStore

MODEL_DIR = os.environ.get("SQAC_STATIC_MODEL_DIR", "sqac/models/potion-8m")


@unittest.skipUnless(
    StaticSimHashEncoder.available(MODEL_DIR),
    f"static model files not found in {MODEL_DIR}",
)
class TestStaticEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enc = StaticSimHashEncoder(model_dir=MODEL_DIR)

    def test_paraphrase_above_threshold(self):
        a = "If a system component fails, the backup takes over seamlessly"
        b = "When a part of the system goes down, the standby system continues operation"
        sim = self.enc.similarity(self.enc.encode_bits(a), self.enc.encode_bits(b))
        self.assertGreaterEqual(sim, 0.60, f"paraphrase sim {sim:.3f} below threshold")

    def test_semantic_catch_x86_arm64(self):
        """The flagship case: 'x86' query must reach the ARM64 rule."""
        a = "all production deploys must target ARM64 architecture"
        b = "the build pipeline only supports x86 images"
        sim = self.enc.similarity(self.enc.encode_bits(a), self.enc.encode_bits(b))
        self.assertGreaterEqual(sim, 0.60, f"x86->ARM64 sim {sim:.3f} below threshold")

    def test_no_leakage_on_unrelated(self):
        """Unrelated pairs stay near the 0.5 noise floor (leakage guard)."""
        pairs = [
            ("quantum flux capacitor illuminates purple elephants",
             "our deployment policy requires two approvers"),
            ("the chef garnished the risotto with truffle oil",
             "kernel panics often indicate driver incompatibility"),
        ]
        for a, b in pairs:
            sim = self.enc.similarity(self.enc.encode_bits(a), self.enc.encode_bits(b))
            self.assertLess(sim, 0.60, f"unrelated pair scored {sim:.3f}")

    def test_batch_matches_single(self):
        texts = ["alpha beta gamma", "deployment approval policy", "x86 image build"]
        batch = self.enc.encode_bits_batch(texts)
        for t, b in zip(texts, batch):
            self.assertEqual(b, self.enc.encode_bits(t))

    def test_empty_text_is_zero_vector(self):
        self.assertEqual(self.enc.encode_bits(""), bytes(self.enc.dims // 8))
        self.assertEqual(self.enc.encode_bits("   "), bytes(self.enc.dims // 8))


@unittest.skipUnless(
    StaticSimHashEncoder.available(MODEL_DIR),
    f"static model files not found in {MODEL_DIR}",
)
class TestStaticStoreIntegration(unittest.TestCase):
    TMP = ".test_static_tier.sqac"

    def tearDown(self):
        if os.path.exists(self.TMP):
            os.unlink(self.TMP)

    def test_auto_selection_prefers_static(self):
        store = SqacStore(semantic=True)
        self.assertIsInstance(store._sem_encoder, StaticSimHashEncoder)

    def test_explicit_minilm_still_selectable(self):
        if not MiniLMSimHashEncoder.available():
            self.skipTest("torch+transformers not installed")
        store = SqacStore(semantic=True, semantic_model=MiniLMSimHashEncoder.DEFAULT_MODEL)
        self.assertIsInstance(store._sem_encoder, MiniLMSimHashEncoder)

    def test_cartridge_roundtrip_static(self):
        store = SqacStore(semantic=True)
        store.add("all deploys are ARM64 only", key="deployment target")
        store.add("auth middleware validates every request", key="api security")
        store.save(self.TMP, name="static-test")

        loaded = SqacStore.load(self.TMP)
        self.assertIsInstance(loaded._sem_encoder, StaticSimHashEncoder)
        self.assertTrue(loaded.semantic)

        # semantic vectors persisted, not recomputed
        self.assertIsNotNone(loaded._sem_keys[0])
        hits = loaded.search("what image do we deploy for x86 machines", top_k=1)
        self.assertTrue(hits, "semantic search found nothing after reload")
        self.assertIn("ARM64", hits[0].content)

    def test_minilm_cartridge_still_loads(self):
        """A MiniLM-keyed cartridge must load on the MiniLM path (compat)."""
        if not MiniLMSimHashEncoder.available():
            self.skipTest("torch+transformers not installed")
        tmp = ".test_minilm_cartridge.sqac"
        try:
            store = SqacStore(semantic=True, semantic_model=MiniLMSimHashEncoder.DEFAULT_MODEL)
            store.add("legacy fact encoded with minilm", key="legacy")
            store.save(tmp, name="minilm-test")
            loaded = SqacStore.load(tmp)
            self.assertIsInstance(loaded._sem_encoder, MiniLMSimHashEncoder)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()

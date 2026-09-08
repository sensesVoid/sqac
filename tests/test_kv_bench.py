import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqac.kvcache as kvc


class KVConstants(unittest.TestCase):
    def test_llama31_8b_per_token_bytes(self):
        # 32 layers × 8 KV heads × (K+V) × 128 head_dim × fp16(2B)
        self.assertEqual(kvc.model_kv_bytes_per_token("llama-3.1-8b"), 131072)

    def test_known_geometries(self):
        for m in kvc.known_models():
            self.assertGreater(kvc.model_kv_bytes_per_token(m), 0)

    def test_precision_variants(self):
        base = kvc.model_kv_bytes_per_token("llama-3.1-8b")  # fp16
        self.assertEqual(base, kvc.model_kv_bytes_per_token("llama-3.1-8b", "bf16"))
        self.assertEqual(base * 2, kvc.model_kv_bytes_per_token("llama-3.1-8b", "fp32"))
        self.assertEqual(base // 2, kvc.model_kv_bytes_per_token("llama-3.1-8b", "fp8"))

    def test_unknown_precision_rejected(self):
        with self.assertRaises(ValueError):
            kvc.model_kv_bytes_per_token("llama-3.1-8b", "int4")


class KVEstimate(unittest.TestCase):
    def test_matches_literature_figure(self):
        # 131,072 B/token × 1M tokens = 122.07 GiB (Llama-3.1-8B, fp16, GQA)
        est = kvc.estimate_sparse_recall(1_000_000, 1_000_000)
        self.assertAlmostEqual(est.stuffing_kv_bytes / 2**30, 122.07, delta=0.01)

    def test_stuffing_equals_sparse_at_full_recall(self):
        est = kvc.estimate_sparse_recall(5000, 5000)
        self.assertEqual(est.reduction_x, 1.0)
        self.assertAlmostEqual(est.savings_pct, 0.0)

    def test_ratio_equals_token_ratio(self):
        est = kvc.estimate_sparse_recall(135_000, 135)
        self.assertAlmostEqual(est.reduction_x, 1000.0, delta=0.01)
        self.assertEqual(round(est.savings_pct, 1), 99.9)

    def test_topk_sweep_descending_reduction(self):
        ests = kvc.estimate_sweep(135_000, (1, 3, 10))
        reductions = [e.reduction_x for e in ests]
        self.assertEqual(reductions, sorted(reductions, reverse=True))
        self.assertGreater(ests[0].reduction_x, 100)

    def test_linear_scaling(self):
        base = kvc.estimate_sparse_recall(1000, 100).sparse_kv_bytes  # 100 tok
        ten = kvc.estimate_sparse_recall(10000, 1000).sparse_kv_bytes  # 1000 tok
        self.assertEqual(ten / base, 10.0)


if __name__ == "__main__":
    unittest.main()
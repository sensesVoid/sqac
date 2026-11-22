import os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqac.attention import render_block, count_tokens, CURRENT_CAP, PENDING_CAP, MAX_TOKENS
from sqac.offloader import ContextOffloader

class TestRenderBlock(unittest.TestCase):
    def test_header(self):
        out = render_block(7, [], [])
        self.assertEqual(out, "[SQAC attention | turn 7]\n")

    def test_sections_and_caps(self):
        current = [(f"task {i}", f"m{i}") for i in range(10)]
        pending = [(f"pend {i}", f"p{i}") for i in range(12)]
        out = render_block(12, current, pending)
        self.assertIn("Current:", out)
        self.assertIn("Pending:", out)
        self.assertEqual(out.count("\n- task"), CURRENT_CAP)
        self.assertEqual(out.count("\n- pend"), PENDING_CAP)
        self.assertIn(f"- +{10 - CURRENT_CAP} more", out)
        self.assertIn(f"- +{12 - PENDING_CAP} more", out)

    def test_mem_ids_present(self):
        out = render_block(1, [("a", "MID-A")], [])
        self.assertIn("(mem_id MID-A)", out)

    def test_empty_sections_omitted(self):
        out = render_block(2, [], [("p", "M1")])
        self.assertNotIn("Current:", out)
        self.assertIn("Pending:", out)

    def test_token_budget(self):
        long_content = "z" * 200
        out = render_block(0, [(long_content, "m0")] * 20, [(long_content, "m1")] * 20)
        self.assertLessEqual(count_tokens(out), MAX_TOKENS)

class TestOffloaderCadence(unittest.TestCase):
    def _o(self):
        return ContextOffloader(path=None, window=8, semantic=False)

    def test_attention_due_every_n_turns(self):
        o = self._o()
        for i in range(1, 5):
            o._turn_count = i
            self.assertFalse(o.attention_due(5))
        o._turn_count = 5
        self.assertTrue(o.attention_due(5))
        o._turn_count = 10
        self.assertTrue(o.attention_due(5))

    def test_attention_dirty(self):
        o = self._o()
        self.assertFalse(o.take_attention_dirty())
        o.mark_attention_dirty()
        self.assertTrue(o.take_attention_dirty())
        self.assertFalse(o.take_attention_dirty())

if __name__ == "__main__":
    unittest.main()

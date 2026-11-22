import os, shutil, sys, tempfile, unittest
from pathlib import Path
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

class TestFallback(unittest.TestCase):
    def test_current_from_continuity(self):
        from sqac.attention import render_block_with_fallback
        from sqac.continuity import ContinuityStore
        with tempfile.TemporaryDirectory() as d:
            cs = ContinuityStore.load(Path(d) / "continuity.json")
            cs.touch("myproj", goal="ship v3")
            out = render_block_with_fallback(3, [], [], continuity=cs, project="myproj")
            self.assertIn("ship v3", out)
            self.assertIn("Current:", out)

    def test_pending_from_offloader(self):
        from sqac.attention import render_block_with_fallback
        from sqac.offloader import Turn
        off = ContextOffloader(path=None, window=8, semantic=False)
        off._buffer = [Turn(role="user", text="recent deploy question"), Turn(role="assistant", text="answer")]
        out = render_block_with_fallback(0, [], [], offloader=off)
        self.assertIn("recent deploy question", out)
        self.assertIn("Pending:", out)

    def test_no_fallback_when_tasks_exist(self):
        from sqac.attention import render_block_with_fallback
        from sqac.continuity import ContinuityStore
        with tempfile.TemporaryDirectory() as d:
            cs = ContinuityStore.load(Path(d) / "continuity.json")
            cs.touch("myproj", goal="ship v3")
            out = render_block_with_fallback(3, [("my task", "m1")], [], continuity=cs, project="myproj")
            self.assertIn("my task", out)
            self.assertNotIn("ship v3", out)

    def test_offloader_recent(self):
        from sqac.offloader import Turn
        off = ContextOffloader(path=None, window=8, semantic=False)
        off._buffer = [Turn(role="user", text="u1"), Turn(role="assistant", text="a1"),
                       Turn(role="user", text="u2")]
        self.assertEqual(off.recent(2), ["a1", "u2"])


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

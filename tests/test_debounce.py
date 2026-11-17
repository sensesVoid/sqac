"""Session-bucket save debounce tests (v1.2.2).

mem_observe no longer rewrites session.sqac on every turn; saves are
debounced by turn count / elapsed seconds and forced on explicit persist
points (mem_save, mem_checkpoint, mem_graduate, mem_swap).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqac.offloader import ContextOffloader, Turn  # noqa: F401


def _offloader(tmp_path, **kwargs):
    return ContextOffloader(tmp_path / "session.sqac", window=8, semantic=False, **kwargs)


class TestShouldFlush(unittest.TestCase):
    def test_false_on_fresh_offloader(self):
        d = Path(self._td())
        o = _offloader(d)
        self.assertFalse(o.should_flush(10, 60))

    def test_true_after_turn_threshold(self):
        d = Path(self._td())
        o = _offloader(d)
        for i in range(9):
            o.observe("user", f"turn {i}")
        self.assertFalse(o.should_flush(10, 60))
        o.observe("user", "turn 9")
        self.assertTrue(o.should_flush(10, 60))

    def test_reset_after_save(self):
        d = Path(self._td())
        o = _offloader(d)
        for i in range(10):
            o.observe("user", f"turn {i}")
        self.assertTrue(o.should_flush(10, 60))
        o.save()
        self.assertFalse(o.should_flush(10, 60))
        self.assertFalse(o._since_flush)

    def test_secs_zero_forces_flush(self):
        d = Path(self._td())
        o = _offloader(d)
        o.observe("user", "only turn")
        self.assertTrue(o.should_flush(10, 0))

    def test_turns_one_restores_legacy(self):
        d = Path(self._td())
        o = _offloader(d)
        self.assertFalse(o.should_flush(1, 60))
        o.observe("user", "t")
        self.assertTrue(o.should_flush(1, 60))

    def test_empty_turn_does_not_count(self):
        d = Path(self._td())
        o = _offloader(d)
        o.observe("user", "")
        o.observe("user", "   ")
        self.assertFalse(o.should_flush(1, 60))

    def _td(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return self._tmp.name


@unittest.skipUnless(Path("sqac_mcp.py").exists(), "repo root layout expected")
class TestMcpDebounce(unittest.TestCase):
    """End-to-end: mem_observe debounces session.sqac rewrites."""

    def setUp(self):
        os.environ["SQAC_SEMANTIC"] = "0"
        os.environ["SQAC_SESSION_FLUSH_TURNS"] = "10"
        os.environ["SQAC_SESSION_FLUSH_SECS"] = "60"
        self._td = tempfile.TemporaryDirectory()
        os.environ["SQAC_MEM_DIR"] = self._td.name

    def tearDown(self):
        self._td.cleanup()
        for key in ("SQAC_SEMANTIC", "SQAC_SESSION_FLUSH_TURNS",
                    "SQAC_SESSION_FLUSH_SECS", "SQAC_MEM_DIR"):
            os.environ.pop(key, None)

    def test_observe_debounces_until_threshold(self):
        from mcp import Client, StdioServerParameters
        import asyncio

        async def scenario():
            params = StdioServerParameters(
                command=sys.executable, args=["sqac_mcp.py"],
                env={
                    "SQAC_MEM_DIR": self._td.name,
                    "SQAC_SEMANTIC": "0",
                    "SQAC_SESSION_FLUSH_TURNS": "10",
                    "SQAC_SESSION_FLUSH_SECS": "60",
                    "PATH": os.environ.get("PATH", ""),
                },
            )
            async with Client(params) as c:
                for i in range(9):
                    await c.call_tool("mem_observe",
                                      {"role": "user", "text": f"turn {i}: hello"})
                    self.assertFalse(
                        (Path(self._td.name) / "session.sqac").exists(),
                        f"session.sqac should not exist after turn {i + 1}",
                    )
                await c.call_tool("mem_observe",
                                  {"role": "user", "text": "turn 9: hello"})
                self.assertTrue((Path(self._td.name) / "session.sqac").exists())

        try:
            asyncio.run(scenario())
        except Exception as exc:  # pragma: no cover - diagnostic aid
            self.fail(f"scenario failed: {exc}")


if __name__ == "__main__":
    unittest.main()

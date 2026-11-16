"""Cross-CLI integration tests: memory persists across sessions."""

import os
import tempfile
import unittest

from sqac.store import SqacStore
from sqac.continuity import ContinuityStore, bootstrap_packet


class TestCrossCliIntegration(unittest.TestCase):
    def test_memory_persists_across_sessions(self):
        """Facts written in one session are readable in another."""
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "memory.sqac")
            # Session 1: write
            store = SqacStore()
            store.add("Team Atlas maintains payments", key="payments", kind="fact")
            store.save(db)
            # Session 2: read
            store2 = SqacStore.load(db)
            hits = store2.search("who maintains payments?")
            self.assertTrue(hits)
            self.assertIn("Atlas", hits[0].content)

    def test_continuity_bootstrap_roundtrip(self):
        """Bootstrap packet reflects prior session state."""
        with tempfile.TemporaryDirectory() as td:
            cpath = os.path.join(td, "continuity.json")
            cs = ContinuityStore(cpath)
            cs.touch("default", host="opencode", goal="migrate auth", summary="scope agreed")
            cs.checkpoint("default", host="opencode", goal="migrate auth", summary="scope agreed")
            cs.save()

            # New session (different CLI)
            cs2 = ContinuityStore.load(cpath)
            pkt = bootstrap_packet("default", cs2, host="claude-code")
            self.assertEqual(pkt["last_state"]["last_host"], "opencode")
            self.assertEqual(pkt["last_state"]["goal"], "migrate auth")
            self.assertEqual(len(pkt["checkpoints"]), 1)

    def test_cross_cli_memory_dir_shared(self):
        """Two store instances sharing the same directory see each other's writes."""
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "shared.sqac")
            s1 = SqacStore()
            s1.add("fact from cli A", key="fact-a", kind="fact")
            s1.save(db)

            s2 = SqacStore.load(db)
            s2.add("fact from cli B", key="fact-b", kind="fact")
            s2.save(db)

            s3 = SqacStore.load(db)
            self.assertEqual(len(s3), 2)
            hits = s3.search("fact from cli")
            self.assertGreaterEqual(len(hits), 1)

    def test_agent_bootstrap_cross_session(self):
        """Agent state persists across sessions."""
        import sqac.agent as am
        from pathlib import Path
        td = tempfile.mkdtemp()
        am.AGENT_DIR = Path(td) / "agents"
        am.ACTIVE_AGENT_PATH = Path(td) / "agent_active"

        ag = am.create_agent("atlas", role="assistant")
        ag.set("identity", "I am Atlas")
        ag.set("goal", "Complete migration", priority=2)
        ag.save()

        # New session
        ag2 = am.load_agent("atlas")
        bp = ag2.bootstrap()
        self.assertIn("Atlas", bp)
        self.assertIn("Complete migration", bp)


if __name__ == "__main__":
    unittest.main()

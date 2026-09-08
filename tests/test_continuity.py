import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqac.continuity import (
    ContinuityStore,
    bootstrap_packet,
    detect_host,
    server_instructions,
)
from sqac import mcp_setup


class ContinuityStoreTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = Path(self.td.name) / "continuity.json"

    def tearDown(self):
        self.td.cleanup()

    def test_touch_save_load_roundtrip(self):
        c = ContinuityStore(self.path)
        c.touch("acme", host="opencode", model="m-a", goal="migrate auth", summary="scoped")
        c.save()
        c2 = ContinuityStore.load(self.path)
        rec = c2.record("acme")
        self.assertEqual(rec["last_host"], "opencode")
        self.assertEqual(rec["goal"], "migrate auth")
        self.assertEqual(rec["last_summary"], "scoped")
        self.assertIsNotNone(rec["last_activity"])

    def test_checkpoint_appends_trail_and_caps(self):
        c = ContinuityStore(self.path)
        for i in range(15):
            c.checkpoint("p", host="h", goal=f"g{i}", summary=f"s{i}")
        self.assertEqual(len(c.last_checkpoints("p", n=10)), 10)
        self.assertEqual(c.last_checkpoints("p", n=1)[0]["goal"], "g14")

    def test_corrupt_file_starts_clean(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        c = ContinuityStore.load(self.path)
        self.assertEqual(c.projects(), [])
        c.touch("ok", host="h")
        c.save()
        self.assertIn("ok", ContinuityStore.load(self.path).projects())

    def test_new_project_honest_no_state(self):
        c = ContinuityStore(self.path)
        pkt = bootstrap_packet("fresh", c, host="claude-code", model="m")
        self.assertIn("no prior state", pkt["summary"])
        self.assertIn("fresh", pkt["continuity_note"])
        self.assertEqual(pkt["recent"], [])

    def test_bootstrap_packet_carries_handoff(self):
        c = ContinuityStore(self.path)
        c.checkpoint("acme", host="opencode", goal="deploy ARM64", summary="image pinned slim")
        pkt = bootstrap_packet("acme", c, host="claude-code", model="m2",
                               session_stats={"entries": 4}, rack_stats={"facts": {"entries": 9}})
        self.assertEqual(pkt["last_state"]["goal"], "deploy ARM64")
        self.assertIn("deploy ARM64", pkt["summary"])
        self.assertIn("opencode", pkt["summary"])  # prior host surfaced
        self.assertEqual(pkt["memory_entries"], 9)
        self.assertEqual(pkt["session"]["entries"], 4)

    def test_instructions_are_memory_protocol(self):
        s = server_instructions()
        for token in ("mem_bootstrap", "mem_checkpoint", "mem_observe", "mem_search"):
            self.assertIn(token, s)


class HostDetectionTest(unittest.TestCase):
    def test_override_wins(self):
        self.assertEqual(detect_host("cursor"), "cursor")

    def test_known_or_unknown(self):
        valid = {"opencode", "claude-code", "codex", "cursor", "zed", "cody",
                 "gh-copilot", "cinnamon", "continue", "mcp-inspector", "unknown"}
        self.assertIn(detect_host(), valid)


class McpSetupTest(unittest.TestCase):
    def test_opencode_snippet_points_at_shared_dir(self):
        s = mcp_setup.opencode("/shared/mem")
        self.assertIn('"command": "sqac-mcp --dir /shared/mem"', s)
        self.assertIn('"type": "local"', s)

    def test_claude_code_command(self):
        s = mcp_setup.claude_code("/shared/mem")
        self.assertIn("claude mcp add sqac --stdio", s)

    def test_all_snippets_include_hook(self):
        s = mcp_setup.all_snippets("/mem")
        for key in ("opencode", "Claude Code", "Claude Desktop", "Codex", "Cursor", "Zed"):
            self.assertIn(key, s)
        self.assertIn("mem_bootstrap", s)

    def test_hook_project_writes_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            proj = Path(td)
            rc1 = mcp_setup.main(["--host", "opencode", "--hook-project", str(proj)])
            self.assertEqual(rc1, 0)
            agents = proj / "AGENTS.md"
            self.assertTrue(agents.exists())
            first = agents.read_text(encoding="utf-8")
            self.assertIn("mem_bootstrap", first)
            mcp_setup.main(["--host", "opencode", "--hook-project", str(proj)])
            second = agents.read_text(encoding="utf-8")
            self.assertEqual(second.count("mem_bootstrap"), first.count("mem_bootstrap"),
                             "re-running must not duplicate the hook")

    def test_unknown_host_errors(self):
        rc = mcp_setup.main(["--host", "nope"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
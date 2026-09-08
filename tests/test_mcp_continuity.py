import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_HAS_MCP = importlib.util.find_spec("mcp") is not None


def _text(result):
    return result.content[0].text


def _tool_names(result):
    """list_tools result shapes vary across mcp SDK versions; unwrap safely."""
    data = result.tools if hasattr(result, "tools") else (
        result["tools"] if isinstance(result, dict) else result
    )
    return {(t.name if hasattr(t, "name") else t["name"]) for t in data}


@unittest.skipUnless(_HAS_MCP, "mcp optional dependency not installed")
class McpServerContinuityTest(unittest.TestCase):
    """Spins the real sqac-mcp stdio server and drives it like an agent CLI."""

    def setUp(self):
        os.environ["SQAC_SEMANTIC"] = "0"  # lexical-only: no model files needed
        self._td = tempfile.TemporaryDirectory()
        os.environ["SQAC_MEM_DIR"] = self._td.name

    def tearDown(self):
        self._td.cleanup()
        os.environ.pop("SQAC_SEMANTIC", None)
        os.environ.pop("SQAC_MEM_DIR", None)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_cross_cli_continuity(self):
        from mcp import Client, StdioServerParameters

        async def scenario():
            params = StdioServerParameters(command=sys.executable, args=["sqac_mcp.py"])
            async with Client(params) as c:
                names = _tool_names(await c.list_tools())
                for want in ("mem_bootstrap", "mem_checkpoint", "mem_sparsify",
                             "mem_observe", "mem_search", "mem_write"):
                    self.assertIn(want, names)

                r = await c.call_tool("mem_bootstrap", {"project": "acme", "host": "opencode", "model": "m-a"})
                pkt = json.loads(_text(r))
                self.assertEqual(pkt["project"], "acme")
                self.assertIn("no prior state", pkt["summary"])

                await c.call_tool("mem_observe", {"role": "user", "text": "migrate acme auth to OIDC"})
                await c.call_tool("mem_checkpoint", {
                    "project": "acme", "host": "opencode",
                    "goal": "migrate auth to OIDC",
                    "summary": "scope agreed, PKCE flow chosen",
                })

                r = await c.call_tool("mem_bootstrap", {"project": "acme", "host": "claude-code", "model": "m-b"})
                pkt2 = json.loads(_text(r))
                # packet reflects the state BEFORE this session engaged:
                # the prior worker (opencode) is what the resuming model must see
                self.assertEqual(pkt2["last_state"]["last_host"], "opencode")
                self.assertEqual(pkt2["last_state"]["goal"], "migrate auth to OIDC")
                self.assertEqual(len(pkt2["checkpoints"]), 1)
                self.assertGreaterEqual(len(pkt2["recent"]), 1)

                r = await c.call_tool("mem_sparsify", {})
                self.assertEqual(json.loads(_text(r))["demoted"], 0)  # nothing cold yet

                rt = await c.read_resource("memory://context")
                blob = rt.contents[0]
                text = blob.text if hasattr(blob, "text") else blob
                self.assertIn("acme", text if isinstance(text, str) else str(text))

        self._run(scenario())

    def test_shared_dir_across_two_server_instances(self):
        """Two server processes on the SAME memory dir see each other's data —
        the same as two different CLIs sharing ~/.sqacm."""
        from mcp import Client, StdioServerParameters

        async def scenario():
            p1 = StdioServerParameters(command=sys.executable, args=["sqac_mcp.py"])
            async with Client(p1) as c1:
                await c1.call_tool("mem_write", {
                    "content": "The deploy key lives in vault, path secret/deploy",
                    "key": "deploy-key-loc", "kind": "fact",
                })
            p2 = StdioServerParameters(command=sys.executable, args=["sqac_mcp.py"])
            async with Client(p2) as c2:
                r = await c2.call_tool("mem_search", {"query": "where is the deploy key"})
                hits = json.loads(_text(r))["hits"]
                self.assertTrue(any("vault" in h["content"] for h in hits),
                                "second server must see the first server's write")

        self._run(scenario())


if __name__ == "__main__":
    unittest.main()
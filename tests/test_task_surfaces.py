import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqac_mcp


def _text(r):
    return r.content[0].text if hasattr(r, "content") else r


class TestTaskResponseAttention(unittest.TestCase):
    """C4/S3: task tool responses carry the refreshed attention block."""

    def setUp(self):
        self._td = tempfile.mkdtemp()
        os.environ["SQAC_MEM_DIR"] = self._td
        os.environ["SQAC_SEMANTIC"] = "0"
        os.environ["SQAC_PROJECT"] = "projA"

    def tearDown(self):
        for k in ("SQAC_MEM_DIR", "SQAC_SEMANTIC", "SQAC_PROJECT"):
            os.environ.pop(k, None)
        shutil.rmtree(self._td, ignore_errors=True)

    def test_task_add_returns_attention(self):
        r = sqac_mcp.task_add(content="ship v2")
        pkt = json.loads(_text(r))
        self.assertTrue(pkt["ok"])
        self.assertEqual(pkt["status"], "pending")  # D: default is pending
        self.assertIn("attention", pkt)
        self.assertIn("ship v2", pkt["attention"])
        self.assertIn(pkt["mem_id"], pkt["attention"])

    def test_task_done_returns_attention_and_rejects_cross_project(self):
        r = sqac_mcp.task_add(content="task one", project="projA")
        mid = json.loads(_text(r))["mem_id"]
        # cross-project graduation is rejected
        os.environ["SQAC_PROJECT"] = "projB"
        r2 = sqac_mcp.task_done(mem_id=mid)
        pkt2 = json.loads(_text(r2))
        self.assertFalse(pkt2["found"])
        self.assertFalse(pkt2["graduated"])
        # same-project graduation succeeds and refreshes the block
        os.environ["SQAC_PROJECT"] = "projA"
        r3 = sqac_mcp.task_done(mem_id=mid)
        pkt3 = json.loads(_text(r3))
        self.assertTrue(pkt3["graduated"])
        self.assertIn("attention", pkt3)

    def test_task_list_filters_by_active_project(self):
        sqac_mcp.task_add(content="alpha task", project="projA")
        sqac_mcp.task_add(content="beta task", project="projB")
        os.environ["SQAC_PROJECT"] = "projA"
        tasks = json.loads(_text(sqac_mcp.task_list()))["tasks"]
        contents = [t["content"] for t in tasks]
        self.assertIn("alpha task", contents)
        self.assertNotIn("beta task", contents)


if __name__ == "__main__":
    unittest.main()
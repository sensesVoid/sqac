import os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqac.tasks import TaskManager
from sqac.attention import render_block_with_fallback
from sqac.continuity import ContinuityStore
from sqac.store import KIND_FACT, KIND_TASK
from sqac.offloader import ContextOffloader, Turn

class TestTasksIntegration(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.mkdtemp()
        os.environ["SQAC_MEM_DIR"] = self._td

    def tearDown(self):
        os.environ.pop("SQAC_MEM_DIR", None)
        shutil.rmtree(self._td, ignore_errors=True)

    def test_task_done_fact_exact_lookup(self):
        tm = TaskManager()
        r = tm.add("ship q4", status="current")
        mid = r["mem_id"]
        res = tm.done(mid)
        self.assertTrue(res["graduated"])
        e = tm._rack.get_by_mem_id(mid, cartridges=["facts"])
        self.assertIsNotNone(e)
        self.assertEqual(e[1]["kind"], KIND_FACT)
        self.assertEqual(e[1]["content"], "ship q4")

    def test_block_shows_explicit_tasks_over_goal(self):
        tm = TaskManager()
        tm.add("alpha task", status="current")
        with tempfile.TemporaryDirectory() as d:
            cs = ContinuityStore.load(os.path.join(d, "c.json"))
            cs.touch("p", goal="the big goal")
            cur = [(t["content"], t["mem_id"]) for t in tm.list(status="current")]
            out = render_block_with_fallback(5, cur, [], continuity=cs, project="p")
            self.assertIn("alpha task", out)
            self.assertNotIn("the big goal", out)

    def test_block_fallback_when_no_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            cs = ContinuityStore.load(os.path.join(d, "c.json"))
            cs.touch("p", goal="the big goal")
            off = ContextOffloader(path=None, window=8, semantic=False)
            off._buffer = [Turn(role="user", text="hello there")]
            out = render_block_with_fallback(3, [], [], continuity=cs,
                                             offloader=off, project="p")
            self.assertIn("the big goal", out)
            self.assertIn("hello there", out)
            self.assertIn("[SQAC attention", out)

if __name__ == "__main__":
    unittest.main()

import os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqac.tasks import TaskManager, project_name
from sqac.store import SqacStore


class TestTasks(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.mkdtemp()
        os.environ["SQAC_MEM_DIR"] = self._td

    def tearDown(self):
        for k in ("SQAC_MEM_DIR", "SQAC_PROJECT", "SQAC_NORMALIZE"):
            os.environ.pop(k, None)
        shutil.rmtree(self._td, ignore_errors=True)

    def test_add_list_status(self):
        tm = TaskManager()
        r1 = tm.add("ship v2", status="current")
        self.assertIn("mem_id", r1)
        r2 = tm.add("plan v3", status="pending")
        all_tasks = tm.list()
        self.assertGreaterEqual(len(all_tasks), 2)
        cur = tm.list(status="current")
        pend = tm.list(status="pending")
        self.assertIn("ship v2", [t["content"] for t in cur])
        self.assertIn("plan v3", [t["content"] for t in pend])

    def test_default_status_is_pending(self):
        tm = TaskManager()
        r = tm.add("something urgent")
        self.assertEqual(r["status"], "pending")
        tasks = tm.list(status="pending")
        self.assertTrue(any(t["content"] == "something urgent" for t in tasks))

    def test_done_cross_project_rejected(self):
        tm = TaskManager()
        r = tm.add("other project task", status="current", project="otherproj")
        res = tm.done(r["mem_id"])
        self.assertFalse(res["found"])
        self.assertFalse(res["graduated"])
        # task still alive in its project
        self.assertIsNotNone(tm._rack.get_by_mem_id(r["mem_id"], cartridges=["tasks"]))

    def test_done_graduates_preserving_id(self):
        tm = TaskManager()
        r = tm.add("finish q4 review", status="current", project=project_name())
        mid = r["mem_id"]
        # find task entry index in tasks store
        st = tm._store()
        tidx = next((i for i, e in enumerate(st._entries)
                     if not e.get("deleted") and e["meta"]["mem_id"] == mid), None)
        self.assertIsNotNone(tidx)
        res = tm.done(mid)
        self.assertTrue(res["found"])
        self.assertTrue(res["graduated"])
        # task deleted from tasks
        self.assertIsNone(tm._rack.get_by_mem_id(mid, cartridges=["tasks"]))
        # fact exists with same mem_id in facts
        found = tm._rack.get_by_mem_id(mid, cartridges=["facts"])
        self.assertIsNotNone(found)
        self.assertEqual(found[0], "facts")

    def test_done_not_found(self):
        tm = TaskManager()
        # 'facts' gets auto-created by add_with_mem_id on the done path
        res = tm.done("nonexistent-id")
        self.assertFalse(res["found"])
        self.assertFalse(res["graduated"])

    def test_done_same_project_graduates(self):
        tm = TaskManager()
        r = tm.add("local task", status="current", project=project_name())
        res = tm.done(r["mem_id"])
        self.assertTrue(res["found"])
        self.assertTrue(res["graduated"])

    def test_tasks_excluded_from_search(self):
        tm = TaskManager()
        tm.add("unique task word zebrazzalpha", status="current")
        # ordinary search on the tasks store excludes kind=task
        store = tm._store()
        hits = store.search("zebrazzalpha", top_k=10)
        self.assertEqual(hits, [])
        # explicit kind=task includes them
        hits2 = store.search("zebrazzalpha", top_k=10, kind="task")
        self.assertTrue(hits2)

    def test_project_filter_in_list(self):
        tm = TaskManager()
        tm.add("proj task", status="current", project="alpha")
        alpha = tm.list(project="alpha")
        self.assertTrue(any(t["content"] == "proj task" for t in alpha))

    def test_memid_unique(self):
        tm = TaskManager()
        a = tm.add("a", status="current")
        b = tm.add("b", status="current")
        self.assertNotEqual(a["mem_id"], b["mem_id"])

if __name__ == "__main__":
    unittest.main()

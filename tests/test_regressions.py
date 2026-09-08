"""Regression tests for bugs found in the audit-remediation pass.

Run: python -m pytest tests/test_regressions.py -q
"""

import os
import tempfile
import unittest

from sqac import SqacStore
from sqac.rack import CartridgeRack


class TestStoreSearchRegressions(unittest.TestCase):
    def setUp(self):
        self.store = SqacStore()

    def test_exact_hit_fills_remaining_top_k(self):
        """C1: an exact hit used to short-circuit and return 1 result even
        when top_k > 1. The fuzzy tier must still fill the window."""
        self.store.add("deploy to arm64 runners", key="deploy arm64", kind="fact")
        self.store.add("deploy pipeline runs on arm64", kind="fact")
        hits = self.store.search("deploy arm64", top_k=3)
        self.assertGreaterEqual(len(hits), 2)
        self.assertEqual(hits[0].mode, "exact")
        self.assertEqual(hits[1].mode, "fuzzy")

    def test_top_k_zero_returns_empty(self):
        """L1: top_k <= 0 silently returned 1 hit."""
        self.store.add("paris france", kind="fact")
        self.assertEqual(self.store.search("paris france", top_k=0), [])
        self.assertEqual(self.store.search("paris france", top_k=-3), [])

    def test_exact_hit_only_when_nothing_else_relates(self):
        """The exact-hit seeding must not invent unrelated hits."""
        self.store.add("alpha fact one", key="target", kind="skill")
        self.store.add("alpha fact two", key="other", kind="skill")
        hits = self.store.search("target", top_k=40)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].mode, "exact")


class TestStoreMutationRegressions(unittest.TestCase):
    def test_delete_keeps_live_sibling_exact_mapping(self):
        """M1: deleting one entry of an exact-key pair dropped the live
        sibling's mapping, so it fell back to fuzzy lookup."""
        store = SqacStore()
        i0 = store.add("first answer", key="shared key")
        i1 = store.add("second answer", key="shared key")
        store.delete(i0)
        hits = store.search("shared key")
        self.assertEqual(hits[0].mode, "exact")
        self.assertEqual(hits[0].content, "second answer")

    def test_add_snapshots_meta(self):
        """M5: add() stored the caller's meta dict by reference."""
        store = SqacStore()
        meta = {"grp": "x"}
        store.add("content", meta=meta, kind="fact")
        meta["grp"] = "MUTATED"
        self.assertEqual(store._entries[0]["meta"]["grp"], "x")

    def test_hit_trust_and_meta_are_copied(self):
        """M2: Hit.trust/meta aliased the live store entry dict."""
        store = SqacStore()
        store.add("some content", meta={"k": "v"}, kind="fact")
        hit = store.search("some content")[0]
        hit.trust["pwned"] = "yz"
        hit.meta["pwned"] = "zz"
        entry = store._entries[0]
        self.assertNotIn("pwned", entry["trust"])
        self.assertNotIn("pwned", entry["meta"])

    def test_as_dict_returns_copies(self):
        """M2: as_dict leaked live references into caller-owned data."""
        store = SqacStore()
        store.add("some content", meta={"k": "v"}, kind="fact")
        d = store.search("some content")[0].as_dict()
        d["trust"]["pwned"] = "yz"
        self.assertNotIn("pwned", store._entries[0]["trust"])


class TestRackSaveRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rack = CartridgeRack(directory=self.tmp.name, semantic=False)

    def test_save_none_persists_all(self):
        self.rack.create("one")
        self.rack["one"].add("content one", key="k1", kind="fact")
        self.rack.create("two")
        self.rack["two"].add("content two", key="k2", kind="fact")
        self.rack.save()
        names = self.rack.names()
        self.assertIn("one", names)
        self.assertIn("two", names)

    def test_save_empty_persists_nothing(self):
        self.rack.create("one")
        self.rack["one"].add("content one", key="k1", kind="fact")
        self.rack.save(names=[])
        loaded = SqacStore.load(os.path.join(self.tmp.name, "one.sqac"))
        self.assertEqual(len(loaded), 0)


class TestServerHelpers(unittest.TestCase):
    """Pure security helpers from sqac.server (no live HTTP server).

    The full server test (tests/test_server.py) is a known slow/hanging file
    and is excluded from the fast subset; these exercise the hardened bits.
    """

    @classmethod
    def setUpClass(cls):
        from sqac import server
        cls.server = server

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server._api_key = "test-key"
        self.server._state.clear()
        self.server._state["dir"] = __import__("pathlib").Path(self.tmp.name)

    def test_cartridge_path_rejects_traversal(self):
        p = self.server
        for bad in ("../evil", "a/b", "..", ".", "", "a\\b"):
            with self.assertRaises(Exception, msg=f"accepted {bad!r}"):
                p._cartridge_path(bad)
        ok = p._cartridge_path("memory")
        self.assertTrue(str(ok).startswith(self.tmp.name))
        self.assertTrue(str(ok).endswith("memory.sqac"))

    def test_verify_key_uses_compare_digest(self):
        # Passing key works, wrong key 401s.
        self.server._verify_key("test-key")
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            self.server._verify_key("wrong-key")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_teach_lock_dedup(self):
        p = self.server
        a = p._teach_lock_for("mem")
        b = p._teach_lock_for("mem")
        self.assertIs(a, b)
        self.assertIsNot(a, p._teach_lock_for("other"))

    def test_search_threshold_does_not_leak(self):
        """/search with a per-request threshold must restore the store's
        original fuzzy_threshold afterwards."""
        p = self.server
        store = __import__("sqac.store", fromlist=["SqacStore"]).SqacStore()
        store.add("alpha beta gamma delta", kind="fact")
        p._state["default_store"] = store
        original = store.fuzzy_threshold
        try:
            class R:
                query = "alpha beta gamma"
                top_k = 3
                kind = None
                threshold = 0.99
                cartridge = None
            p.search_endpoint(R())
            self.assertEqual(store.fuzzy_threshold, original)
        finally:
            store.fuzzy_threshold = original


if __name__ == "__main__":
    unittest.main()
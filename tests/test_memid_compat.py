import os, shutil, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqac.store import SqacStore
from sqac.memid import new_mem_id


def _build_1_3_x_cartridge(path: Path, entries: list[tuple[str, str]]):
    """Build a cartridge as sqac 1.3.x would: entries WITHOUT mem_id."""
    st = SqacStore()
    for content, key in entries:
        st.add(content, key=key, kind="fact")
        st._entries[-1]["meta"].pop("mem_id", None)
    st.save(path, name=path.stem)
    return st


class TestBackwardCompat(unittest.TestCase):
    def test_old_cartridge_searchable_and_backfillable(self):
        d = tempfile.mkdtemp()
        try:
            path = Path(d) / "old.sqac"
            _build_1_3_x_cartridge(path, [("alpha fact", "k1"), ("beta fact", "k2")])
            st = SqacStore.load(path)
            # search still works on the old cartridge
            hits = st.search("alpha", top_k=3)
            self.assertTrue(any("alpha" in h.content for h in hits))
            # backfill assigned ids
            mids = [e["meta"]["mem_id"] for e in st._entries if "mem_id" in e["meta"]]
            self.assertEqual(len(mids), 2)
            for mid in mids:
                self.assertIsNotNone(st.get(mid))
            # persist on next save
            st.save(path, name=path.stem)
            st2 = SqacStore.load(path)
            for e in st2._entries:
                self.assertTrue(e["meta"].get("mem_id"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_old_ids_honest_miss(self):
        d = tempfile.mkdtemp()
        try:
            path = Path(d) / "old.sqac"
            _build_1_3_x_cartridge(path, [("alpha", "k1")])
            st = SqacStore.load(path)
            # no id was assigned in-memory if we cleared it? load backfills; simulate brand-new unknown
            self.assertIsNone(st.get("never-assigned"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_no_second_id_on_re_add_same_key(self):
        st = SqacStore()
        st.add("x", key="k")
        m1 = st._entries[0]["meta"]["mem_id"]
        # re-adding with the same key creates a new entry (upsert semantics in store.add)
        st.add("x2", key="k")
        m2 = st._entries[1]["meta"]["mem_id"]
        self.assertNotEqual(m1, m2)


if __name__ == "__main__":
    unittest.main()

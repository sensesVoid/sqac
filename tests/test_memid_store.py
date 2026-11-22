import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqac.store import SqacStore
from sqac.memid import new_mem_id

class TestMemIdStore(unittest.TestCase):
    def test_add_assigns_mem_id(self):
        st = SqacStore()
        idx = st.add("hello world")
        e = st._entries[idx]
        self.assertIn("mem_id", e["meta"])
        mid = e["meta"]["mem_id"]
        self.assertEqual(len(mid), 26)
        self.assertEqual(st._id_index[mid], idx)
        self.assertIs(st.get(mid), e)

    def test_get_missing_returns_none(self):
        st = SqacStore()
        self.assertIsNone(st.get("NOPE"))
        self.assertIsNone(st.get(None))

    def test_delete_keeps_index_but_get_none(self):
        st = SqacStore()
        idx = st.add("fact")
        mid = st._entries[idx]["meta"]["mem_id"]
        st.delete(idx)
        self.assertIsNone(st.get(mid))
        # index still points
        self.assertEqual(st._id_index.get(mid), idx)

    def test_backfill_on_load(self):
        # build cartridge without mem_id by manually saving? Use store then strip? Instead test in-memory load via file
        d = tempfile.mkdtemp()
        path = os.path.join(d, "t.sqac")
        st = SqacStore()
        st.add("old entry")
        # strip mem_id to simulate old cartridge
        st._entries[0]["meta"].pop("mem_id", None)
        st.save(path, name="t", description="d")
        st2 = SqacStore.load(path)
        e = st2._entries[0]
        self.assertIn("mem_id", e["meta"])
        mid = e["meta"]["mem_id"]
        self.assertIsNotNone(st2.get(mid))
        # reload persists after save
        st2.save(path)
        st3 = SqacStore.load(path)
        self.assertEqual(st3._entries[0]["meta"]["mem_id"], mid)

    def test_compact_preserves_ids(self):
        st = SqacStore()
        i1 = st.add("a")
        i2 = st.add("b")
        mid1 = st._entries[i1]["meta"]["mem_id"]
        mid2 = st._entries[i2]["meta"]["mem_id"]
        st.delete(i1)
        st.compact()
        self.assertIsNone(st.get(mid1))
        self.assertIsNotNone(st.get(mid2))

    def test_unique_ids(self):
        st = SqacStore()
        ids = {st.add(f"x{i}") for i in range(5)}
        # ids not indices; check meta uniqueness
        mids = [st._entries[i]["meta"]["mem_id"] for i in range(5)]
        self.assertEqual(len(set(mids)), 5)



class TestMemIdRack(unittest.TestCase):
    def test_rack_compact_preserves_ids(self):
        from sqac.rack import CartridgeRack
        d = tempfile.mkdtemp()
        rack = CartridgeRack(directory=d, routes={"fact": "facts"}, default="facts", semantic=False, max_entries=10)
        rack.create("facts")
        ids = []
        for i in range(5):
            rack.write_routed(f"fact {i}", key=f"k{i}", kind="fact")
            st = rack._stores["facts"]
            ids.append(st._entries[-1]["meta"]["mem_id"])
        rack.compact("facts")
        st = rack._stores["facts"]
        for mid in ids:
            self.assertIsNotNone(st.get(mid))
        mids = [e["meta"]["mem_id"] for e in st._entries if not e.get("deleted")]
        self.assertEqual(len(mids), len(set(mids)))

    def test_rack_shard_split_preserves_ids(self):
        from sqac.rack import CartridgeRack
        d = tempfile.mkdtemp()
        rack = CartridgeRack(directory=d, routes={"fact": "facts"}, default="facts", semantic=False, max_entries=5)
        rack.create("facts")
        for i in range(7):
            rack.write_routed(f"fact {i}", key=f"k{i}", kind="fact")
        ids = []
        for name, st in rack._stores.items():
            for e in st._entries:
                if not e.get("deleted"):
                    ids.append(e["meta"]["mem_id"])
        self.assertEqual(len(ids), 7)
        self.assertEqual(len(set(ids)), 7)
        for name, st in rack._stores.items():
            for e in st._entries:
                if not e.get("deleted"):
                    self.assertIsNotNone(st.get(e["meta"]["mem_id"]))


if __name__ == "__main__":
    unittest.main()

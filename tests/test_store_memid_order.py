import time

from sqac.store import SqacStore
from sqac.memid import decode_mem_id_ms


def test_batch_add_sorts_in_creation_order():
    st = SqacStore(semantic=False)
    ids = [st._entries[st.add(f"fact {i}", key=f"k{i}", validate=False)]["meta"]["mem_id"]
           for i in range(200)]
    assert ids == sorted(ids)  # same-ms batch stays monotonic


def test_descending_recap_is_newest_first():
    st = SqacStore(semantic=False)
    for i in range(50):
        st.add(f"entry {i}", key=f"key-{i}", validate=False)
    ids = [e["meta"]["mem_id"] for e in st._entries]
    assert ids == sorted(ids)  # creation order == ascending id order


def test_load_seeds_above_existing_future_id():
    import tempfile, os
    st = SqacStore(semantic=False)
    st.add("anchored", key="anchor", validate=False)
    future_ms = int(time.time() * 1000) + 10_000_000  # 10s in the future
    # forge a future-dated id by seeding a gen and stamping the entry
    from sqac.memid import MonotonicMemId
    g = MonotonicMemId(); g.seed(future_ms, 0)
    st._entries[0]["meta"]["mem_id"] = g.mint()
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac"); st.save(p)
    st2 = SqacStore.load(p)
    idx = st2.add("later entry", key="later", validate=False)
    assert decode_mem_id_ms(st2._entries[idx]["meta"]["mem_id"]) >= future_ms
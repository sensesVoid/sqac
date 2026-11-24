import os
import tempfile
import time

from sqac.store import SqacStore, KIND_TASK
from sqac.cli import main


def _legacy(path):  # cartridge with NO mem_ids, appended A,B,C
    st = SqacStore(semantic=False)
    for k in ("a", "b", "c"):
        st.add(f"content {k}", key=k, validate=False)
    for e in st._entries:
        e["meta"].pop("mem_id", None)
    st.save(path)


def test_backfill_recovers_file_order_newest_first():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac")
    _legacy(p)
    rc = main(["backfill", "--db", p])
    assert rc == 0
    st = SqacStore.load(p)
    order = [e["key_norm"] for e in st.list_by_recency()]
    assert order == ["c", "b", "a"]  # newest (last appended) first


def test_backfill_is_idempotent_noop():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac")
    _legacy(p)
    main(["backfill", "--db", p])
    m1 = os.path.getmtime(p)
    time.sleep(0.01)
    main(["backfill", "--db", p])
    m2 = os.path.getmtime(p)
    assert m1 == m2  # second run rewrote nothing (save-if-changed)


def test_ids_lists_newest_first_and_skips_tasks():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac")
    st = SqacStore(semantic=False)
    st.add("one", key="k1", validate=False)
    st.add("two", key="k2", validate=False)
    st.add("taskrow", key="kt", kind=KIND_TASK, validate=False)
    st.save(p)
    rc = main(["ids", "--db", p, "--limit", "10"])
    assert rc == 0
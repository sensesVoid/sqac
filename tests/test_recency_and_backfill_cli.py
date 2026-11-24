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


def test_backfill_skips_non_memory_cartridges():
    import os
    import tempfile
    import time

    from sqac.cli import main
    from sqac.store import SqacStore

    d = tempfile.mkdtemp()
    p = os.path.join(d, "agent.sqac")
    # build a cartridge, stamp it as a non-memory type in the header extra
    st = SqacStore(semantic=False)
    st.add("agent memory", key="agentkey", validate=False)
    # strip mem_ids to simulate a legacy non-memory cartridge that WOULD be backfilled if not skipped
    for e in st._entries:
        e["meta"].pop("mem_id", None)
    st._header_extra = {"cartridge_type": "agent"}
    st.save(p)
    m1 = os.path.getmtime(p)
    time.sleep(0.01)
    rc = main(["backfill", "--db", p])  # --db targets this one file directly
    assert rc == 0          # run succeeded (did not crash on non-memory)
    assert os.path.getmtime(p) == m1  # file untouched — skipped, not rewritten


def test_backfill_skips_cartridges_under_skipped_segment():
    import os
    import tempfile
    import time

    from sqac.cli import main
    from sqac.store import SqacStore

    d = tempfile.mkdtemp()
    subdir = os.path.join(d, "archives")
    os.makedirs(subdir)
    p = os.path.join(subdir, "old.sqac")
    st = SqacStore(semantic=False)
    st.add("legacy archive entry", key="archkey", validate=False)
    for e in st._entries:
        e["meta"].pop("mem_id", None)
    st.save(p)
    m1 = os.path.getmtime(p)
    time.sleep(0.01)
    # directory backfill: no --db file present -> enumerate $SQAC_MEM_DIR (= d)
    prev = os.environ.get("SQAC_MEM_DIR")
    os.environ["SQAC_MEM_DIR"] = d
    try:
        rc = main(["backfill", "--db", os.path.join(d, "nonexistent.sqac")])  # not a file -> enumerate $SQAC_MEM_DIR
    finally:
        if prev is None:
            os.environ.pop("SQAC_MEM_DIR", None)
        else:
            os.environ["SQAC_MEM_DIR"] = prev
    assert rc == 0
    assert os.path.getmtime(p) == m1  # under archives/ -> skipped, not rewritten
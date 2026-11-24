import contextlib
import io
import json
import os
import tempfile
import time

from sqac.store import SqacStore, KIND_TASK
from sqac.cli import main
from sqac.format import read_cartridge


def _disk_mem_ids(path):
    """Read mem_ids straight off the cartridge FILE (not the in-memory store).

    Returns the mem_id of every live entry in file order, None where missing.
    This is the assertion that proves the backfill command actually persisted
    ids to disk rather than only minting them in memory.
    """
    cart = read_cartridge(path)
    out = []
    for e in cart.entries:
        if e.deleted:
            continue
        meta = e.payload.get("meta") if isinstance(e.payload, dict) else None
        out.append(meta.get("mem_id") if isinstance(meta, dict) else None)
    return out


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
    # sanity: the file genuinely has no ids before the run
    assert all(m is None for m in _disk_mem_ids(p))
    rc = main(["backfill", "--db", p])
    assert rc == 0
    # ids must be present ON DISK, not just in memory
    disk_ids = _disk_mem_ids(p)
    assert all(m is not None for m in disk_ids)
    st = SqacStore.load(p)
    order = [e["key_norm"] for e in st.list_by_recency()]
    assert order == ["c", "b", "a"]  # newest (last appended) first
    # a fresh load resolves the newest id to a real entry
    st2 = SqacStore.load(p)
    assert st2.get(_disk_mem_ids(p)[0]) is not None


def test_backfill_is_idempotent_noop():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac")
    _legacy(p)
    main(["backfill", "--db", p])
    # run 1 actually wrote ids to disk
    assert all(m is not None for m in _disk_mem_ids(p))
    m1 = os.path.getmtime(p)
    time.sleep(0.01)
    main(["backfill", "--db", p])
    m2 = os.path.getmtime(p)
    assert m1 == m2  # second run rewrote nothing (save-if-changed)


def test_backfill_bare_enumerates_memory_dir(monkeypatch):
    """Bare `sqac backfill` (no --db) enumerates SQAC_MEM_DIR and persists."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac")
    _legacy(p)
    monkeypatch.setenv("SQAC_MEM_DIR", d)
    rc = main(["backfill"])
    assert rc == 0
    # the cartridge under the memory dir got its ids written to disk
    assert all(m is not None for m in _disk_mem_ids(p))


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


def test_ids_lists_newest_first_and_skips_tasks():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.sqac")
    st = SqacStore(semantic=False)
    st.add("one", key="k1", validate=False)
    st.add("two", key="k2", validate=False)
    st.add("taskrow", key="kt", kind=KIND_TASK, validate=False)
    st.save(p)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["ids", "--db", p, "--limit", "10"])
    assert rc == 0

    rows = json.loads(buf.getvalue())
    keys = [r["key_norm"] for r in rows]

    # task entries are working memory — excluded from the listing
    assert "kt" not in keys
    # both normal entries are present
    assert "k1" in keys and "k2" in keys
    # newest-first by mem_id (descending): last-added (k2) appears first
    assert keys == ["k2", "k1"]
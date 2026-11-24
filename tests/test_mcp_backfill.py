import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import sqac_mcp  # noqa: F401
except Exception:  # pragma: no cover - import guarded below
    sqac_mcp = None

pytestmark = pytest.mark.skipif(sqac_mcp is None, reason="mcp/server deps unavailable")


def test_mem_backfill_tool(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SQAC_MEM_DIR", d)
    monkeypatch.setenv("SQAC_SEMANTIC", "0")
    import importlib

    sqac_mcp = importlib.reload(sys.modules["sqac_mcp"])

    # returns a JSON string, even on an empty rack
    out = sqac_mcp.mem_backfill()
    assert isinstance(out, str)
    packet = json.loads(out)
    assert isinstance(packet["minted"], int)
    assert isinstance(packet["cartridges"], list)


def test_mem_backfill_mints_for_legacy_entries(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SQAC_MEM_DIR", d)
    monkeypatch.setenv("SQAC_SEMANTIC", "0")

    from sqac.store import SqacStore
    from sqac.format import read_cartridge

    # Build a GENUINE legacy cartridge on disk: entries with NO mem_ids,
    # saved before the server ever mounts it. Placing it at the rack root
    # gives it the clean rack name "legacy". This file state actually occurs
    # in production (an old memory file), unlike popping ids off a live
    # in-memory store after load — which never happens.
    legacy_path = Path(d) / "legacy.sqac"
    legacy = SqacStore(semantic=False)
    legacy.add("Paris is the capital of France.", key="k1", validate=False)
    legacy.add("Berlin is the capital of Germany.", key="k2", validate=False)
    for e in legacy._entries:
        e["meta"].pop("mem_id", None)
    legacy.save(legacy_path)

    # prove the file really lacks ids before we start
    def _disk_ids(path):
        cart = read_cartridge(path)
        ids = []
        for e in cart.entries:
            if e.deleted:
                continue
            meta = e.payload.get("meta") if isinstance(e.payload, dict) else None
            ids.append(meta.get("mem_id") if isinstance(meta, dict) else None)
        return ids

    assert all(m is None for m in _disk_ids(legacy_path))

    import importlib

    # Server reload auto-mounts legacy.sqac; load-time backfill mints ids in
    # memory (so the server can hand them out) but the file still lacks them.
    sqac_mcp = importlib.reload(sys.modules["sqac_mcp"])

    out = sqac_mcp.mem_backfill()
    assert isinstance(out, str)
    packet = json.loads(out)
    assert isinstance(packet["minted"], int)
    assert isinstance(packet["cartridges"], list)
    assert packet["minted"] == 2
    assert "legacy" in packet["cartridges"]

    # PROOF: the ids now exist ON DISK, not merely in memory
    disk_ids = _disk_ids(legacy_path)
    assert all(m is not None for m in disk_ids)

    # A second run finds nothing left to mint (save-if-changed is idempotent).
    again = json.loads(sqac_mcp.mem_backfill())
    assert again["minted"] == 0
    assert again["cartridges"] == []


def test_mem_backfill_empty_rack(monkeypatch):
    """Empty rack: nothing to mint, clean JSON shape."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv("SQAC_MEM_DIR", d)
    monkeypatch.setenv("SQAC_SEMANTIC", "0")
    import importlib

    sqac_mcp = importlib.reload(sys.modules["sqac_mcp"])
    out = sqac_mcp.mem_backfill()
    assert isinstance(out, str)
    packet = json.loads(out)
    assert packet["minted"] == 0
    assert packet["cartridges"] == []
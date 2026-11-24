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
    import importlib

    sqac_mcp = importlib.reload(sys.modules["sqac_mcp"])

    sqac_mcp.mem_write(content="Paris is the capital of France.", kind="fact")
    sqac_mcp.mem_write(content="Berlin is the capital of Germany.", kind="fact")

    # Simulate legacy entries written before mem_ids existed: strip the
    # minted ids from the in-memory facts store so backfill has work to do.
    facts = sqac_mcp._state().rack["facts"]
    for e in facts._entries:
        meta = e.get("meta")
        if isinstance(meta, dict):
            meta.pop("mem_id", None)

    out = sqac_mcp.mem_backfill()
    assert isinstance(out, str)
    packet = json.loads(out)
    assert isinstance(packet["minted"], int)
    assert isinstance(packet["cartridges"], list)
    assert packet["minted"] == 2
    assert "facts" in packet["cartridges"]

    # A second run finds nothing left to mint (save-if-changed is idempotent).
    again = json.loads(sqac_mcp.mem_backfill())
    assert again["minted"] == 0
    assert again["cartridges"] == []
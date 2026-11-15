"""Hash-chained audit log for vault operations."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

AUDIT_PATH = Path.home() / ".sqacm" / "vault-audit.jsonl"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _prev_hash() -> str:
    if not AUDIT_PATH.exists():
        return ""
    try:
        lines = AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return ""
        last = json.loads(lines[-1])
        return last.get("prev", "")
    except Exception:
        return ""


def append_audit(
    vault: str = "default",
    name: str = "",
    op: str = "",
    success: bool = True,
    bytes_count: int = 0,
    host: str = "",
    session: str = "",
    pid: int = 0,
    error: str = "",
) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prev = _prev_hash()
    # Chain: prev = sha256 of previous record's canonical bytes
    if AUDIT_PATH.exists():
        try:
            lines = AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                prev = _sha256_bytes(_canonical(json.loads(lines[-1])))
        except Exception:
            prev = ""

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vault": vault,
        "name": name,
        "op": op,
        "host": host,
        "session": session,
        "pid": pid or os.getpid(),
        "success": success,
        "bytes": bytes_count,
        "error": error,
        "prev": prev,
    }
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_audit(limit: Optional[int] = None, tail: Optional[int] = None) -> list[dict]:
    if not AUDIT_PATH.exists():
        return []
    lines = AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if tail is not None:
        records = records[-tail:]
    if limit is not None:
        records = records[:limit]
    return records


def verify_audit_chain() -> bool:
    if not AUDIT_PATH.exists():
        return True
    lines = AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return True
    prev = ""
    for line in lines:
        rec = json.loads(line)
        if rec.get("prev", "") != prev:
            return False
        prev = _sha256_bytes(_canonical({k: v for k, v in rec.items() if k != "prev"}))
    return True

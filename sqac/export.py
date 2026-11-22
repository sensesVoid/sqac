"""Human-readable export and diff for .sqac cartridges.

`sqac export` and `sqac diff` make memory changes reviewable in Git PRs,
auditable across sessions, and portable to Markdown/JSON backups.

Deliberately out of scope: encrypted cartridges (SQAE/SQSE) refuse to
export — their payloads never leave the vault/in-process APIs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .format import read_cartridge
from .store import KIND_NAMES, SqacStore

_ENCRYPTED_MAGICS = (b"SQAE", b"SQSE")


class ExportError(Exception):
    """Raised when a cartridge cannot be exported or diffed."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_plaintext(path: Path) -> None:
    if not path.exists():
        raise ExportError(f"cartridge not found: {path}")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic in _ENCRYPTED_MAGICS:
        raise ExportError(
            f"refusing to export encrypted cartridge: {path} — "
            "vault/secret payloads stay in-process"
        )
    try:
        read_cartridge(str(path))
    except Exception as exc:
        raise ExportError(f"not a SQAC cartridge: {path} ({exc})")


def _entries(store: SqacStore) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in store._entries:
        if e.get("deleted"):
            continue
        out.append({
            "content": e.get("content", ""),
            "kind": KIND_NAMES.get(e.get("kind", 0), "generic"),
            "source": e.get("source", ""),
            "key_norm": e.get("key_norm", ""),
            "meta": dict(e.get("meta") or {}),
        })
    return out


def export_cartridge(path: str | Path, fmt: str = "md") -> str:
    """Serialize a plaintext cartridge to Markdown or JSON."""
    p = Path(path)
    _check_plaintext(p)
    if fmt not in ("md", "json"):
        raise ExportError(f"unsupported format: {fmt!r} (want 'md' or 'json')")
    store = SqacStore.load(p)
    entries = _entries(store)
    ctype = store.cartridge_type

    if fmt == "json":
        return json.dumps(
            {
                "cartridge": p.name,
                "name": p.stem,
                "cartridge_type": ctype,
                "entries": entries,
                "entry_count": len(entries),
                "exported": _now_iso(),
                "format": "sqac-export/1",
            },
            indent=2,
            ensure_ascii=False,
        )

    lines = [
        f"# SQAC Export — {p.stem}",
        "",
        f"- cartridge: `{p.name}`",
        f"- type: `{ctype}` · entries: {len(entries)} · exported: {_now_iso()}",
        "",
    ]
    for i, e in enumerate(entries, 1):
        key_bit = f" · key `{e['key_norm']}`" if e["key_norm"] else ""
        lines.append(f"## {i}. [kind={e['kind']} · source={e['source']}{key_bit}]")
        lines.append("")
        lines.append(e["content"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _index(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """key_norm -> entry; entries without a key fall back to content hash."""
    idx: dict[str, dict[str, Any]] = {}
    for e in entries:
        key = e["key_norm"] or hash(e["content"])
        if key not in idx:
            idx[key] = e
    return idx


def diff_cartridges(old_path: str | Path, new_path: str | Path, fmt: str = "md") -> str:
    """Diff two plaintext cartridges: added / removed / updated / reclassified."""
    old_p, new_p = Path(old_path), Path(new_path)
    for p in (old_p, new_p):
        _check_plaintext(p)
    if fmt not in ("md", "json"):
        raise ExportError(f"unsupported format: {fmt!r} (want 'md' or 'json')")

    old_entries = _entries(SqacStore.load(old_p))
    new_entries = _entries(SqacStore.load(new_p))
    old_map, new_map = _index(old_entries), _index(new_entries)

    added = [new_map[k] for k in new_map if k not in old_map]
    removed = [old_map[k] for k in old_map if k not in new_map]
    updated = [
        {"old": old_map[k], "new": new_map[k]}
        for k in new_map
        if k in old_map and new_map[k]["content"] != old_map[k]["content"]
    ]
    reclassified = [
        {"old": old_map[k], "new": new_map[k]}
        for k in new_map
        if k in old_map and new_map[k]["kind"] != old_map[k]["kind"]
    ]

    report: dict[str, Any] = {
        "old": str(old_p),
        "new": str(new_p),
        "added": added,
        "removed": removed,
        "updated": updated,
        "reclassified": reclassified,
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "updated": len(updated),
            "reclassified": len(reclassified),
        },
        "generated": _now_iso(),
        "format": "sqac-diff/1",
    }

    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False)

    lines = [
        f"# SQAC Diff — {old_p.name} → {new_p.name}",
        "",
        f"- {report['summary']['added']} added · "
        f"{report['summary']['removed']} removed · "
        f"{report['summary']['updated']} updated · "
        f"{report['summary']['reclassified']} reclassified",
        "",
    ]

    def _section(title: str, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        lines.append(f"## {title} ({len(items)})")
        lines.append("")
        for it in items:
            lines.append(f"- [{it['kind']}] {it['content']}")
        lines.append("")

    _section("Added", added)
    _section("Removed", removed)
    for u in updated:
        lines.append(f"## Updated (kind={u['new']['kind']}, key={u['new']['key_norm'] or 'content'})")
        lines.append(f"- old: {u['old']['content']}")
        lines.append(f"- new: {u['new']['content']}")
        lines.append("")
    for r in reclassified:
        lines.append(f"## Reclassified ({r['old']['kind']} → {r['new']['kind']})")
        lines.append(f"- {r['new']['content']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

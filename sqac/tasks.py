"""SQAC task state: per-project current/pending tasks.

v1.4.0 (deliverable proposal "HTTP parity & Client Platform" follow-ups).

Tasks live in a per-memory-dir `tasks.sqac` cartridge (kind=task). They are
excluded from ordinary search by default (Q1): `task_list` and the attention
block are their surface. `task_done` graduates the task into a durable
`kind=fact` memory, preserving the same mem_id so the agent keeps a precise
handle for exact lookup.

Per-project: tasks stamp the project via the same derivation as mem_checkpoint;
task_done rejects cross-project ids.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .rack import CartridgeRack
from .store import KIND_NAMES, KIND_TASK, SqacStore

TASKS_ROUTES = {"task": "tasks"}
STATUSES = ("current", "pending")


def tasks_dir() -> Path:
    """Memory dir for task state (SQAC_MEM_DIR-aware)."""
    if os.environ.get("SQAC_MEM_DIR"):
        return Path(os.environ["SQAC_MEM_DIR"]).expanduser()
    return Path.home() / ".sqacm"


def _status(e: dict) -> str:
    return str(e.get("meta", {}).get("status", "pending"))


def _project(e: dict) -> str:
    return str(e.get("meta", {}).get("project", "default"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def project_name(project: Optional[str] = None) -> str:
    """Active project key: explicit arg > SQAC_PROJECT > cwd basename > 'default'.

    (MCP's _project_name is the same derivation; this copy lets the tasks
    module be used without importing sqac_mcp — deliberate layering.)
    """
    if project and project.strip():
        return project.strip()
    env = os.environ.get("SQAC_PROJECT")
    if env and env.strip():
        return env.strip()
    try:
        return Path.cwd().resolve().name
    except Exception:
        return "default"


class TaskManager:
    """Manages task entries in the rack-mounted `tasks` cartridge."""

    def __init__(self, rack: Optional[CartridgeRack] = None):
        if rack is None:
            rack = CartridgeRack(
                directory=tasks_dir(), routes=dict(TASKS_ROUTES), default="tasks",
                auto_load=True,
            )
        self._rack = rack

    def _store(self) -> SqacStore:
        try:
            return self._rack._require("tasks")
        except Exception:
            pass
        if "tasks" not in self._rack.names():
            self._rack.create("tasks")
        return self._rack._require("tasks")

    def add(self, content: str, status: str = "current", project: Optional[str] = None,
            mem_id: Optional[str] = None, meta: Optional[dict] = None) -> dict[str, Any]:
        """Add a task entry. Returns {mem_id, content, status, project}."""
        if status not in STATUSES:
            status = "current"
        proj = project_name(project) or "default"
        m = dict(meta or {})
        m.update({"status": status, "project": proj, "updated": _now()})
        if "created" not in m:
            m["created"] = _now()
        if mem_id:
            m["mem_id"] = mem_id
        st = self._store()
        idx = st.add_with_mem_id(content, mem_id=mem_id or "", source="tasks", kind="task",
                                 meta=m)
        e = st._entries[idx] if idx is not None and idx < len(st._entries) else st._entries[-1]
        self._rack.save()
        return {"mem_id": e["meta"]["mem_id"], "content": e["content"],
                "status": e["meta"]["status"], "project": e["meta"]["project"]}

    def list(self, status: Optional[str] = None, project: Optional[str] = None,
             n: int = 25, sort: bool = True) -> list[dict[str, Any]]:
        """List task entries.

        status: "current", "pending", or None for all.
        project: filter by project (None = all projects).
        sort: True = current first, then pending (updated-desc).
        """
        st = self._store()
        out = []
        for e in st._entries:
            if e.get("deleted"):
                continue
            if status is not None and _status(e) != status:
                continue
            if project is not None and _project(e) != project:
                continue
            out.append({"mem_id": e["meta"]["mem_id"], "content": e["content"],
                        "status": _status(e), "project": _project(e),
                        "updated": e["meta"].get("updated", "")})
        if sort:
            cur = sorted([d for d in out if d["status"] == "current"], key=lambda d: d["updated"], reverse=True)
            pnd = sorted([d for d in out if d["status"] == "pending"], key=lambda d: d["updated"], reverse=True)
            out = cur + pnd
        return out[:n]

    def done(self, mem_id: str, to_fact: bool = True) -> dict[str, Any]:
        """Graduate a task to a durable kind=fact memory (same mem_id),
        tombstoning the task entry. Returns {found, graduated, mem_id, content}."""
        res0 = self._rack.get_by_mem_id(mem_id, cartridges=["tasks"])
        st = self._store()
        idx = st.delete_entry_by_mem_id(mem_id)
        if res0 is None and idx is None:
            return {"found": False, "graduated": False, "mem_id": mem_id, "content": ""}
        content = (st._entries[idx]["content"] if idx is not None
                   and idx < len(st._entries)
                   and not st._entries[idx].get("deleted")
                   else (res0[1]["content"] if res0 else ""))
        graduated = False
        fact_mid = ""
        if to_fact:
            try:
                self._rack.add_with_mem_id(
                    "facts", content,
                    key=(st._entries[idx].get("key_norm") if idx is not None and idx < len(st._entries)
                         and not st._entries[idx].get("deleted") else content),
                    source="tasks", kind="fact",
                    meta={"status": "done",
                          "project": _project(st._entries[idx]) if (idx is not None and idx < len(st._entries)
                                                                   and not st._entries[idx].get("deleted")) else "default",
                          "from_task": True},
                    mem_id=mem_id)
                graduated = True
                fact_mid = mem_id
            except Exception:
                pass
        # ensure tasks store has the entry marked deleted; compact for tidiness
        if idx is not None and idx < len(st._entries):
            st.delete(idx)
        st.compact()
        self._rack.save()
        facts_st = self._rack._stores.get("facts")
        found = (facts_st is not None and bool(fact_mid and facts_st.get(fact_mid))) or bool(res0)
        return {"found": found, "graduated": graduated, "mem_id": fact_mid or mem_id,
                "content": content}

    def update(self, mem_id: str, status: Optional[str] = None,
               content: Optional[str] = None) -> dict[str, Any]:
        st = self._store()
        idx = next((i for i, x in enumerate(st._entries)
                    if x["meta"].get("mem_id") == mem_id and not x.get("deleted")), None)
        if idx is None:
            return {"found": False, "updated": False}
        if status:
            st._entries[idx]["meta"]["status"] = status
        if content is not None:
            st._entries[idx]["content"] = content
        st._entries[idx]["meta"]["updated"] = _now()
        self._rack.save()
        return {"found": True, "updated": True}

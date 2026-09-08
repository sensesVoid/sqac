"""Cross-CLI continuity — hand task state between agent harnesses.

The problem this solves: sessions are ephemeral per CLI. If a user drives a
task in opencode, then switches to Claude Code, the fresh model has no idea
what was just decided. SQAC's cartridges already survive process restarts;
this module adds the *handoff* state on top — which host last worked on a
project, the active goal, the last summary, and a small checkpoint trail — so
any new model on any CLI can call bootstrap and be "in the loop" immediately.

The state lives in ``continuity.json`` inside the shared memory directory
(e.g. ``~/.sqacm/continuity.json``). All CLIs point at the same memory dir, so
switching hosts is seamless: one file, shared by every harness/MCP client.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

CONTINUITY_VERSION = 1
DEFAULT_PROJECT = "default"
_CHECKPOINT_CAP = 10
_AGE_CUTOFF = 60 * 60 * 24 * 30  # 30 days before a stale project is pruned

# Fingerprints read from process ancestry / env to identify the calling CLI.
# Order matters: first match wins (most specific first).
_HOST_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("opencode", ("opencode",)),
    ("claude-code", ("claude-code", "claude_code", "claude", "claude-desktop")),
    ("codex", ("codex-cli", "codex")),
    ("cursor", ("cursor",)),
    ("zed", ("zed",)),
    ("cody", ("cody",)),
    ("gh-copilot", ("gh-copilot", "copilot")),
    ("cinnamon", ("cinnamon",)),
    ("continue", ("continue",)),
    ("mcp-inspector", ("mcp-inspector", "inspector")),
]


def _cmdline(pid: int) -> str:
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").lower()
    except (OSError, IOError, ValueError):
        return ""


def detect_host(override: Optional[str] = None) -> str:
    """Best-effort detection of the host CLI, walking the process ancestry.

    ``override`` wins when supplied (callers may pass it explicitly). Falls
    back to env fingerprints, then a walk of up to 4 ancestors' cmdlines via
    /proc (Linux). Returns "unknown" when nothing matches — never raises.
    """
    if override:
        return override
    for hint, needles in _HOST_HINTS:
        for key in (
            f"SQAC_HOST_{hint.upper().replace('-', '_')}",
            f"{hint.upper().replace('-', '_')}_PROJECT_DIR",
        ):
            if os.environ.get(key):
                return hint
    pid = os.getpid()
    for _ in range(5):
        cmd = _cmdline(pid)
        for hint, needles in _HOST_HINTS:
            if any(n in cmd for n in needles):
                return hint
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text(errors="replace")
            fields = stat.rsplit(")", 1)
            pid = int(fields[1].split()[0]) if len(fields) == 2 else 0
        except (OSError, IndexError, ValueError):
            break
        if pid <= 1:
            break
    return "unknown"


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


class ContinuityStore:
    """Persistent, per-project handoff state. One JSON sidecar per memory dir."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {"version": CONTINUITY_VERSION, "projects": {}}

    # ── io ──────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path) -> "ContinuityStore":
        st = cls(path)
        if st.path.exists():
            try:
                raw = json.loads(st.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("projects"), dict):
                    st._data = raw
            except (json.JSONDecodeError, OSError):
                pass  # corrupt sidecar: start clean, don't crash the session
        st._data.setdefault("version", CONTINUITY_VERSION)
        st._data.setdefault("projects", {})
        return st

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    # ── project records ────────────────────────────────────────────────
    def _record(self, project: str) -> dict[str, Any]:
        rec = self._data["projects"].get(project)
        if rec is None:
            rec = {
                "last_host": None,
                "last_model": None,
                "last_activity": None,
                "goal": None,
                "last_summary": None,
                "checkpoints": [],
            }
            self._data["projects"][project] = rec
        rec.setdefault("checkpoints", [])
        return rec

    def projects(self) -> list[str]:
        return [p for p, r in self._data["projects"].items() if not self._stale(r)]

    def _stale(self, rec: dict[str, Any]) -> bool:
        ta = rec.get("last_activity")
        return bool(ta) and (_now() - float(ta) > _AGE_CUTOFF)

    def touch(
        self,
        project: str,
        host: Optional[str] = None,
        model: Optional[str] = None,
        goal: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record that a host just engaged with a project (bootstrap/observe)."""
        rec = self._record(project)
        rec["last_activity"] = _now()
        if host:
            rec["last_host"] = host
        if model:
            rec["last_model"] = model
        if goal is not None:
            rec["goal"] = goal
        if summary is not None:
            rec["last_summary"] = summary
        return rec

    def checkpoint(
        self,
        project: str,
        host: Optional[str] = None,
        goal: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> dict[str, Any]:
        """Persist a task-boundary handoff and keep the checkpoint trail."""
        rec = self.touch(project, host=host, goal=goal, summary=summary)
        cps = rec["checkpoints"]
        cps.append(
            {
                "ts": _iso(_now()),
                "host": host or rec.get("last_host"),
                "goal": goal if goal is not None else rec.get("goal"),
                "summary": summary if summary is not None else rec.get("last_summary"),
            }
        )
        del cps[: -_CHECKPOINT_CAP]
        return rec

    def record(self, project: str) -> dict[str, Any]:
        return {
            k: v for k, v in self._record(project).items() if k != "checkpoints"
        }

    def last_checkpoints(self, project: str, n: int = 3) -> list[dict[str, Any]]:
        cps = self._record(project).get("checkpoints", [])
        return cps[-n:]


def _describe(rec: dict[str, Any]) -> str:
    """One or two sentences stating where the project was left off."""
    parts = []
    host = rec.get("last_host")
    ts = rec.get("last_activity")
    when = _iso(float(ts)) if ts else None
    goal = rec.get("goal")
    summary = rec.get("last_summary")
    if when:
        parts.append(f"last active on {host or 'unknown host'} at {when}")
    elif host:
        parts.append(f"last active on {host}")
    if goal:
        parts.append(f"active goal: {goal}")
    if summary:
        parts.append(summary)
    return " ".join(parts) if parts else f"no prior state recorded for this project."


def bootstrap_packet(
    project: str,
    continuity: ContinuityStore,
    host: Optional[str] = None,
    model: Optional[str] = None,
    session_stats: Optional[dict[str, Any]] = None,
    rack_stats: Optional[dict[str, Any]] = None,
    recent: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """The "get to work" packet a fresh model needs to be in the loop.

    Everything is best-effort: missing inputs degrade gracefully. Never
    fabricates — a project with no prior state gets an honest "no prior
    state", and `recent` is only included when the caller supplies it.
    """
    rec = continuity.record(project)
    cps = continuity.last_checkpoints(project, n=3)
    mem_entries = 0
    if rack_stats:
        mem_entries = sum(s.get("entries", 0) for s in rack_stats.values())
    return {
        "project": project,
        "host": host,
        "model": model,
        "last_state": rec,
        "checkpoints": cps,
        "summary": _describe(rec),
        "session": session_stats or {},
        "memory_entries": mem_entries,
        "cartridges": sorted(rack_stats.keys()) if rack_stats else [],
        "recent": recent or [],
        "continuity_note": (
            f"You are continuing work on '{project}'. {_describe(rec)} "
            "Use mem_search / mem_recall for details, mem_observe to record "
            "new turns, and mem_checkpoint at task boundaries so the next "
            "session (any CLI) picks up exactly here. If memory is empty, say "
            "so rather than inventing context."
        ),
    }


_SERVER_INSTRUCTIONS = (
    "SQAC gives you persistent, shared memory that survives across sessions and CLI harnesses "
    "(opencode, Claude Code, Codex, Cursor, Claude Desktop, ...).\n\n"
    "MEMORY PROTOCOL:\n"
    "1. At the START of a session, call mem_bootstrap once to load your working context "
    "(project, active goal, last summary, which host worked last). This puts you in the loop "
    "even after a CLI switch.\n"
    "2. Record meaningful turns with mem_observe(role, text) so the running conversation stays "
    "captured.\n"
    "3. At every task boundary (goal reached, blocker hit, context switch), call "
    "mem_checkpoint(goal=..., summary=...) so any later session — same CLI or a different one — "
    "resumes from exactly this point.\n"
    "4. Search durable knowledge with mem_search; recall recent work with mem_recall. "
    "mem_graduate promotes stable session facts into long-term cartridges; mem_sparsify keeps "
    "the working memory sparse.\n"
    "5. Recall is fail-safe: it returns empty rather than guessing. If memory has nothing on a "
    "topic, say \"I don't have that in memory\" — never fabricate.\n\n"
    "CODE STRUCTURE (AST) TOOLS:\n"
    "Use these when you need to understand code structure, find callers/callees, or assess "
    "the impact of a change:\n"
    "- ast_init: build the structural index for a project (run once, auto-cached)\n"
    "- ast_explore(name): get a symbol's definition, docstring, callers, and callees\n"
    "- ast_blast(name): find all symbols affected by changing this symbol (upstream callers)\n"
    "- ast_callers(name): who calls this function/method?\n"
    "- ast_callees(name): what does this function/method call?\n"
    "- ast_search(query): fuzzy search across all symbols by name or docstring\n"
    "- ast_stats: symbol/edge/language counts\n"
    "Use ast_init first if the graph is empty or stale. The tools auto-load cached graphs."
)


def server_instructions() -> str:
    return _SERVER_INSTRUCTIONS
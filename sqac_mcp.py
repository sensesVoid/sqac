#!/usr/bin/env python3
"""
MCP server exposing the SQAC store as native tools (mem_search / mem_write /
mem_swap and friends) for Claude Desktop, Cursor, Zed, and any MCP client.

The demo promise: teach a fact -> quit -> reopen -> still knows.

Config (env vars, optional):
  SQAC_MEM_DIR   directory that holds the cartridge files (default ~/.sqacm)
  SQAC_SEMANTIC  1 to enable the static-embedding semantic tier (default),
                 0 for lexical+exact only (faster, no model load)

Run:
  python sqac_mcp.py                 # stdio transport
"""

from __future__ import annotations

import functools
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import Field

from mcp.server.mcpserver import MCPServer

from sqac.continuity import (
    ContinuityStore,
    bootstrap_packet,
    detect_host,
    server_instructions,
)
from sqac.dms import DMS
from sqac.rack import CartridgeRack
from sqac.offloader import ContextOffloader
from sqac.graph_mcp import register_graph_tools

SERVER_NAME = "sqac_mcp"
DEFAULT_ROUTES = {"fact": "facts", "doc": "docs", "skill": "skills"}
DEFAULT_CARTRIDGES = ("facts", "docs", "skills")
VALID_KINDS = ("generic", "fact", "skill", "doc", "turn")
SESSION_BUDGET = 25_000  # DMS budget: capacity sweep sweet spot

mcp = MCPServer(
    name=SERVER_NAME,
    title="SQAC Memory Server",
    description=(
        "SQAC: Symbolic Query Addressable Cartridge. Persistent memory + "
        "structural code intelligence for LLMs. Memory tools: mem_bootstrap "
        "(cross-CLI context), mem_search (fuzzy/semantic recall), mem_write "
        "(store facts/skills), mem_checkpoint (task handoff), mem_sparsify "
        "(DMS eviction). Code structure tools: ast_init (build index), "
        "ast_explore (symbol definition + callers + callees), ast_blast "
        "(blast radius — what breaks if I change this), ast_callers, "
        "ast_callees, ast_search (fuzzy symbol search), ast_stats."
    ),
    instructions=server_instructions(),
    version="0.1.0",
)

# Register graph tools (code intelligence)
register_graph_tools(mcp)


@dataclass
class _Config:
    dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SQAC_MEM_DIR", str(Path.home() / ".sqacm"))
        ).expanduser()
    )
    semantic: bool = field(default_factory=lambda: os.environ.get("SQAC_SEMANTIC", "1") != "0")


# ── state ────────────────────────────────────────────────────────────────────

class _MemoryState:
    def __init__(self, cfg: _Config):
        self.cfg = cfg
        cfg.dir.mkdir(parents=True, exist_ok=True)
        self.rack = CartridgeRack(
            directory=cfg.dir,
            routes=dict(DEFAULT_ROUTES),
            default="facts",
            semantic=cfg.semantic,
            min_confidence=0.60,
            auto_load=True,  # mounts existing *.sqac (never the session bucket)
            max_entries=25_000,  # auto-split at capacity benchmark sweet spot
            folder_routing=True,  # facts/, skills/, docs/ subdirectories
        )
        # Ensure the default cartridge set exists. create() is safe: existing
        # files are loaded, never wiped.
        for name in DEFAULT_CARTRIDGES:
            if name not in self.rack:
                self.rack.create(name, description=f"{name} cartridge (auto-created)")
        self.session = ContextOffloader(
            cfg.dir / "session.sqac", window=8, semantic=cfg.semantic,
            dms=DMS(budget=SESSION_BUDGET),
        )
        self.continuity = ContinuityStore(cfg.dir / "continuity.json")

    def save(self) -> None:
        self.rack.save()
        self.session.save()
        self.continuity.save()


_STATE_LOCK = threading.RLock()
_STATE: Optional[_MemoryState] = None


def _state() -> _MemoryState:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = _MemoryState(_Config())
    return _STATE


def _serialized(fn):
    """Serialize tool execution. MCP v2 runs sync handlers on worker
    threads; _STATE_LOCK (an RLock) keeps search/recall/graduate consistent
    without blocking lazy state init (_state re-enters the same lock)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _STATE_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def _err(exc: Exception) -> str:
    return f"Error: {type(exc).__name__}: {exc}"


def _hit_json(h) -> dict[str, Any]:
    return {
        "content": h.content,
        "confidence": h.confidence,
        "source": h.source,
        "mode": h.mode,
        "cartridge": h.meta.get("cartridge"),
        "latency_ms": h.meta.get("latency_ms"),
        "kind": h.meta.get("kind"),
    }


def _project_name(project: Optional[str]) -> str:
    """Resolve the project key: explicit arg > env > cwd basename > default."""
    if project and project.strip():
        return project.strip()
    env = os.environ.get("SQAC_PROJECT")
    if env and env.strip():
        return env.strip()
    try:
        return os.getcwd().split("/")[-1] or "default"
    except OSError:
        return "default"


def _recent_exchanges(st, project: str, n: int = 5) -> list[dict[str, Any]]:
    """Surface the most relevant exchanges for the active goal/project.

    Fail-safe: empty list when nothing recalls (never fabricate)."""
    try:
        query = (
            st.continuity.record(project).get("goal")
            or st.continuity.record(project).get("last_summary")
            or project
        )
        return st.session.recall_detailed(query, top_k=n)
    except Exception:
        return []


# ── read tools ───────────────────────────────────────────────────────────────

@mcp.tool(
    name="mem_bootstrap",
    annotations={
        "title": "Load Working Context (cross-CLI)",
        "read_only_hint": True,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": True,
    },
)
@_serialized
def mem_bootstrap(
    project: Annotated[Optional[str], Field(description="Project key; inferred from cwd if omitted")] = None,
    host: Annotated[Optional[str], Field(description="Host CLI name override (auto-detected by default)")] = None,
    model: Annotated[Optional[str], Field(description="Model id to record as last active")] = None,
) -> str:
    """Load your working context at session start — puts any model in the loop.

    Returns the project's continuity packet: last active host/model/time, the
    active goal, the last summary, recent checkpoints, session stats, and the
    most relevant recent exchanges. Also records that THIS host just engaged,
    so a later session on a different CLI resumes seamlessly.

    Call this ONCE when a session starts, especially after a CLI switch.

    Returns:
        JSON packet. A project with no prior state reports it honestly
        ("no prior state recorded") instead of fabricating context.
    """
    try:
        st = _state()
        proj = _project_name(project)
        host_name = detect_host(host)
        # Packet first: it should describe where the work was left off BEFORE
        # this session engaged (so a model resuming after a CLI switch sees
        # the PREVIOUS host + goal). Then record this engagement for the next
        # session to find.
        packet = bootstrap_packet(
            project=proj,
            continuity=st.continuity,
            host=host_name,
            model=model,
            session_stats=st.session.stats(),
            rack_stats=st.rack.stats(),
            recent=_recent_exchanges(st, proj),
        )
        st.continuity.touch(proj, host=host_name, model=model)
        st.continuity.save()
        return _ok(packet)
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_checkpoint",
    annotations={
        "title": "Save a Task Handoff",
        "read_only_hint": False,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_checkpoint(
    project: Annotated[Optional[str], Field(description="Project key; inferred from cwd if omitted")] = None,
    goal: Annotated[Optional[str], Field(description="The active goal (carried into the next session)")] = None,
    summary: Annotated[Optional[str], Field(description="What was accomplished / where it left off")] = None,
    host: Annotated[Optional[str], Field(description="Host CLI name override (auto-detected by default)")] = None,
) -> str:
    """Persist a task-boundary handoff so any future session (any CLI) resumes here.

    Records the active goal, a human-readable summary, and the host/time into
    the shared continuity state, plus appends to the checkpoint trail. Call
    this when a task completes, hits a blocker, or you switch contexts.

    Returns:
        JSON: {"project", "saved", "goal", "summary", "last_host"}.
    """
    try:
        st = _state()
        proj = _project_name(project)
        host_name = detect_host(host)
        rec = st.continuity.checkpoint(proj, host=host_name, goal=goal, summary=summary)
        st.continuity.save()
        return _ok(
            {
                "project": proj,
                "saved": True,
                "goal": rec.get("goal"),
                "summary": rec.get("last_summary"),
                "last_host": rec.get("last_host"),
                "checkpoints": len(rec.get("checkpoints", [])),
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_sparsify",
    annotations={
        "title": "Prune Working Memory (DMS)",
        "read_only_hint": False,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_sparsify(
    top_k: Annotated[Optional[int], Field(ge=1, le=100000, description="Limit demotions to the N lowest-utility exchanges (default: all eligible)")] = None,
) -> str:
    """Run Dynamic Memory Sparsification: demote cold, low-utility session
    exchanges into durable facts so the working memory stays sparse and scans
    stay fast.

    No-op when the session has no DMS policy configured.

    Returns:
        JSON: {"demoted", "session_entries", "kinds"}.
    """
    try:
        st = _state()
        demoted = st.session.sparsify(top_k=top_k)
        st.session.save()
        return _ok(
            {
                "demoted": demoted,
                "session_entries": st.session.stats().get("entries", 0),
                "kinds": st.session.stats().get("kinds", {}),
                "note": (
                    "Demoted exchanges became durable facts (still recallable, "
                    "just out of the hot basket)."
                    if demoted
                    else "Nothing eligible; no DMS policy, or nothing cold enough."
                ),
            }
        )
    except Exception as e:
        return _err(e)


@mcp.resource(
    "memory://context",
    title="Memory Continuity Context",
    description="Cross-CLI working-context packet (project, goal, last summary, recent exchanges).",
    mime_type="text/markdown",
)
@_serialized
def _resource_context() -> str:
    """Text rendering of the bootstrap packet for clients that auto-read resources."""
    st = _state()
    proj = _project_name(None)
    host_name = detect_host()
    packet = bootstrap_packet(
        project=proj,
        continuity=st.continuity,
        host=host_name,
        session_stats=st.session.stats(),
        rack_stats=st.rack.stats(),
        recent=_recent_exchanges(st, proj),
    )
    st.continuity.touch(proj, host=host_name)
    st.continuity.save()
    lines = [f"# {packet['project']}", "", packet["summary"], ""]
    if packet["checkpoints"]:
        lines.append("## Recent checkpoints")
        for cp in packet["checkpoints"]:
            lines.append(f"- [{cp['host']} @ {cp['ts']}] goal={cp['goal']!r} — {cp['summary']}")
        lines.append("")
    if packet["recent"]:
        lines.append("## Recent context")
        for r in packet["recent"]:
            lines.append(f"- (salience {r.get('salience')}) {r.get('content', '')[:200]}")
        lines.append("")
    lines.append(f"Memory: {packet['memory_entries']} long-term entries, "
                 f"{packet['session'].get('entries', 0)} session entries.")
    return "\n".join(lines)

@mcp.tool(
    name="mem_search",
    annotations={
        "title": "Search Long-Term Memory",
        "read_only_hint": True,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_search(
    query: Annotated[str, Field(min_length=1, max_length=2000, description="Search text, e.g. 'how does routing work'")],
    top_k: Annotated[int, Field(ge=1, le=100, description="Maximum results to return (default 3)")] = 3,
    kind: Annotated[Optional[str], Field(description="Filter by kind: fact, doc, skill, turn, generic")] = None,
    cartridges: Annotated[Optional[list[str]], Field(description="Limit search to these cartridge names")] = None,
    group_key: Annotated[Optional[str], Field(description="Group-aware dedupe by this key (e.g. 'skill')")] = None,
    recall: Annotated[int, Field(ge=5, le=500, description="Candidate pool per cartridge before grouping (default 40)")] = 40,
) -> str:
    """Search the long-term rack for memories matching a query.

    Merges exact, lexical, and (if enabled) semantic tiers across the chosen
    cartridges, ranked by confidence. Pass group_key='skill' to deduplicate
    to the best hit per skill (entries without that meta key are kept as
    their own hits, so grouping never hides a match). Does NOT modify memory.

    Returns:
        JSON: {"query", "count", "hits"} with per-hit content, confidence,
        source, mode, cartridge, latency_ms, kind. Empty hits when nothing
        clears the confidence floor.

    Examples:
        - Use when: "what do I know about <topic>" -> query=...
        - Use when: "which skill covers <trigger>" -> group_key="skill"
    """
    try:
        st = _state()
        if group_key:
            hits = st.rack.search_grouped(
                query, top_k=top_k, group_key=group_key,
                recall=recall, kind=kind, cartridges=cartridges,
            )
        else:
            hits = st.rack.search(query, top_k=top_k, kind=kind, cartridges=cartridges)
        return _ok({"query": query, "count": len(hits), "hits": [_hit_json(h) for h in hits]})
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_recall",
    annotations={
        "title": "Recall Working Memory (text)",
        "read_only_hint": True,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_recall(
    query: Annotated[str, Field(min_length=1, max_length=2000, description="What to recall from the recent conversation")],
    top_k: Annotated[int, Field(ge=1, le=20, description="Number of exchanges to return (default 2)")] = 2,
) -> str:
    """Return the most relevant recent exchanges of the conversation as a
    readable text block (best for injecting context into a prompt).

    Returns:
        Plain text of the recalled exchanges, or "No matching session
        context." when nothing matches -- recall FAILS SAFE, never
        fabricates.
    """
    try:
        text = _state().session.recall(query, top_k=top_k)
        return text if text else "No matching session context."
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_recall_detailed",
    annotations={
        "title": "Recall Working Memory (structured)",
        "read_only_hint": True,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_recall_detailed(
    query: Annotated[str, Field(min_length=1, max_length=2000, description="What to recall from the recent conversation")],
    top_k: Annotated[int, Field(ge=1, le=20, description="Number of exchanges to return (default 2)")] = 2,
) -> str:
    """Return structured records for the most relevant recent exchanges.

    Each record carries content, confidence, the exchange index, the turn
    indices it spans, salience, and the tier that matched.

    Returns:
        JSON: {"count", "records": [...]}.
    """
    try:
        recs = _state().session.recall_detailed(query, top_k=top_k)
        return _ok({"count": len(recs), "records": recs})
    except Exception as e:
        return _err(e)


# ── write tools ──────────────────────────────────────────────────────────────

@mcp.tool(
    name="mem_observe",
    annotations={
        "title": "Record a Conversation Turn",
        "read_only_hint": False,
        "destructive_hint": False,
        "idempotent_hint": False,
        "open_world_hint": False,
    },
)
@_serialized
def mem_observe(
    role: Annotated[str, Field(min_length=1, max_length=50, description="Who spoke: 'user', 'assistant', or 'system'")],
    text: Annotated[str, Field(min_length=1, max_length=20000, description="The message content")],
) -> str:
    """Record one conversation turn into the working session.

    Feed user/assistant messages here as a conversation proceeds. When the
    buffer fills, older low-salience exchanges are offloaded (evicted) from
    live context but remain recallable. Persists the session after the turn.

    Returns:
        JSON: {"recorded", "session_entries", "evicted"}.
    """
    try:
        st = _state()
        evicted = st.session.observe(role, text)
        st.session.save()
        st.continuity.touch(_project_name(None), host=detect_host(), summary=text[:400])
        st.continuity.save()
        return _ok(
            {
                "recorded": True,
                "session_entries": st.session.stats().get("entries", 0),
                "evicted": evicted,
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_write",
    annotations={
        "title": "Write a Memory",
        "read_only_hint": False,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_write(
    content: Annotated[str, Field(min_length=1, max_length=20000, description="The memory to store, e.g. 'The capital of France is Paris.'")],
    key: Annotated[Optional[str], Field(description="Stable id; re-writing the same key replaces the old entry")] = None,
    kind: Annotated[str, Field(description="Knowledge kind: fact, doc, skill, turn, generic (default fact)")] = "fact",
    cartridge: Annotated[Optional[str], Field(description="Write to this cartridge name instead of kind-routing")] = None,
    meta: Annotated[Optional[dict[str, Any]], Field(description="Optional metadata attached to the memory")] = None,
) -> str:
    """Store a memory, upserting by key when one is given.

    Kind-routes to the facts/docs/skills cartridges unless `cartridge` names
    one explicitly. Persists the rack to disk so the fact survives process
    restarts. Writing the same key again replaces the old content.

    Returns:
        JSON: {"written", "cartridge", "kind", "entries"}.
    """
    try:
        st = _state()
        kind = kind or "fact"
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
        if cartridge:
            st.rack.write(cartridge, content, key=key, meta=meta, source="mcp", kind=kind)
        else:
            st.rack.write_routed(content, key=key, meta=meta, source="mcp", kind=kind)
        st.rack.save()
        return _ok(
            {
                "written": True,
                "cartridge": cartridge,
                "kind": kind,
                "entries": sum(st.rack.stats()[n]["entries"] for n in st.rack.names()),
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_graduate",
    annotations={
        "title": "Graduate Session Memories",
        "read_only_hint": False,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_graduate(
    target_cartridge: Annotated[Optional[str], Field(description="Receiving cartridge (default: rack default facts)")] = None,
    threshold: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence threshold to promote (default 0.40)")] = 0.40,
) -> str:
    """Promote durable session memories into a long-term cartridge.

    Runs the graduation pass over the working session: memories that are
    confident and don't duplicate existing facts are promoted. The result is
    saved to disk. Re-running after promotion promotes nothing new.

    Returns:
        JSON: {"target", "kind_hint", "threshold", "reviewed", "promoted",
        "skipped"}.
    """
    try:
        st = _state()
        kinds = {"facts": "fact", "docs": "doc", "skills": "skill"}
        target = target_cartridge or st.rack.default or "facts"
        target_kind = kinds.get(target, "fact")
        report = st.rack.graduate(
            st.session, target_name=target, threshold=threshold,
        )
        st.rack.save()
        st.session.save()
        return _ok(
            {
                "target": target,
                "kind_hint": target_kind,
                "threshold": threshold,
                "reviewed": report.get("reviewed", 0),
                "promoted": report.get("promoted", 0),
                "skipped": report.get("skipped", 0),
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_swap",
    annotations={
        "title": "Archive and Rotate Working Session",
        "read_only_hint": False,
        "destructive_hint": False,
        "idempotent_hint": False,
        "open_world_hint": False,
    },
)
@_serialized
def mem_swap(
    archive_name: Annotated[Optional[str], Field(description="Optional archive name suffix (default: archival timestamp)")] = None,
) -> str:
    """Archive the current working session to its own cartridge file and
    start a fresh session, keeping all long-term cartridges intact.

    Returns:
        JSON: {"archived", "archive_file", "session_reset"}.
    """
    try:
        st = _state()
        suffix = archive_name or "archive"
        archive_dir = st.cfg.dir / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        name = f"session-{suffix}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        archive_file = archive_dir / f"{name}.sqac"
        st.session.save(archive_file)
        st.session = ContextOffloader(
            st.cfg.dir / "session.sqac", window=8, semantic=st.cfg.semantic,
            dms=DMS(budget=SESSION_BUDGET),
        )
        return _ok(
            {"archived": name, "archive_file": str(archive_file), "session_reset": True}
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_cartridge_create",
    annotations={
        "title": "Create a Cartridge",
        "read_only_hint": False,
        "destructive_hint": True,
        "idempotent_hint": False,
        "open_world_hint": False,
    },
)
@_serialized
def mem_cartridge_create(
    name: Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9_-]+$", description="Unique cartridge name (letters, digits, underscore, dash)")],
    description: Annotated[str, Field(max_length=500, description="Optional description recorded in the header (default '')")] = "",
) -> str:
    """Create a new (empty) cartridge and mount it. Creating a name that
    already exists RESETS that cartridge -- prefer mem_write to append.

    Returns:
        JSON: {"created": name, "mounted": [...]}.
    """
    try:
        st = _state()
        st.rack.create(name, description=description)
        st.rack.save()
        return _ok({"created": name, "mounted": st.rack.names()})
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_cartridge_list",
    annotations={
        "title": "List Cartridges",
        "read_only_hint": True,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_cartridge_list() -> str:
    """List the mounted cartridges with per-cartridge stats.

    Returns:
        JSON: {"cartridges": [{"name", "entries", "dims", "encoder",
        "semantic", "kinds"}...]}.
    """
    try:
        st = _state()
        stats = st.rack.stats()
        return _ok(
            {
                "cartridges": [
                    {
                        "name": name,
                        "entries": s.get("entries", 0),
                        "dims": s.get("dims"),
                        "encoder": s.get("encoder"),
                        "semantic": s.get("semantic"),
                        "kinds": s.get("kinds"),
                    }
                    for name, s in stats.items()
                ]
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_stats",
    annotations={
        "title": "Memory Stats",
        "read_only_hint": True,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_stats() -> str:
    """Summary of the whole memory system: cartridges, session, directory.

    Returns:
        JSON: {"directory", "semantic", "cartridges": {...}, "session": {...}}.
    """
    try:
        st = _state()
        return _ok(
            {
                "directory": str(st.cfg.dir),
                "semantic": st.cfg.semantic,
                "cartridges": st.rack.stats(),
                "session": st.session.stats(),
            }
        )
    except Exception as e:
        return _err(e)


@mcp.tool(
    name="mem_save",
    annotations={
        "title": "Persist Memory to Disk",
        "read_only_hint": False,
        "destructive_hint": False,
        "idempotent_hint": True,
        "open_world_hint": False,
    },
)
@_serialized
def mem_save() -> str:
    """Persist all cartridges and the working session to disk.

    Write operations already persist; this is an explicit checkpoint.

    Returns:
        JSON: {"saved": true}.
    """
    try:
        _state().save()
        return _ok({"saved": True})
    except Exception as e:
        return _err(e)


def main(argv: Optional[list[str]] = None) -> int:
    """Console entry point (sqac-mcp). Parses CLI flags and runs the stdio
    server; a fresh SQAC_MEM_DIR is created on first call."""
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(prog=SERVER_NAME, description="SQAC memory MCP server")
    ap.add_argument("--dir", help="memory directory (default $SQAC_MEM_DIR or ~/.sqacm)")
    ap.add_argument("--no-semantic", action="store_true", help="disable the semantic tier")
    args = ap.parse_args(argv)
    if args.dir:
        os.environ["SQAC_MEM_DIR"] = args.dir
    if args.no_semantic:
        os.environ["SQAC_SEMANTIC"] = "0"
    asyncio.run(mcp.run_stdio_async())
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
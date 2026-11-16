"""Agent MCP tools — persona + working state."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer


def _agent_dir() -> Path:
    return Path.home() / ".sqacm" / "agents"


def _active_agent_name() -> Optional[str]:
    p = Path.home() / ".sqacm" / "agent_active"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def _load_agent(name: Optional[str] = None):
    import sqac.agent as am
    if name is None:
        name = _active_agent_name()
    if name is None:
        return None
    try:
        return am.load_agent(name)
    except FileNotFoundError:
        return None


def register_agent_tools(mcp: MCPServer) -> None:
    """Register agent MCP tools."""

    @mcp.tool()
    def agent_list() -> dict:
        """List available agents."""
        import sqac.agent as am
        return {"ok": True, "agents": am.list_agents(), "active": _active_agent_name()}

    @mcp.tool()
    def agent_use(name: str) -> dict:
        """Set the active agent."""
        import sqac.agent as am
        am.set_active_agent(name)
        return {"ok": True, "active": name}

    @mcp.tool()
    def agent_show(name: Optional[str] = None) -> dict:
        """Show agent bootstrap packet."""
        ag = _load_agent(name)
        if ag is None:
            return {"ok": False, "error": "agent_not_found"}
        return {"ok": True, "bootstrap": ag.bootstrap()}

    @mcp.tool()
    def agent_bootstrap(name: Optional[str] = None, max_chars: int = 4000) -> dict:
        """Get agent bootstrap packet for context injection."""
        ag = _load_agent(name)
        if ag is None:
            return {"ok": False, "error": "agent_not_found"}
        return {"ok": True, "packet": ag.bootstrap(max_chars=max_chars)}

    @mcp.tool()
    def agent_update(section: str, content: str, key: Optional[str] = None, priority: int = 1) -> dict:
        """Update agent state (identity, instruction, preference, goal, decision, note)."""
        ag = _load_agent()
        if ag is None:
            return {"ok": False, "error": "no_active_agent"}
        idx = ag.set(section, content, key=key, priority=priority)
        return {"ok": True, "section": section, "entry": idx}

    @mcp.tool()
    def agent_done(key_or_query: str) -> dict:
        """Mark a goal or decision as done."""
        ag = _load_agent()
        if ag is None:
            return {"ok": False, "error": "no_active_agent"}
        ok = ag.complete(key_or_query)
        return {"ok": ok}

    @mcp.tool()
    def agent_search(query: str, name: Optional[str] = None) -> dict:
        """Search agent memory."""
        ag = _load_agent(name)
        if ag is None:
            return {"ok": False, "error": "agent_not_found"}
        hits = ag.search(query)
        return {"ok": True, "hits": [{"content": h.content, "confidence": h.confidence, "mode": h.mode} for h in hits]}

    # Privacy: agent state is plaintext in v1

"""Cloud MCP tools — sync status and trigger."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer


def _sqac_dir() -> Path:
    return Path(os.environ.get("SQAC_MEM_DIR", str(Path.home() / ".sqacm"))).expanduser()


def register_cloud_tools(mcp: MCPServer) -> None:
    """Register cloud MCP tools."""

    @mcp.tool()
    def cloud_status() -> dict:
        """Show cloud sync status: connectivity, object count, last sync."""
        from sqac.cloud import load_cloud_config, create_backend_from_config, SyncEngine
        config = load_cloud_config()
        if config is None:
            return {"ok": False, "error": "cloud_not_configured", "hint": "run: sqac cloud init --provider <name> --bucket <bucket>"}
        try:
            backend = create_backend_from_config(config)
        except Exception as exc:
            return {"ok": False, "error": "backend_error", "detail": str(exc)}
        engine = SyncEngine(backend, _sqac_dir(), config)
        objects = engine.status()
        return {
            "ok": True,
            "provider": config.get("provider"),
            "bucket": config.get("bucket"),
            "sync_mode": config.get("sync_mode", "off"),
            "objects": [{"key": o.remote_key, "status": o.status} for o in objects],
            "total": len(objects),
        }

    @mcp.tool()
    def cloud_sync(direction: str = "both") -> dict:
        """Trigger cloud sync. direction: push, pull, or both."""
        from sqac.cloud import load_cloud_config, create_backend_from_config, SyncEngine
        config = load_cloud_config()
        if config is None:
            return {"ok": False, "error": "cloud_not_configured"}
        try:
            backend = create_backend_from_config(config)
        except Exception as exc:
            return {"ok": False, "error": "backend_error", "detail": str(exc)}
        engine = SyncEngine(backend, _sqac_dir(), config)
        results = {}
        if direction in ("push", "both"):
            results["push"] = engine.push()
        if direction in ("pull", "both"):
            results["pull"] = engine.pull()
        return {"ok": True, "direction": direction, "results": results}

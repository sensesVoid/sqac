"""Vault MCP tools — reference-only, no plaintext exfiltration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer


def _vault_path() -> Path:
    from sqac.vault import DEFAULT_VAULT_PATH
    return DEFAULT_VAULT_PATH


def _open_vault():
    from sqac.vault import Vault
    path = _vault_path()
    if not path.exists():
        return None
    passphrase = os.environ.get("SQAC_VAULT_PASSPHRASE")
    try:
        return Vault.open(path, passphrase=passphrase)
    except Exception:
        return None


def vault_list() -> dict:
    """List vault entries (metadata only, no values)."""
    v = _open_vault()
    if v is None:
        return {"ok": False, "error": "vault_not_found", "hint": "run: sqac vault init"}
    return {"ok": True, "entries": v.list()}


def vault_meta(name: str) -> dict:
    """Get metadata for a vault entry (label, type, tags, policy). No value."""
    v = _open_vault()
    if v is None:
        return {"ok": False, "error": "vault_not_found"}
    try:
        return {"ok": True, "meta": v.get_meta(name)}
    except KeyError as exc:
        return {"ok": False, "error": "not_found", "detail": str(exc)}


def vault_http(name: str, method: str, url: str, body: Optional[str] = None, headers: Optional[dict] = None) -> dict:
    """Proxy an HTTP request using a vault secret. Host must be allowlisted."""
    v = _open_vault()
    if v is None:
        return {"ok": False, "error": "vault_not_found"}
    path = _vault_path()
    if not path.exists():
        return {"ok": False, "error": "vault_not_found"}
    try:
        v = v or _open_vault()
    except Exception:
        pass
    # Check if vault is locked by attempting a metadata op
    try:
        v.get_meta(name)
    except Exception as exc:
        if "locked" in str(exc).lower():
            return {"ok": False, "error": "vault_locked", "hint": "set SQAC_VAULT_PASSPHRASE or use keychain"}
    try:
        body_bytes = body.encode() if body else None
        result = v.http(name, method, url, headers=headers, body=body_bytes)
        return {"ok": True, **result}
    except KeyError as exc:
        return {"ok": False, "error": "not_found", "detail": str(exc)}
    except ValueError as exc:
        return {"ok": False, "error": "policy_violation", "detail": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": "request_failed", "detail": str(exc)}


def vault_audit(limit: int = 20) -> dict:
    """Read recent vault audit entries (no values)."""
    from sqac.vault_audit import read_audit
    entries = read_audit(tail=limit)
    return {"ok": True, "entries": entries}


def register_vault_tools(mcp: MCPServer) -> None:
    """Register vault MCP tools. No value-returning tools."""
    mcp.tool()(vault_list)
    mcp.tool()(vault_meta)
    mcp.tool()(vault_http)
    mcp.tool()(vault_audit)

    # Deliberately NOT registered: vault_get, vault_resolve, vault_put, vault_exec

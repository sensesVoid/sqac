"""Secret Domain MCP tools — reference-only, no plaintext, ever.

Registered tools expose metadata about secret references (names, resolution
kinds, descriptions, tags). There is deliberately no tool that resolves or
returns a secret value.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer


def _secret_dir() -> Path:
    return Path.home() / ".sqacm" / "secrets"


def _default_domain() -> str:
    return os.environ.get("SQAC_SECRET_DOMAIN", "default")


def _open_domain(domain: Optional[str] = None):
    from sqac.secret_domain import SECRET_DIR, SecretDomain

    name = (domain or _default_domain()).strip()
    path = SECRET_DIR / f"{name}.sqac"
    if not path.exists():
        return None
    passphrase = os.environ.get("SQAC_SECRET_PASSPHRASE")
    try:
        return SecretDomain.open(path, passphrase=passphrase)
    except Exception:
        return None


def secret_list(domain: Optional[str] = None) -> dict:
    """List secret references in a domain (metadata only, never values)."""
    from sqac.secret_domain import list_domains

    name = (domain or _default_domain()).strip()
    target = _secret_dir() / f"{name}.sqac"
    if not target.exists():
        return {"ok": False, "error": "domain_not_found",
                "hint": f"run: sqac secret init {name}",
                "domains": list_domains()}
    d = _open_domain(name)
    if d is None:
        return {"ok": False, "error": "domain_locked",
                "hint": "set SQAC_SECRET_PASSPHRASE or unlock the keychain"}
    return {"ok": True, "domain": name, "refs": d.list()}


def secret_meta(name: str, domain: Optional[str] = None) -> dict:
    """Get metadata for a secret reference (no values, no resolution)."""
    domain_name = (domain or _default_domain()).strip()
    d = _open_domain(domain_name)
    if d is None:
        return {"ok": False, "error": "domain_not_found_or_locked"}
    try:
        return {"ok": True, "domain": domain_name, "meta": d.get_ref_meta(name)}
    except KeyError as exc:
        return {"ok": False, "error": "not_found", "detail": str(exc)}


def register_secret_tools(mcp: MCPServer) -> None:
    """Register secret-domain MCP tools. No value-returning tools, ever."""
    mcp.tool()(secret_list)
    mcp.tool()(secret_meta)

    # Deliberately NOT registered: secret_resolve, secret_get, secret_put

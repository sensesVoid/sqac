"""Backend-only secret resolution for SQAC.

Resolves secret references to plaintext values **inside the trusted backend
process**. Plaintext values MUST NOT be returned to UI clients, MUST NOT be
logged, and SHOULD be held in memory only as long as needed for the outbound
provider/tool call.

This module is for library/server-side use only. The MCP surface
(``sqac.secret_mcp``) and HTTP surfaces expose metadata and audit — never
plaintext.

Supported ref forms:
- ``vault:NAME``                 -> secret NAME in the active/default vault
- ``vault:VAULT/NAME``           -> secret NAME in a named vault (multi-vault)
- ``${vault:NAME}`` / ``${vault:VAULT/NAME}``  -> braced forms
- ``provider:1password:<ref>``   -> 1Password CLI fetch (op read <ref>)
- ``provider:bitwarden:<ref>``   -> Bitwarden CLI fetch (bw get password <ref>)
- ``op://...`` / ``bw://...``    -> convenience provider aliases
- ``secret:DOMAIN/NAME``         -> indirection via a Secret Domain cartridge
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


class SecretResolutionError(RuntimeError):
    """Raised when a secret reference cannot be resolved safely."""


_AUDIT_REF_MAX = 120


def _safe_ref_for_audit(ref: str) -> str:
    """Truncate/sanitize a ref for audit logs (never include resolved values)."""
    ref = (ref or "").strip()
    if len(ref) > _AUDIT_REF_MAX:
        return ref[: _AUDIT_REF_MAX - 3] + "..."
    return ref


def _audit_file(event: dict[str, Any]) -> None:
    """Default audit sink: append to the shared hash-chained audit log.

    Only ref metadata and outcomes are recorded — never values.
    """
    try:
        from .vault_audit import append_audit
        append_audit(
            vault=str(event.get("vault", "secret-resolver")),
            name=str(event.get("ref", ""))[:_AUDIT_REF_MAX],
            op=str(event.get("event", "resolve")),
            success=str(event.get("event", "")) == "secret_resolve_ok",
            bytes_count=0,
            error=str(event.get("error", "")),
        )
    except Exception:
        pass  # audit must never break resolution


@dataclass
class SecretResolver:
    """Resolve secret references in the trusted backend process.

    Args:
        sqac_dir: Base SQAC memory directory (default ``~/.sqacm``).
        default_vault: Vault name for unqualified ``vault:`` refs.
        audit_callback: Optional callable(event_dict) for audit logging.
            The callback MUST NOT receive secret values. When ``None``,
            events are appended to the shared ``vault-audit.jsonl`` log.
        allow_providers: Whether to allow provider CLI fetches (1P/BW).
    """

    sqac_dir: Path = None  # type: ignore[assignment]
    default_vault: str = "default"
    audit_callback: Optional[Callable[[dict[str, Any]], None]] = None
    allow_providers: bool = True

    def __post_init__(self) -> None:
        if self.sqac_dir is None:
            self.sqac_dir = Path.home() / ".sqacm"
        self.sqac_dir = Path(self.sqac_dir)

    # -- audit -----------------------------------------------------------

    def _audit(self, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), **event}
        if self.audit_callback is not None:
            try:
                self.audit_callback(event)
            except Exception:
                pass  # audit must never break resolution
        else:
            _audit_file(event)

    # -- parsing ----------------------------------------------------------

    @staticmethod
    def parse_ref(ref: str) -> tuple[str, dict[str, str]]:
        """Parse a secret reference into ``(kind, parts)``.

        Returns one of:
            ``("vault", {"vault": <name|None>, "name": <secret>})``
            ``("provider", {"provider": <id>, "ref": <provider-ref>})``
            ``("secret_domain", {"domain": <domain>, "name": <name>})``

        Raises:
            SecretResolutionError: on empty/unsupported refs.
        """
        ref = (ref or "").strip()
        if not ref:
            raise SecretResolutionError("Empty secret reference")

        m = re.match(r"^\$\{vault:(?:(?P<vault>[^/]+)/)?(?P<name>.+)\}$", ref)
        if m:
            return "vault", {"vault": m.group("vault"), "name": m.group("name")}

        m = re.match(r"^(?:vault://|vault:)(?:(?P<vault>[^/]+)/)?(?P<name>.+)$", ref)
        if m:
            return "vault", {"vault": m.group("vault"), "name": m.group("name")}

        m = re.match(r"^secret:(?P<domain>[^/]+)/(?P<name>.+)$", ref)
        if m:
            return "secret_domain", {"domain": m.group("domain"), "name": m.group("name")}

        m = re.match(r"^provider:(?P<provider>1password|bitwarden):(?P<ref>.+)$", ref)
        if m:
            return "provider", {"provider": m.group("provider"), "ref": m.group("ref")}

        if ref.startswith("op://"):
            return "provider", {"provider": "1password", "ref": ref}
        if ref.startswith("bw://"):
            return "provider", {"provider": "bitwarden", "ref": ref[5:]}

        raise SecretResolutionError(
            f"Unsupported secret reference format: {_safe_ref_for_audit(ref)!r}. "
            "Expected vault:NAME, vault:VAULT/NAME, secret:DOMAIN/NAME, "
            "provider:1password:<op://...>, or provider:bitwarden:<id>."
        )

    # -- resolution --------------------------------------------------------

    def _resolve_vault_ref(self, vault_name: Optional[str], secret_name: str) -> str:
        from .vault_registry import VaultRegistry
        from .vault import Vault

        vname = vault_name or self.default_vault
        registry_dir = self.sqac_dir / "vaults"
        try:
            reg = VaultRegistry(registry_dir)
            try:
                v = reg.open(vname)
            except Exception:
                legacy = self.sqac_dir / "vaults" / f"{vname}.sqac"
                default = self.sqac_dir / "vaults" / "default.sqac"
                if vname == "default" and default.exists():
                    v = Vault.open(default)
                elif legacy.exists():
                    v = Vault.open(legacy)
                else:
                    raise
            return v.resolve(secret_name)
        except SecretResolutionError:
            raise
        except Exception as exc:
            raise SecretResolutionError(
                f"Failed to resolve vault ref vault:{vname}/{secret_name}: {exc}"
            ) from exc

    def _resolve_provider_ref(self, provider: str, provider_ref: str) -> str:
        if not self.allow_providers:
            raise SecretResolutionError("Provider-based secret resolution is disabled")
        from . import providers
        try:
            raw = providers.fetch(provider, provider_ref)
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="strict")
            return str(raw)
        except Exception as exc:
            raise SecretResolutionError(
                f"Failed to fetch from provider {provider!r}: {exc}"
            ) from exc

    def _resolve_secret_domain_ref(self, domain: str, name: str) -> str:
        """Resolve ``secret:DOMAIN/NAME`` via a Secret Domain cartridge.

        The domain stores reference metadata only; the target ref is resolved
        recursively. Chains of secret-domain indirection are rejected.
        """
        from .secret_domain import SecretDomain

        domain_path = self.sqac_dir / "secrets" / f"{domain}.sqac"
        if not domain_path.exists():
            raise SecretResolutionError(
                f"Secret domain {domain!r} not found at {domain_path}. "
                "Create it with `sqac secret init` or use a direct vault/provider ref."
            )
        try:
            sd = SecretDomain.open(domain_path)
            meta = sd.get_ref_meta(name)
        except KeyError as exc:
            raise SecretResolutionError(f"Secret {name!r} not found in domain {domain!r}") from exc
        except Exception as exc:
            raise SecretResolutionError(
                f"Failed to read secret domain {domain!r}: {exc}"
            ) from exc
        target = meta.get("ref")
        if not target:
            raise SecretResolutionError(f"Secret {name!r} in domain {domain!r} has no ref")
        if str(target).startswith("secret:"):
            raise SecretResolutionError("Secret domain indirection chains are not allowed")
        return self.resolve(str(target), _depth=1)

    def resolve(self, ref: str, *, agent_context: Optional[dict[str, Any]] = None,
                _depth: int = 0) -> str:
        """Resolve a secret reference to plaintext (backend-only).

        Args:
            ref: Secret reference string.
            agent_context: Optional agent binding info for authorization
                checks. When it contains ``secret_refs``/``secret_ref``, the
                ref must be present in the allowlist or resolution fails
                closed.
            _depth: Internal recursion guard for secret-domain indirection.

        Returns:
            Plaintext secret value. Callers MUST NOT log or expose it.

        Raises:
            SecretResolutionError: on any failure (fail closed).
        """
        if _depth > 2:
            raise SecretResolutionError("Secret reference indirection too deep")

        kind, parts = self.parse_ref(ref)
        safe_ref = _safe_ref_for_audit(ref)
        t0 = time.time()

        if agent_context:
            allowed = list(agent_context.get("secret_refs") or [])
            primary = agent_context.get("secret_ref")
            if primary and primary not in allowed:
                allowed = [primary] + allowed
            if allowed:
                norm = lambda s: (s or "").strip()
                if norm(ref) not in {norm(a) for a in allowed}:
                    err = SecretResolutionError(
                        f"Secret ref not allowed for agent: {safe_ref}"
                    )
                    err.secret_resolver_denied = True  # type: ignore[attr-defined]
                    self._audit({
                        "event": "secret_resolve_denied",
                        "ref": safe_ref,
                        "reason": "not_in_agent_allowlist",
                    })
                    raise err

        try:
            if kind == "vault":
                value = self._resolve_vault_ref(parts.get("vault"), parts["name"])
            elif kind == "provider":
                value = self._resolve_provider_ref(parts["provider"], parts["ref"])
            elif kind == "secret_domain":
                value = self._resolve_secret_domain_ref(parts["domain"], parts["name"])
            else:
                raise SecretResolutionError(f"Unknown ref kind: {kind}")

            self._audit({
                "event": "secret_resolve_ok",
                "ref": safe_ref,
                "kind": kind,
                "latency_ms": round((time.time() - t0) * 1000, 2),
            })
            return value
        except SecretResolutionError as exc:
            if not getattr(exc, "secret_resolver_denied", False):
                self._audit({
                    "event": "secret_resolve_error",
                    "ref": safe_ref,
                    "kind": kind,
                    "error": str(exc),
                })
            raise
        except Exception as exc:
            self._audit({
                "event": "secret_resolve_error",
                "ref": safe_ref,
                "kind": kind,
                "error": str(exc),
            })
            raise SecretResolutionError(f"Failed to resolve {safe_ref}: {exc}") from exc

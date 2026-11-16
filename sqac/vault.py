"""Vault cartridge: encrypted secret/PII storage with reference-only access."""

from __future__ import annotations

import base64
import os
import re
import secrets
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .format import (
    CartridgeHeader,
    Entry,
    VAULT_WRITE_VERSION,
    read_cartridge,
    write_cartridge,
)
from .vault_crypto import (
    generate_dek,
    encrypt_value,
    decrypt_value,
    wrap_dek,
    unwrap_dek,
    derive_kek_argon2id,
    keychain_get,
    keychain_set,
    KeychainUnavailable,
    VaultCryptoError,
    get_unlock_cache,
)


# ── constants ──────────────────────────────────────────────────────────────

VAULT_DIR = Path.home() / ".sqacm" / "vaults"
DEFAULT_VAULT_PATH = VAULT_DIR / "default.sqac"
VALUE_CAP = 64 * 1024  # 64 KB per entry
VAULT_NAME = "default"


# ── reference parsing ──────────────────────────────────────────────────────

_REF_RE = re.compile(r"^(?:vault://|vault:)(?:(?P<vault>[^/]+)/)?(?P<name>.+)$")
_BRACED_RE = re.compile(r"^\$\{vault:(?:(?P<vault>[^/]+)/)?(?P<name>.+)\}$")


def parse_vault_ref(s: str) -> Optional[tuple[Optional[str], str]]:
    """Parse a vault reference.

    Returns ``(vault_name, secret_name)`` where *vault_name* is ``None``
    when the ref does not specify a vault (``vault:NAME``), or ``None``
    when *s* is not a vault reference at all.
    """
    s = s.strip()
    m = _REF_RE.match(s)
    if m:
        v = m.group("vault")
        return (v, m.group("name")) if v else (None, m.group("name"))
    m = _BRACED_RE.match(s)
    if m:
        v = m.group("vault")
        return (v, m.group("name")) if v else (None, m.group("name"))
    return None


def canonical_ref(name: str, vault: Optional[str] = None) -> str:
    if vault:
        return f"vault:{vault}/{name}"
    return f"vault:{name}"


def _looks_like_pii_name(name: str) -> bool:
    """Warn if a name looks like an email/phone (PII belongs in value)."""
    if "@" in name or re.match(r"^\+?\d[\d\s\-]{6,}$", name):
        return True
    return False


# ── helpers ────────────────────────────────────────────────────────────────

def _vault_path(path: Optional[str | Path] = None) -> Path:
    return Path(path) if path else DEFAULT_VAULT_PATH


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _key_bits_for_name(name: str) -> bytes:
    """Derive key_bits for exact lookup from name (hash-based)."""
    h = hashlib.sha256(name.encode()).digest()
    # Pad/truncate to dims//8 (1024 dims = 128 bytes)
    return (h * 4)[:128]


import hashlib


def _entry_to_dict(e) -> dict:
    """Convert an Entry to its vault payload dict."""
    return dict(e.payload) if isinstance(e.payload, dict) else {}


# ── Vault class ────────────────────────────────────────────────────────────

class Vault:
    """Encrypted vault cartridge API."""

    def __init__(self, path: str | Path, header, entries: list, dek: bytes):
        self.path = Path(path)
        self.header = header
        self._entries = entries
        self._dek = dek
        self._meta: dict[str, int] = {}
        for i, e in enumerate(entries):
            p = e.payload if isinstance(e.payload, dict) else {}
            self._meta[p.get("name", "")] = i

    @classmethod
    def create(cls, path, mode: str = "keychain", passphrase: Optional[str] = None,
               master_key: Optional[bytes] = None, provider: Optional[str] = None,
               provider_ref: Optional[str] = None) -> "Vault":
        """Create a new vault cartridge."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dek = generate_dek()
        salt = os.urandom(16)
        wrap_nonce = ""
        wrapped_dek = ""

        if mode == "provider":
            from .providers import fetch as _fetch_provider
            if provider is None or provider_ref is None:
                raise ValueError("provider and provider_ref required for provider mode")
            master_key = _fetch_provider(provider, provider_ref)
            wn, wct = wrap_dek(dek, master_key)
            wrap_nonce = wn
            wrapped_dek = wct
            kdf = "provider"
            salt_b64 = ""
        elif mode == "keychain":
            if master_key is None:
                # Generate a random master key and store in keychain
                master_key = secrets.token_bytes(32)
                try:
                    keychain_set(value=master_key.hex())
                except KeychainUnavailable:
                    # Fall back to passphrase mode
                    mode = "portable"
                    if passphrase is None:
                        import getpass
                        passphrase = getpass.getpass("Vault passphrase: ")
            if mode == "keychain":
                wn, wct = wrap_dek(dek, master_key)
                wrap_nonce = wn
                wrapped_dek = wct
                kdf = "keychain"
                salt_b64 = ""
            else:
                # portable mode
                if passphrase is None:
                    import getpass
                    passphrase = getpass.getpass("Vault passphrase: ")
                kek = derive_kek_argon2id(passphrase, salt)
                wn, wct = wrap_dek(dek, kek)
                wrap_nonce = wn
                wrapped_dek = wct
                kdf = "argon2id"
                salt_b64 = base64.b64encode(salt).decode("ascii")
        else:
            # portable mode
            if passphrase is None:
                import getpass
                passphrase = getpass.getpass("Vault passphrase: ")
            kek = derive_kek_argon2id(passphrase, salt)
            wn, wct = wrap_dek(dek, kek)
            wrap_nonce = wn
            wrapped_dek = wct
            kdf = "argon2id"
            salt_b64 = base64.b64encode(salt).decode("ascii")

        vault_extra = {
            "schema": 1,
            "kdf": kdf,
            "salt": salt_b64,
            "time_cost": 3,
            "memory_cost": 65536,
            "parallelism": 4,
            "wrap_nonce": wrap_nonce,
            "wrapped_dek": wrapped_dek,
            "created": _now(),
        }
        if kdf == "keychain":
            vault_extra["keychain_service"] = "sqac-vault"
        elif kdf == "provider":
            vault_extra["provider"] = provider
            vault_extra["provider_ref"] = provider_ref

        hdr = CartridgeHeader(
            version=VAULT_WRITE_VERSION,
            dims=1024,
            extra={
                "cartridge_type": "vault",
                "sync_policy": "never",
                "vault": vault_extra,
            },
        )
        write_cartridge(str(path), hdr, {}, [])
        return cls(path, hdr, [], dek)

    @classmethod
    def open(cls, path, passphrase: Optional[str] = None) -> "Vault":
        """Open an existing vault cartridge and unwrap the DEK."""
        path = Path(path)
        cart = read_cartridge(str(path))
        hdr = cart.header
        vinfo = hdr.extra.get("vault", {})
        kdf = vinfo.get("kdf", "keychain")

        mtime = path.stat().st_mtime
        # Only use cache when no explicit passphrase (keychain mode)
        if passphrase is None:
            cached = get_unlock_cache().get(str(path), mtime)
            if cached is not None:
                return cls(path, hdr, cart.entries, cached)

        if kdf == "keychain":
            hex_key = keychain_get(
                service=vinfo.get("keychain_service", "sqac-vault"),
                username="default",
            )
            if hex_key is None:
                # Fall back to passphrase
                if passphrase is None:
                    passphrase = os.environ.get("SQAC_VAULT_PASSPHRASE")
                if passphrase is None:
                    raise VaultCryptoError("vault locked: no keychain key and no passphrase")
                salt = base64.b64decode(vinfo["salt"].encode()) if vinfo.get("salt") else os.urandom(16)
                kek = derive_kek_argon2id(passphrase, salt, vinfo.get("time_cost", 3), vinfo.get("memory_cost", 65536), vinfo.get("parallelism", 4))
            else:
                master_key = bytes.fromhex(hex_key)
                kek = master_key
        elif kdf == "provider":
            from .providers import fetch as _fetch_provider
            provider = vinfo.get("provider")
            provider_ref = vinfo.get("provider_ref")
            if provider is None or provider_ref is None:
                raise VaultCryptoError("vault locked: missing provider info")
            try:
                master_key = _fetch_provider(provider, provider_ref)
            except Exception as e:
                raise VaultCryptoError(f"vault locked: provider fetch failed: {e}")
            kek = master_key
        else:
            if passphrase is None:
                passphrase = os.environ.get("SQAC_VAULT_PASSPHRASE")
            if passphrase is None:
                raise VaultCryptoError("vault locked: passphrase required")
            salt = base64.b64decode(vinfo["salt"].encode())
            kek = derive_kek_argon2id(passphrase, salt, vinfo.get("time_cost", 3), vinfo.get("memory_cost", 65536), vinfo.get("parallelism", 4))

        dek = unwrap_dek(vinfo["wrap_nonce"], vinfo["wrapped_dek"], kek)
        get_unlock_cache().put(str(path), mtime, dek)
        return cls(path, hdr, cart.entries, dek)

    def _find_entry_idx(self, name: str) -> Optional[int]:
        for i, e in enumerate(self._entries):
            p = e.payload if isinstance(e.payload, dict) else {}
            if p.get("name") == name:
                return i
        return None

    def put(self, name: str, value: str | bytes, entry_type: str = "secret", label: str = "", tags: list = None, policy: dict = None) -> dict:
        if tags is None:
            tags = []
        if policy is None:
            policy = {}
        if _looks_like_pii_name(name):
            warnings.warn(f"vault entry name {name!r} looks like PII; consider using an opaque identifier", stacklevel=2)
        if isinstance(value, str):
            value = value.encode("utf-8")
        if len(value) > VALUE_CAP:
            raise ValueError(f"value exceeds {VALUE_CAP} byte cap")
        # Validate policy
        allowed_hosts = policy.get("allowed_hosts", [])
        auth = policy.get("auth", "bearer")
        if auth not in ("bearer",) and not auth.startswith("header:") and not auth.startswith("query:"):
            raise ValueError(f"invalid auth mode: {auth}")
        if "*" in allowed_hosts:
            raise ValueError("wildcard hosts not allowed by default")

        now = _now()
        nonce, ct = encrypt_value(self._dek, value, vault=VAULT_NAME, name=name, entry_type=entry_type)
        payload = {
            "name": name,
            "entry_type": entry_type,
            "label": label,
            "tags": tags,
            "policy": {"allowed_hosts": allowed_hosts, "auth": auth, "header_name": policy.get("header_name")},
            "created": now,
            "updated": now,
            "nonce": nonce,
            "ct": ct,
        }

        idx = self._find_entry_idx(name)
        if idx is not None:
            old = self._entries[idx].payload if isinstance(self._entries[idx].payload, dict) else {}
            payload["created"] = old.get("created", now)
            self._entries[idx] = Entry(
                key_bits=_key_bits_for_name(name),
                payload=payload,
                kind=0,
                encrypted=True,
            )
        else:
            self._entries.append(Entry(
                key_bits=_key_bits_for_name(name),
                payload=payload,
                kind=0,
                encrypted=True,
            ))

        self._save()
        self._meta[name] = self._find_entry_idx(name)
        # Audit
        _audit("put", name, success=True, bytes=len(value))
        return {"name": name, "entry_type": entry_type, "label": label, "tags": tags}

    def get_meta(self, name: str) -> dict:
        idx = self._find_entry_idx(name)
        if idx is None:
            raise KeyError(f"vault entry not found: {name}")
        p = self._entries[idx].payload if isinstance(self._entries[idx].payload, dict) else {}
        _audit("meta", name, success=True, bytes=0)
        return {
            "name": p.get("name"),
            "entry_type": p.get("entry_type"),
            "label": p.get("label", ""),
            "tags": p.get("tags", []),
            "policy": p.get("policy", {}),
            "created": p.get("created"),
            "updated": p.get("updated"),
        }

    def list(self) -> list[dict]:
        out = []
        for e in self._entries:
            p = e.payload if isinstance(e.payload, dict) else {}
            out.append({
                "name": p.get("name"),
                "entry_type": p.get("entry_type"),
                "label": p.get("label", ""),
                "tags": p.get("tags", []),
            })
        return out

    def remove(self, name: str) -> bool:
        idx = self._find_entry_idx(name)
        if idx is None:
            return False
        del self._entries[idx]
        self._save()
        self._meta.pop(name, None)
        _audit("remove", name, success=True, bytes=0)
        return True

    def resolve(self, name: str) -> str:
        """Internal only — never exposed through MCP."""
        idx = self._find_entry_idx(name)
        if idx is None:
            raise KeyError(f"vault entry not found: {name}")
        e = self._entries[idx]
        p = e.payload if isinstance(e.payload, dict) else {}
        val = decrypt_value(
            self._dek,
            p["nonce"],
            p["ct"],
            vault=VAULT_NAME,
            name=name,
            entry_type=p.get("entry_type", "secret"),
        )
        return val.decode("utf-8")

    def rotate(self, mode: Optional[str] = None, passphrase: Optional[str] = None) -> None:
        # Generate new DEK
        new_dek = generate_dek()
        # Re-encrypt all entries
        for e in self._entries:
            p = e.payload if isinstance(e.payload, dict) else {}
            val = decrypt_value(self._dek, p["nonce"], p["ct"], vault=VAULT_NAME, name=p["name"], entry_type=p.get("entry_type", "secret"))
            n, ct = encrypt_value(new_dek, val, vault=VAULT_NAME, name=p["name"], entry_type=p.get("entry_type", "secret"))
            p["nonce"] = n
            p["ct"] = ct
            p["updated"] = _now()
        self._dek = new_dek
        # Re-wrap DEK
        if mode == "keychain" or (mode is None and self.header.extra.get("vault", {}).get("kdf") == "keychain"):
            master_key = secrets.token_bytes(32)
            try:
                keychain_set(value=master_key.hex())
            except KeychainUnavailable:
                mode = "portable"
            if mode != "portable":
                wn, wct = wrap_dek(new_dek, master_key)
                kdf = "keychain"
                salt_b64 = ""
        else:
            if passphrase is None:
                import getpass
                passphrase = getpass.getpass("New vault passphrase: ")
            salt = os.urandom(16)
            kek = derive_kek_argon2id(passphrase, salt)
            wn, wct = wrap_dek(new_dek, kek)
            kdf = "argon2id"
            salt_b64 = base64.b64encode(salt).decode("ascii")

        vinfo = self.header.extra["vault"]
        vinfo["kdf"] = kdf
        vinfo["salt"] = salt_b64
        vinfo["wrap_nonce"] = wn
        vinfo["wrapped_dek"] = wct
        vinfo["created"] = vinfo.get("created", _now())
        self._save()
        get_unlock_cache().clear(str(self.path))
        _audit("rotate", "*", success=True, bytes=0)

    def exec_env(self, mapping: dict[str, str], cmd: list[str], redact: bool = True) -> "ExecResult":
        """Resolve vault refs in mapping and run cmd with them injected."""
        import subprocess

        # Resolve only listed variables
        resolved: dict[str, str] = {}
        for var, ref in mapping.items():
            parsed = parse_vault_ref(ref)
            if parsed is None:
                raise ValueError(f"not a vault reference: {ref!r}")
            vault_name, name = parsed
            # Multi-vault: resolve from the named vault if specified
            if vault_name is not None:
                from .vault_registry import VaultRegistry
                reg = VaultRegistry()
                resolved[var] = reg.resolve(name, vault=vault_name)
            else:
                resolved[var] = self.resolve(name)

        # Build child env: copy current, strip SQAC_VAULT_*, inject resolved
        child_env = {k: v for k, v in os.environ.items() if not k.startswith("SQAC_VAULT_")}
        child_env.update(resolved)

        proc = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
        stdout = proc.stdout
        stderr = proc.stderr
        if redact:
            for var, val in resolved.items():
                stdout = stdout.replace(val, f"[REDACTED:vault:{var}]")
                stderr = stderr.replace(val, f"[REDACTED:vault:{var}]")
        _audit("exec", ",".join(mapping.keys()), success=proc.returncode == 0, bytes=len(proc.stdout) + len(proc.stderr))
        return ExecResult(returncode=proc.returncode, stdout=stdout, stderr=stderr)

    def http(self, name: str, method: str, url: str, headers: Optional[dict] = None, body: Optional[bytes] = None, timeout: int = 30):
        """Proxy HTTP request with secret injection. See spec 9.4."""
        import urllib.request
        import urllib.error

        idx = self._find_entry_idx(name)
        if idx is None:
            raise KeyError(f"vault entry not found: {name}")
        e = self._entries[idx]
        p = e.payload if isinstance(e.payload, dict) else {}
        policy = p.get("policy", {})
        allowed_hosts = policy.get("allowed_hosts", [])
        auth = policy.get("auth", "bearer")
        secret = self.resolve(name)

        # Parse URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError("HTTPS required")
        host = parsed.hostname
        if host not in allowed_hosts:
            raise ValueError(f"host {host!r} not in allowed_hosts")
        if "*" in allowed_hosts:
            raise ValueError("wildcard hosts not allowed")

        # Build headers
        req_headers = dict(headers or {})
        if auth == "bearer":
            req_headers["Authorization"] = f"Bearer {secret}"
        elif auth.startswith("header:"):
            hname = auth.split(":", 1)[1]
            req_headers[hname] = secret
        elif auth.startswith("query:"):
            param = auth.split(":", 1)[1]
            # append to URL
            sep = "&" if parsed.query else "?"
            url = f"{url}{sep}{param}={secret}"

        # Redirect handler that checks allowlist
        class _RedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                new_parsed = urlparse(newurl)
                if new_parsed.scheme != "https":
                    raise ValueError("redirect to non-HTTPS rejected")
                if new_parsed.hostname not in allowed_hosts:
                    raise ValueError(f"redirect host {new_parsed.hostname!r} not allowed")
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(_RedirectHandler)
        req = urllib.request.Request(url, data=body, headers=req_headers, method=method.upper())
        try:
            with opener.open(req, timeout=timeout) as resp:
                resp_body = resp.read(1_048_576 + 1)
                if len(resp_body) > 1_048_576:
                    resp_body = resp_body[:1_048_576]
                _audit("http", name, success=True, bytes=len(resp_body), host=host)
                return {
                    "status": resp.status,
                    "headers": dict(resp.headers),
                    "body": resp_body.decode("utf-8", errors="replace"),
                }
        except urllib.error.HTTPError as exc:
            body = exc.read(1_048_576)
            _audit("http", name, success=False, bytes=len(body), host=host, error=str(exc.code))
            return {"status": exc.code, "body": body.decode("utf-8", errors="replace")}

    def _save(self) -> None:
        hdr = self.header
        hdr.extra["vault"] = hdr.extra.get("vault", {})
        hdr.extra["sync_policy"] = hdr.extra.get("sync_policy") or "never"
        write_cartridge(str(self.path), hdr, {}, self._entries)

    def set_sync_policy(self, policy: str) -> None:
        """Set the cartridge-level cloud sync policy (fail-closed sync engine honors it)."""
        if policy not in ("write_through", "never", "local_only"):
            raise ValueError("sync_policy must be one of write_through|never|local_only")
        self.header.extra["sync_policy"] = policy
        self._save()

    def scan(self, paths: list[str]) -> list[dict]:
        """Scan plaintext cartridges for secrets/PII. Returns list of findings."""
        from .pii import scan_cartridge
        findings = []
        for p in paths:
            findings.extend(scan_cartridge(p))
        return findings

    @staticmethod
    def scan_all(paths: list[str]) -> list[dict]:
        from .pii import scan_cartridge
        findings = []
        for p in paths:
            findings.extend(scan_cartridge(p))
        return findings


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


# ── audit helper ───────────────────────────────────────────────────────────

def _audit(op: str, name: str, success: bool, bytes: int = 0, host: str = "", error: str = ""):
    try:
        from .vault_audit import append_audit
        append_audit(vault=VAULT_NAME, name=name, op=op, success=success, bytes_count=bytes, host=host, error=error)
    except Exception:
        pass  # audit must never break operations


# ── module-level convenience ───────────────────────────────────────────────

def open_vault(path=None, passphrase=None):
    return Vault.open(_vault_path(path), passphrase=passphrase)


def create_vault(path=None, mode="keychain", passphrase=None):
    return Vault.create(_vault_path(path), mode=mode, passphrase=passphrase)

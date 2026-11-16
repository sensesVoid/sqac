"""Secret Domain cartridge: references and metadata only, never raw values.

A Secret Domain is a registry of *pointers* to secrets an application or
agent is authorized to use. It never stores plaintext secret material;
values live in Vault cartridges or external providers (1Password,
Bitwarden) and are resolved only inside the trusted backend process
(see ``sqac.secret_resolver``).

Layout
------
Domains live at ``~/.sqacm/secrets/<name>.sqac`` and are managed through
module-level helpers (``create_domain`` / ``open_domain`` /
``list_domains``), mirroring the multi-vault and multi-agent patterns.

At-rest protection
------------------
The whole domain file is encrypted with an SQSE envelope (AES-256-GCM,
same primitives as vault/agent: keychain-held master key by default, or
Argon2id + passphrase in portable mode). Set ``SQAC_SECRET_PASSPHRASE``
for non-interactive passphrase unlock. ``--unencrypted`` is available for
development only.

Ref format
----------
- ``vault:NAME`` / ``vault:VAULT/NAME`` / ``vault://...`` / ``${vault:...}``
- ``provider:1password:<ref>`` / ``provider:bitwarden:<ref>``
- ``op://...`` / ``bw://...`` convenience aliases

``secret:DOMAIN/NAME`` targets are rejected to prevent indirection chains.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .credentials import detect_credentials
from .format import (
    CartridgeHeader,
    Entry,
    WRITE_VERSION,
    read_cartridge,
    write_cartridge,
)
from .vault import parse_vault_ref
from .vault_audit import append_audit
from .vault_crypto import (
    VaultCryptoError as DecryptionError,
    derive_kek_argon2id,
    generate_nonce,
    keychain_get,
    keychain_set,
)

# ── constants ────────────────────────────────────────────────────────────────

SECRET_DIR = Path.home() / ".sqacm" / "secrets"
_SECRET_MAGIC = b"SQSE"
_SECRET_AAD = b"sqac-secret-v1"
SCHEMA_VERSION = 1
MAX_NAME_LEN = 128
MAX_REF_LEN = 512
MAX_FIELD_LEN = 2048

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROVIDER_REF_RE = re.compile(r"^provider:(1password|bitwarden):.+")
_DOMAIN_REF_RE = re.compile(r"^secret:([^/]+)/(.+)$")
_OP_RE = re.compile(r"^op://.+")
_BW_RE = re.compile(r"^bw://.+")


# ── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _key_bits_for_name(name: str) -> bytes:
    """Derive key_bits for exact lookup from name (hash-based)."""
    h = hashlib.sha256(name.encode()).digest()
    return (h * 4)[:128]


def parse_domain_ref(ref: str) -> Optional[tuple[str, str]]:
    """Parse ``secret:DOMAIN/NAME`` → ``(domain, name)``, else ``None``."""
    ref = (ref or "").strip()
    m = _DOMAIN_REF_RE.match(ref)
    if m:
        return (m.group(1), m.group(2))
    return None


def classify_ref(ref: str) -> str:
    """Return the resolution kind for *ref*.

    Returns one of ``vault`` | ``provider``.

    Raises:
        ValueError: if the ref is malformed, a ``secret:`` indirection, or
            otherwise unsupported.
    """
    s = (ref or "").strip()
    if not s:
        raise ValueError("empty ref")
    if len(s) > MAX_REF_LEN:
        raise ValueError(f"ref exceeds {MAX_REF_LEN} bytes")
    if parse_vault_ref(s):
        return "vault"
    if _PROVIDER_REF_RE.match(s):
        return "provider"
    if _OP_RE.match(s) or _BW_RE.match(s):
        return "provider"
    if _DOMAIN_REF_RE.match(s):
        raise ValueError(
            "secret:DOMAIN/NAME indirection is not allowed in the Secret Domain "
            "(chains of references are rejected; point at a vault/provider ref)"
        )
    raise ValueError(
        f"unsupported ref format: {s!r}. Expected vault:NAME, vault:VAULT/NAME, "
        "provider:1password:<op://...>, provider:bitwarden:<id>, op://..., or bw://..."
    )


def _shannon_entropy(s: str) -> float:
    import math
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_high_entropy(token: str) -> bool:
    """Conservative heuristic for raw secrets pasted as a single token.

    Long mixed-case single tokens with decent entropy are almost always
    base64/hex/raw key material, never Secret Domain content. Lowercase
    or English-identifier shapes are intentionally not flagged.
    """
    if len(token) < 32:
        return False
    if re.fullmatch(r"[A-Za-z0-9+_=-]+", token) is None:
        return False
    if not (any(ch.islower() for ch in token) and any(ch.isupper() for ch in token)):
        return False
    return _shannon_entropy(token) >= 4.25


def _check_no_credentials(check_entropy: bool = True, **fields: str) -> None:
    """Reject credential-like material in any metadata field."""
    for label, value in fields.items():
        if not value:
            continue
        matches = detect_credentials(str(value))
        if matches:
            names = ", ".join(m.pattern_name for m in matches)
            raise ValueError(
                f"credential-like content detected in {label} ({names}); "
                "the Secret Domain stores references only — put raw values in the vault"
            )
        if not check_entropy:
            continue
        for token in str(value).split():
            if _looks_high_entropy(token):
                raise ValueError(
                    f"credential-like high-entropy token detected in {label}; "
                    "the Secret Domain stores references only — put raw values in the vault"
                )


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("name is required")
    if len(name) > MAX_NAME_LEN or not _NAME_RE.match(name):
        raise ValueError(
            f"invalid name {name!r}: 1-128 chars, start with a letter/digit, "
            "then letters/digits/._-"
        )
    return name


def _encrypt_blob(plaintext: bytes, kek: bytes, kdf_byte: int, salt: bytes) -> bytes:
    """Encrypt plaintext with AES-256-GCM. Returns magic + kdf + salt + nonce + ct."""
    aesgcm = AESGCM(kek)
    nonce = generate_nonce()
    ct = aesgcm.encrypt(nonce, plaintext, _SECRET_AAD)
    return _SECRET_MAGIC + bytes([kdf_byte]) + salt + nonce + ct


def _decrypt_blob(blob: bytes, kek: bytes) -> bytes:
    """Decrypt an encrypted domain file. Raises DecryptionError on failure."""
    if not blob.startswith(_SECRET_MAGIC):
        raise DecryptionError("not an encrypted secret domain file")
    nonce = blob[21:33]
    ct = blob[33:]
    aesgcm = AESGCM(kek)
    try:
        return aesgcm.decrypt(nonce, ct, _SECRET_AAD)
    except Exception as exc:
        raise DecryptionError("secret domain decryption failed: wrong key or corrupted file") from exc


# ── SecretDomain ─────────────────────────────────────────────────────────────

class SecretDomain:
    """Encrypted-capable registry of secret references (metadata only)."""

    def __init__(self, path, name: str, encrypted: bool = False,
                 passphrase: Optional[str] = None, created: Optional[str] = None):
        self.path = Path(path)
        self.name = name
        self._encrypted = encrypted
        self._passphrase = passphrase
        self._created = created or _now()
        self._entries: list[Entry] = []
        self._meta: dict[str, int] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def create(cls, path, name: str = "default", *, encrypted: bool = True,
               passphrase: Optional[str] = None) -> "SecretDomain":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if encrypted:
            pw = passphrase or os.environ.get("SQAC_SECRET_PASSPHRASE") or None
            if pw is None and keychain_get(service="sqac-secret", username=name) is None:
                # Generate and store a master key up front (keychain mode).
                try:
                    keychain_set(service="sqac-secret", username=name,
                                 value=secrets.token_bytes(32).hex())
                except Exception as exc:
                    raise DecryptionError(
                        "keychain unavailable and no passphrase provided; "
                        "set SQAC_SECRET_PASSPHRASE or use portable mode"
                    ) from exc
            passphrase = pw
        domain = cls(path, name, encrypted=encrypted, passphrase=passphrase)
        domain._save()
        return domain

    @classmethod
    def open(cls, path, passphrase: Optional[str] = None) -> "SecretDomain":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"secret domain not found: {path}")
        raw = path.read_bytes()
        encrypted = raw.startswith(_SECRET_MAGIC)
        if encrypted:
            pw = passphrase or os.environ.get("SQAC_SECRET_PASSPHRASE")
            tmp_domain = cls(path, path.stem, encrypted=True, passphrase=pw)
            plaintext = tmp_domain._decrypt_file(pw)
            fd, tmp = tempfile.mkstemp(suffix=".sqac")
            os.close(fd)
            try:
                Path(tmp).write_bytes(plaintext)
                cart = read_cartridge(tmp)
            finally:
                Path(tmp).unlink(missing_ok=True)
        else:
            cart = read_cartridge(str(path))
        h = cart.header
        info = h.extra.get("secret_domain", {}) or {}
        domain = cls(
            path,
            info.get("name", path.stem),
            encrypted=encrypted,
            passphrase=passphrase or os.environ.get("SQAC_SECRET_PASSPHRASE"),
            created=info.get("created"),
        )
        domain._entries = list(cart.entries)
        domain._reindex()
        return domain

    # -- metadata --------------------------------------------------------

    def _header(self) -> CartridgeHeader:
        return CartridgeHeader(
            version=WRITE_VERSION,
            dims=1024,
            extra={
                "cartridge_type": "secret",
                "sync_policy": "never",
                "secret_domain": {
                    "schema": SCHEMA_VERSION,
                    "name": self.name,
                    "created": self._created,
                    "updated": _now(),
                },
                "encrypted": self._encrypted,
            },
        )

    def _reindex(self) -> None:
        self._meta.clear()
        for i, e in enumerate(self._entries):
            p = e.payload if isinstance(e.payload, dict) else {}
            self._meta[p.get("name", "")] = i

    def _find(self, name: str) -> Optional[int]:
        idx = self._meta.get(name)
        if idx is not None and idx < len(self._entries):
            return idx
        for i, e in enumerate(self._entries):
            p = e.payload if isinstance(e.payload, dict) else {}
            if p.get("name") == name:
                self._meta[name] = i
                return i
        return None

    def _get_kek(self, passphrase: Optional[str]) -> tuple[bytes, int, bytes]:
        """Return (kek, kdf_byte, salt). kdf_byte: 0=keychain, 1=argon2id."""
        pw = passphrase or self._passphrase
        if pw is not None:
            salt = os.urandom(16)
            kek = derive_kek_argon2id(pw, salt)
            return kek, 1, salt
        hex_key = keychain_get(service="sqac-secret", username=self.name)
        if hex_key is None:
            raise DecryptionError(
                "secret domain locked: no keychain key and no passphrase; "
                "set SQAC_SECRET_PASSPHRASE"
            )
        salt = os.urandom(16)
        return bytes.fromhex(hex_key), 0, salt

    def _decrypt_file(self, passphrase: Optional[str]) -> bytes:
        raw = self.path.read_bytes()
        if not raw.startswith(_SECRET_MAGIC):
            return raw
        kdf_byte = raw[4]
        salt = raw[5:21]
        if kdf_byte == 0:
            hex_key = keychain_get(service="sqac-secret", username=self.name)
            if hex_key is None:
                pw = passphrase or self._passphrase or os.environ.get("SQAC_SECRET_PASSPHRASE")
                if pw is None:
                    raise DecryptionError(
                        "secret domain locked: no keychain key and no passphrase"
                    )
                kek = derive_kek_argon2id(pw, salt)
            else:
                kek = bytes.fromhex(hex_key)
        else:
            pw = passphrase or self._passphrase or os.environ.get("SQAC_SECRET_PASSPHRASE")
            if pw is None:
                raise DecryptionError("secret domain locked: passphrase required")
            kek = derive_kek_argon2id(pw, salt)
        return _decrypt_blob(raw, kek)

    def _save(self) -> None:
        hdr = self._header()
        plaintext = _serialize(hdr, self._entries)
        if self._encrypted:
            kek, kdf_byte, salt = self._get_kek(self._passphrase)
            self.path.write_bytes(_encrypt_blob(plaintext, kek, kdf_byte, salt))
        else:
            self.path.write_bytes(plaintext)

    # -- CRUD (reference-only) -------------------------------------------

    def put_ref(self, name: str, ref: str, description: str = "",
                tags: Optional[list[str]] = None,
                scopes: Optional[list[str]] = None) -> dict:
        """Store or replace a secret *reference* (never a value)."""
        name = _validate_name(name)
        ref = (ref or "").strip()
        kind = classify_ref(ref)  # raises on bad format / secret: chains
        tags = list(tags or [])
        scopes = list(scopes or [])
        if len(description) > MAX_FIELD_LEN:
            raise ValueError(f"description exceeds {MAX_FIELD_LEN} bytes")
        _check_no_credentials(check_entropy=False, name=name)
        _check_no_credentials(ref=ref, description=description,
                              tags=" ".join(tags), scopes=" ".join(scopes))

        now = _now()
        idx = self._find(name)
        if idx is not None:
            old = self._entries[idx].payload if isinstance(self._entries[idx].payload, dict) else {}
            created = old.get("created", now)
        else:
            created = now
        payload = {
            "name": name,
            "ref": ref,
            "ref_kind": kind,
            "description": description,
            "tags": tags,
            "scopes": scopes,
            "created": created,
            "updated": now,
        }
        entry = Entry(key_bits=_key_bits_for_name(name), payload=payload,
                      kind=0, encrypted=False)
        if idx is not None:
            self._entries[idx] = entry
        else:
            self._entries.append(entry)
            idx = len(self._entries) - 1
        self._meta[name] = idx
        self._save()
        append_audit(vault=f"secret:{self.name}", name=name, op="put", success=True, bytes_count=0)
        return {"name": name, "ref": ref, "ref_kind": kind,
                "description": description, "tags": tags, "scopes": scopes}

    def get_ref_meta(self, name: str) -> dict:
        idx = self._find(name)
        if idx is None:
            raise KeyError(f"secret ref not found: {name}")
        p = self._entries[idx].payload if isinstance(self._entries[idx].payload, dict) else {}
        append_audit(vault=f"secret:{self.name}", name=name, op="get", success=True, bytes_count=0)
        return {
            "name": p.get("name"),
            "ref": p.get("ref"),
            "ref_kind": p.get("ref_kind"),
            "description": p.get("description", ""),
            "tags": p.get("tags", []),
            "scopes": p.get("scopes", []),
            "created": p.get("created"),
            "updated": p.get("updated"),
        }

    def list(self) -> list[dict]:
        out = []
        for e in self._entries:
            p = e.payload if isinstance(e.payload, dict) else {}
            out.append({
                "name": p.get("name"),
                "ref": p.get("ref"),
                "ref_kind": p.get("ref_kind"),
                "description": p.get("description", ""),
                "tags": p.get("tags", []),
                "scopes": p.get("scopes", []),
            })
        return out

    def remove(self, name: str) -> bool:
        idx = self._find(name)
        if idx is None:
            return False
        del self._entries[idx]
        self._reindex()
        self._save()
        append_audit(vault=f"secret:{self.name}", name=name, op="remove", success=True, bytes_count=0)
        return True

    def rename(self, old: str, new: str) -> bool:
        new = _validate_name(new)
        idx = self._find(old)
        if idx is None:
            return False
        if self._find(new) is not None:
            raise ValueError(f"secret ref already exists: {new}")
        p = self._entries[idx].payload if isinstance(self._entries[idx].payload, dict) else {}
        p = dict(p)
        p["name"] = new
        p["updated"] = _now()
        self._entries[idx] = Entry(key_bits=_key_bits_for_name(new), payload=p,
                                   kind=0, encrypted=False)
        self._reindex()
        self._save()
        append_audit(vault=f"secret:{self.name}", name=f"{old}->{new}", op="rename", success=True, bytes_count=0)
        return True


def _serialize(hdr: CartridgeHeader, entries: list[Entry]) -> bytes:
    """Serialize a header + entries to cartridge bytes via a temp file."""
    fd, tmp = tempfile.mkstemp(suffix=".sqac")
    os.close(fd)
    try:
        write_cartridge(tmp, hdr, {}, entries)
        return Path(tmp).read_bytes()
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── module-level helpers ─────────────────────────────────────────────────────

def domain_path(name: str) -> Path:
    return SECRET_DIR / f"{name}.sqac"


def create_domain(name: str = "default", *, encrypted: bool = True,
                  passphrase: Optional[str] = None) -> SecretDomain:
    path = domain_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SecretDomain.create(path, name, encrypted=encrypted, passphrase=passphrase)


def open_domain(name: str = "default", passphrase: Optional[str] = None) -> SecretDomain:
    path = domain_path(name)
    if not path.exists():
        raise FileNotFoundError(f"secret domain not found: {name!r} (run: sqac secret init {name})")
    return SecretDomain.open(path, passphrase=passphrase)


def list_domains() -> list[str]:
    if not SECRET_DIR.exists():
        return []
    return sorted(p.stem for p in SECRET_DIR.glob("*.sqac"))

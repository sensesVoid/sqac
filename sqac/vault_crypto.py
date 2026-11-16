"""Vault crypto primitives: AES-256-GCM, Argon2id, keychain wrappers."""

from __future__ import annotations

import base64
import secrets
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id


class VaultCryptoError(Exception):
    """Raised when vault encryption/decryption fails."""


class KeychainUnavailable(Exception):
    """Raised when the OS keychain cannot be accessed."""


# ── key generation ─────────────────────────────────────────────────────────

def generate_dek() -> bytes:
    """Generate a random 32-byte data encryption key."""
    return secrets.token_bytes(32)


def generate_nonce() -> bytes:
    """Generate a random 12-byte AES-GCM nonce."""
    return secrets.token_bytes(12)


# ── AAD ────────────────────────────────────────────────────────────────────

def aad(vault: str, name: str, entry_type: str) -> bytes:
    """Build the AAD for a vault entry."""
    return f"sqac-vault-v1|{vault}|{name}|{entry_type}".encode()


# ── value encrypt/decrypt ──────────────────────────────────────────────────

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def encrypt_value(dek: bytes, plaintext: bytes, *, vault: str, name: str, entry_type: str) -> tuple[str, str]:
    """Encrypt a value with AES-256-GCM. Returns (nonce_b64, ct_b64)."""
    aesgcm = AESGCM(dek)
    nonce = generate_nonce()
    ct = aesgcm.encrypt(nonce, plaintext, aad(vault, name, entry_type))
    return _b64(nonce), _b64(ct)


def decrypt_value(dek: bytes, nonce_b64: str, ct_b64: str, *, vault: str, name: str, entry_type: str) -> bytes:
    """Decrypt a value. Raises VaultCryptoError on failure."""
    aesgcm = AESGCM(dek)
    try:
        return aesgcm.decrypt(_unb64(nonce_b64), _unb64(ct_b64), aad(vault, name, entry_type))
    except (InvalidTag, Exception) as exc:
        raise VaultCryptoError(f"decryption failed for {name!r}: {exc}") from exc


# ── DEK wrap/unwrap ────────────────────────────────────────────────────────

def wrap_dek(dek: bytes, kek: bytes) -> tuple[str, str]:
    """Wrap a DEK with a KEK using AES-256-GCM. Returns (nonce_b64, wrapped_b64)."""
    aesgcm = AESGCM(kek)
    nonce = generate_nonce()
    wrapped = aesgcm.encrypt(nonce, dek, b"sqac-vault-dek-wrap")
    return _b64(nonce), _b64(wrapped)


def unwrap_dek(wrap_nonce_b64: str, wrapped_b64: str, kek: bytes) -> bytes:
    """Unwrap a DEK. Raises VaultCryptoError on failure."""
    aesgcm = AESGCM(kek)
    try:
        return aesgcm.decrypt(_unb64(wrap_nonce_b64), _unb64(wrapped_b64), b"sqac-vault-dek-wrap")
    except (InvalidTag, Exception) as exc:
        raise VaultCryptoError(f"DEK unwrap failed: {exc}") from exc


# ── Argon2id KDF ───────────────────────────────────────────────────────────

def derive_kek_argon2id(
    passphrase: str,
    salt: bytes,
    time_cost: int = 3,
    memory_cost: int = 65536,
    parallelism: int = 4,
) -> bytes:
    """Derive a 32-byte KEK from a passphrase using Argon2id."""
    kdf = Argon2id(
        salt=salt,
        length=32,
        iterations=time_cost,
        lanes=parallelism,
        memory_cost=memory_cost,
    )
    return kdf.derive(passphrase.encode("utf-8"))


# ── keychain wrappers ──────────────────────────────────────────────────────

def _import_keyring():
    try:
        import keyring
        return keyring
    except ImportError as exc:
        raise KeychainUnavailable("keyring not installed") from exc


def keychain_get(service: str = "sqac-vault", username: str = "default") -> Optional[str]:
    """Get a master key from the OS keychain. Returns hex string or None."""
    kr = _import_keyring()
    try:
        return kr.get_password(service, username)
    except Exception as exc:
        raise KeychainUnavailable(f"keychain get failed: {exc}") from exc


def keychain_set(service: str = "sqac-vault", username: str = "default", value: str = "") -> None:
    """Store a master key in the OS keychain."""
    kr = _import_keyring()
    try:
        kr.set_password(service, username, value)
    except Exception as exc:
        raise KeychainUnavailable(f"keychain set failed: {exc}") from exc


def keychain_delete(service: str = "sqac-vault", username: str = "default") -> None:
    """Delete a master key from the OS keychain."""
    kr = _import_keyring()
    try:
        kr.delete_password(service, username)
    except Exception as exc:
        raise KeychainUnavailable(f"keychain delete failed: {exc}") from exc


# ── unlock cache ───────────────────────────────────────────────────────────

class VaultUnlockCache:
    """In-memory cache for unwrapped DEKs, keyed by (path, mtime)."""

    def __init__(self):
        self._cache: dict[tuple[str, float], bytes] = {}

    def get(self, path: str, mtime: float) -> Optional[bytes]:
        return self._cache.get((path, mtime))

    def put(self, path: str, mtime: float, dek: bytes) -> None:
        self._cache[(path, mtime)] = dek

    def clear(self, path: Optional[str] = None) -> None:
        if path is None:
            self._cache.clear()
        else:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != path}


# module-level singleton
_unlock_cache = VaultUnlockCache()


def get_unlock_cache() -> VaultUnlockCache:
    return _unlock_cache

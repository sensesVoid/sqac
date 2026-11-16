"""Agent cartridge: persona + working state with typed sections."""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Optional

from .format import (
    CartridgeHeader,
    WRITE_VERSION,
    read_cartridge,
    write_cartridge,
)
from .store import SqacStore, Hit
from .vault_crypto import (
    VaultCryptoError as DecryptionError,
    generate_nonce,
    generate_dek,
    wrap_dek,
    unwrap_dek,
    derive_kek_argon2id,
    keychain_get,
    keychain_set,
    keychain_delete,
    _b64,
    _unb64,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Magic bytes for encrypted agent cartridges
_AGENT_MAGIC = b"SQAE"
_AGENT_AAD = b"sqac-agent-v1"


def _encrypt_blob(plaintext: bytes, kek: bytes, kdf_byte: int, salt: bytes) -> bytes:
    """Encrypt plaintext with AES-256-GCM. Returns magic + kdf + salt + nonce + ct."""
    aesgcm = AESGCM(kek)
    nonce = generate_nonce()
    ct = aesgcm.encrypt(nonce, plaintext, _AGENT_AAD)
    return _AGENT_MAGIC + bytes([kdf_byte]) + salt + nonce + ct


def _decrypt_blob(blob: bytes, kek: bytes) -> bytes:
    """Decrypt an encrypted agent cartridge. Raises DecryptionError on failure."""
    if not blob.startswith(_AGENT_MAGIC):
        raise DecryptionError("not an encrypted agent cartridge")
    off = len(_AGENT_MAGIC)
    kdf_byte = blob[off]
    off += 1
    salt = blob[off:off + 16]
    off += 16
    nonce = blob[off:off + 12]
    off += 12
    ct = blob[off:]
    aesgcm = AESGCM(kek)
    try:
        return aesgcm.decrypt(nonce, ct, _AGENT_AAD)
    except Exception as exc:
        raise DecryptionError(f"agent decryption failed: {exc}") from exc


def _derive_kek_passphrase(passphrase: str, salt: bytes) -> bytes:
    from .vault_crypto import derive_kek_argon2id
    return derive_kek_argon2id(passphrase, salt)


# Constants
AGENT_DIR = Path.home() / ".sqacm" / "agents"
ACTIVE_AGENT_PATH = Path.home() / ".sqacm" / "agent_active"
SINGLETON_SECTIONS = {"identity", "instruction", "preference"}
LIST_SECTIONS = {"goal", "decision", "note"}
ALL_SECTIONS = SINGLETON_SECTIONS | LIST_SECTIONS
DEFAULT_BUDGET = 4000


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gen_key(section: str) -> str:
    return f"{section}-{secrets.token_hex(4)}"


# â”€â”€ AgentCartridge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AgentCartridge:
    """Persona + working state cartridge."""

    def __init__(self, path, name: str, role: str = "", semantic: bool = False,
                 encrypted: bool = False, passphrase: Optional[str] = None,
                 model_profile_id: Optional[str] = None,
                 secret_ref: Optional[str] = None,
                 secret_refs: Optional[list[str]] = None,
                 allowed_tools: Optional[list[str]] = None,
                 sync_policy: str = "never",
                 created: Optional[str] = None):
        self.path = Path(path)
        self.name = name
        self.role = role
        self.semantic = semantic
        self._encrypted = encrypted
        self._passphrase = passphrase
        self.model_profile_id = model_profile_id
        self.secret_ref = secret_ref
        self.secret_refs = list(secret_refs or [])
        if secret_ref and secret_ref not in self.secret_refs:
            self.secret_refs = [secret_ref] + self.secret_refs
        self.allowed_tools = list(allowed_tools or [])
        self.sync_policy = sync_policy or "never"
        self._created = created or _now()
        self._store: Optional[SqacStore] = None

    @classmethod
    def create(cls, path, name: str, role: str = "", semantic: bool = False,
               encrypt: bool = False, passphrase: Optional[str] = None,
               model_profile_id: Optional[str] = None,
               secret_ref: Optional[str] = None,
               secret_refs: Optional[list[str]] = None,
               allowed_tools: Optional[list[str]] = None,
               sync_policy: str = "never") -> "AgentCartridge":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        agent = cls(path, name, role, semantic, encrypted=encrypt,
                    passphrase=passphrase, model_profile_id=model_profile_id,
                    secret_ref=secret_ref, secret_refs=secret_refs,
                    allowed_tools=allowed_tools, sync_policy=sync_policy)
        # Create empty cartridge with agent header
        hdr = CartridgeHeader(
            version=WRITE_VERSION,
            dims=1024,
            extra={
                "cartridge_type": "agent",
                "sync_policy": agent.sync_policy,
                "agent": {
                    "schema": 2,
                    "name": name,
                    "role": role,
                    "model_profile_id": model_profile_id,
                    "secret_ref": secret_ref,
                    "secret_refs": list(agent.secret_refs),
                    "allowed_tools": list(agent.allowed_tools),
                    "created": _now(),
                    "updated": _now(),
                },
                "encrypted": encrypt,
            },
        )
        write_cartridge(str(path), hdr, {}, [])
        agent._store = SqacStore(semantic=semantic)
        agent._store._description = f"agent:{name}"
        # Save with agent header by re-writing
        agent._save()
        return agent

    @classmethod
    def load(cls, path, semantic: bool = False,
             passphrase: Optional[str] = None) -> "AgentCartridge":
        path = Path(path)
        raw = path.read_bytes()
        encrypted = raw.startswith(_AGENT_MAGIC)
        if encrypted:
            import tempfile
            pw = passphrase or __import__("os").environ.get("SQAC_AGENT_PASSPHRASE")
            name = path.stem
            tmp_agent = cls(path, name, "", semantic, encrypted=True, passphrase=pw)
            plaintext = tmp_agent._decrypt_file(pw)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sqac")
            import os as _os
            _os.close(tmp_fd)
            Path(tmp_path).write_bytes(plaintext)
            try:
                cart = read_cartridge(tmp_path)
                tmp_fd2, tmp_path2 = tempfile.mkstemp(suffix=".sqac")
                _os.close(tmp_fd2)
                Path(tmp_path2).write_bytes(plaintext)
                try:
                    store = SqacStore.load(tmp_path2)
                finally:
                    Path(tmp_path2).unlink(missing_ok=True)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            h = cart.header
            agent_info = h.extra.get("agent", {}) or {}
            name = agent_info.get("name", path.stem)
            role = agent_info.get("role", "")
            secret_ref = agent_info.get("secret_ref")
            secret_refs = list(agent_info.get("secret_refs") or [])
            if secret_ref and secret_ref not in secret_refs:
                secret_refs = [secret_ref] + secret_refs
            agent = cls(
                path, name, role, semantic, encrypted=True, passphrase=pw,
                model_profile_id=agent_info.get("model_profile_id"),
                secret_ref=secret_ref,
                secret_refs=secret_refs,
                allowed_tools=list(agent_info.get("allowed_tools") or []),
                sync_policy=(h.extra.get("sync_policy") or "never"),
                created=agent_info.get("created"),
            )
            agent._store = store
            agent._store._description = f"agent:{name}"
            return agent

        cart = read_cartridge(str(path))
        h = cart.header
        agent_info = h.extra.get("agent", {}) or {}
        name = agent_info.get("name", path.stem)
        role = agent_info.get("role", "")
        secret_ref = agent_info.get("secret_ref")
        secret_refs = list(agent_info.get("secret_refs") or [])
        if secret_ref and secret_ref not in secret_refs:
            secret_refs = [secret_ref] + secret_refs
        agent = cls(
            path, name, role, semantic,
            model_profile_id=agent_info.get("model_profile_id"),
            secret_ref=secret_ref,
            secret_refs=secret_refs,
            allowed_tools=list(agent_info.get("allowed_tools") or []),
            sync_policy=(h.extra.get("sync_policy") or "never"),
            created=agent_info.get("created"),
        )
        agent._store = SqacStore.load(str(path))
        agent._store._description = f"agent:{name}"
        return agent

    def _get_kek(self, passphrase: Optional[str] = None) -> tuple[bytes, int, bytes]:
        """Return (kek, kdf_byte, salt). kdf_byte: 0=keychain, 1=argon2id."""
        pw = passphrase or self._passphrase
        if pw is not None:
            salt = secrets.token_bytes(16)
            kek = derive_kek_argon2id(pw, salt)
            return kek, 1, salt
        # keychain mode
        hex_key = keychain_get(service="sqac-agent", username=self.name)
        if hex_key is None:
            # generate and store
            master_key = secrets.token_bytes(32)
            keychain_set(service="sqac-agent", username=self.name, value=master_key.hex())
            hex_key = master_key.hex()
        salt = secrets.token_bytes(16)
        return bytes.fromhex(hex_key), 0, salt

    def _encrypt_file(self, passphrase: Optional[str] = None) -> None:
        """Encrypt the cartridge file in place."""
        raw = self.path.read_bytes()
        kek, kdf_byte, salt = self._get_kek(passphrase)
        encrypted = _encrypt_blob(raw, kek, kdf_byte, salt)
        self.path.write_bytes(encrypted)

    def _decrypt_file(self, passphrase: Optional[str] = None) -> bytes:
        """Read and decrypt the cartridge file. Returns plaintext bytes."""
        raw = self.path.read_bytes()
        if not raw.startswith(_AGENT_MAGIC):
            return raw  # not encrypted
        # Need to determine kdf and get kek
        kdf_byte = raw[4]
        salt = raw[5:21]
        if kdf_byte == 0:
            # keychain mode
            hex_key = keychain_get(service="sqac-agent", username=self.name)
            if hex_key is None:
                if passphrase is None:
                    raise DecryptionError("agent locked: no keychain key and no passphrase")
                # fall back to passphrase
                kek = derive_kek_argon2id(passphrase, salt)
            else:
                kek = bytes.fromhex(hex_key)
        else:
            pw = passphrase or self._passphrase
            if pw is None:
                pw = None
                import os as _os
                pw = _os.environ.get("SQAC_AGENT_PASSPHRASE")
            if pw is None:
                raise DecryptionError("agent locked: passphrase required")
            kek = derive_kek_argon2id(pw, salt)
        return _decrypt_blob(raw, kek)

    def _save(self) -> None:
        self.store.save(str(self.path), name=self.name, description=f"agent:{self.name}")
        cart = read_cartridge(str(self.path))
        cart.header.extra["cartridge_type"] = "agent"
        cart.header.extra["sync_policy"] = self.sync_policy or "never"
        cart.header.extra["agent"] = {
            "schema": 2,
            "name": self.name,
            "role": self.role,
            "model_profile_id": self.model_profile_id,
            "secret_ref": self.secret_ref,
            "secret_refs": list(self.secret_refs),
            "allowed_tools": list(self.allowed_tools),
            "created": self._created,
            "updated": _now(),
        }
        cart.header.extra["encrypted"] = self._encrypted
        write_cartridge(str(self.path), cart.header, {}, cart.entries)
        if self._encrypted:
            self._encrypt_file(self._passphrase)

    @property
    def store(self) -> SqacStore:
        if self._store is None:
            if self._encrypted and self.path.read_bytes().startswith(_AGENT_MAGIC):
                plaintext = self._decrypt_file(self._passphrase)
                import tempfile, os as _os
                fd, tmp = tempfile.mkstemp(suffix=".sqac")
                _os.close(fd)
                Path(tmp).write_bytes(plaintext)
                try:
                    self._store = SqacStore.load(tmp)
                finally:
                    Path(tmp).unlink(missing_ok=True)
            else:
                self._store = SqacStore.load(str(self.path))
            self._store._description = f"agent:{self.name}"
        return self._store

    def set_binding(self, model_profile_id: Optional[str] = None,
                    secret_ref: Optional[str] = None,
                    secret_refs: Optional[list[str]] = None,
                    allowed_tools: Optional[list[str]] = None,
                    sync_policy: Optional[str] = None) -> None:
        """Update the agent's schema-v2 binding fields and persist.

        ``secret_ref`` (single primary ref) is kept consistent with
        ``secret_refs`` (the full allowlist). If only ``secret_ref`` is
        given, it is added to the allowlist; if both are given they must
        be consistent; if a new allowlist excludes the current primary,
        the primary is cleared (fail closed).
        """
        if model_profile_id is not None:
            self.model_profile_id = model_profile_id
        if secret_ref is not None:
            self.secret_ref = secret_ref
        if secret_refs is not None:
            self.secret_refs = list(secret_refs)
        if self.secret_ref and self.secret_ref not in self.secret_refs:
            if secret_refs is None:
                # secret_ref given without refs: derive the allowlist
                self.secret_refs = [self.secret_ref] + self.secret_refs
            elif secret_ref is not None:
                # both given and inconsistent: reject
                raise ValueError(
                    f"secret_ref {self.secret_ref!r} must be a member of secret_refs")
            else:
                # new allowlist excludes the current primary: drop it
                # (fail closed — never keep a primary that is not allowed)
                self.secret_ref = None
        if allowed_tools is not None:
            self.allowed_tools = list(allowed_tools)
        if sync_policy is not None:
            if sync_policy not in ("write_through", "never", "local_only"):
                raise ValueError("sync_policy must be one of write_through|never|local_only")
            self.sync_policy = sync_policy
        self._save()

    def encrypt(self, passphrase: Optional[str] = None) -> None:
        """Encrypt this agent cartridge in place."""
        if self._encrypted:
            return
        self._passphrase = passphrase
        self._encrypted = True
        self._save()

    def decrypt(self, passphrase: Optional[str] = None) -> None:
        """Decrypt this agent cartridge in place."""
        if not self._encrypted:
            return
        self._passphrase = passphrase
        self._encrypted = False
        # Read current (encrypted) file, decrypt, write plaintext
        plaintext = self._decrypt_file(passphrase)
        self.path.write_bytes(plaintext)
        # Reload store
        self._store = None

    def rotate_key(self, old_passphrase: str, new_passphrase: str) -> None:
        """Rotate the encryption key."""
        if not self._encrypted:
            raise ValueError("agent is not encrypted")
        plaintext = self._decrypt_file(old_passphrase)
        self._passphrase = new_passphrase
        self._encrypted = True
        self.path.write_bytes(plaintext)
        self._store = None
        self._save()
    def set(self, section: str, content: str, key: Optional[str] = None, priority: int = 1, tags: Optional[list] = None) -> int:
        """Set an entry in a section. Returns entry index."""
        if section not in ALL_SECTIONS:
            raise ValueError(f"unknown section: {section}")
        if tags is None:
            tags = []

        if key is not None:
            # Upsert by key
            for i, e in enumerate(self.store._entries):
                if e.get("meta", {}).get("key") == key and e.get("meta", {}).get("section") == section:
                    self.store._entries[i]["content"] = content
                    self.store._entries[i]["meta"]["priority"] = priority
                    self.store._entries[i]["meta"]["tags"] = tags
                    self.store._entries[i]["meta"]["updated"] = _now()
                    self._save()
                    return i
            # Not found, add new
            key = key

        if key is None and section in SINGLETON_SECTIONS:
            # Replace single existing entry for this section
            for i, e in enumerate(self.store._entries):
                if e.get("meta", {}).get("section") == section:
                    self.store._entries[i]["content"] = content
                    self.store._entries[i]["meta"]["priority"] = priority
                    self.store._entries[i]["meta"]["tags"] = tags
                    self.store._entries[i]["meta"]["updated"] = _now()
                    self._save()
                    return i
            # No existing, generate key
            key = f"{section}-0"

        if key is None and section in LIST_SECTIONS:
            key = _gen_key(section)

        if key is None:
            key = f"{section}-0"

        idx = self.store.add(
            content,
            key=key,
            kind="fact",
            source="agent",
            meta={"section": section, "priority": priority, "status": "open", "tags": tags, "key": key, "updated": _now()},
        )
        self._save()
        return idx

    def note(self, content: str, tags: Optional[list] = None) -> int:
        return self.set("note", content, tags=tags)

    def complete(self, key_or_query: str) -> bool:
        """Mark a goal/decision as done. Returns True if resolved uniquely."""
        # Try exact key match first
        for i, e in enumerate(self.store._entries):
            if e.get("meta", {}).get("key") == key_or_query:
                self.store._entries[i]["meta"]["status"] = "done"
                self._save()
                return True
        # Try fuzzy search
        hits = self.store.search(key_or_query, top_k=5)
        # Filter to open goals/decisions
        candidates = [h for h in hits if h.meta.get("status") == "open" and h.meta.get("section") in ("goal", "decision")]
        if len(candidates) == 1:
            # Find the entry index
            for i, e in enumerate(self.store._entries):
                if e.get("content") == candidates[0].content:
                    self.store._entries[i]["meta"]["status"] = "done"
                    self._save()
                    return True
        return False

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        return self.store.search(query, top_k=top_k)

    def bootstrap(self, max_chars: int = DEFAULT_BUDGET) -> str:
        """Build a bootstrap packet for agent context."""
        parts = []
        used = 0

        # 1. Identity + instructions + preferences (all)
        for section in ["identity", "instruction", "preference"]:
            entries = [e for e in self.store._entries if e.get("meta", {}).get("section") == section]
            for e in entries:
                text = f"[{section}] {e['content']}"
                if used + len(text) + 2 > max_chars:
                    break
                parts.append(text)
                used += len(text) + 2

        # 2. Open goals by priority then recency
        goals = [e for e in self.store._entries if e.get("meta", {}).get("section") == "goal" and e.get("meta", {}).get("status") == "open"]
        goals.sort(key=lambda e: (-e.get("meta", {}).get("priority", 1), e.get("meta", {}).get("updated", "")), reverse=True)
        # Actually sort by priority desc, then recency desc
        goals.sort(key=lambda e: e.get("meta", {}).get("updated", ""), reverse=True)
        goals.sort(key=lambda e: e.get("meta", {}).get("priority", 1), reverse=True)
        for e in goals:
            text = f"[goal] {e['content']}"
            if used + len(text) + 2 > max_chars:
                break
            parts.append(text)
            used += len(text) + 2

        # 3. Last 5 decisions
        decisions = [e for e in self.store._entries if e.get("meta", {}).get("section") == "decision"]
        decisions.sort(key=lambda e: e.get("meta", {}).get("updated", ""), reverse=True)
        for e in decisions[:5]:
            text = f"[decision] {e['content']}"
            if used + len(text) + 2 > max_chars:
                break
            parts.append(text)
            used += len(text) + 2

        # 4. Notes only if budget remains
        notes = [e for e in self.store._entries if e.get("meta", {}).get("section") == "note"]
        for e in notes:
            text = f"[note] {e['content']}"
            if used + len(text) + 2 > max_chars:
                break
            parts.append(text)
            used += len(text) + 2

        header = f"## Agent: {self.name}"
        if self.role:
            header += f" ({self.role})"
        return header + "\n" + "\n".join(parts)

    def stats(self) -> dict:
        sections = {}
        for e in self.store._entries:
            sec = e.get("meta", {}).get("section", "unknown")
            sections[sec] = sections.get(sec, 0) + 1
        return {
            "name": self.name,
            "role": self.role,
            "entries": len(self.store),
            "sections": sections,
        }

    def save(self) -> None:
        self._save()


# â”€â”€ module-level helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _agent_path(name: str) -> Path:
    return AGENT_DIR / f"{name}.sqac"


def list_agents() -> list[str]:
    if not AGENT_DIR.exists():
        return []
    return sorted(p.stem for p in AGENT_DIR.glob("*.sqac"))


def get_active_agent() -> Optional[str]:
    if ACTIVE_AGENT_PATH.exists():
        return ACTIVE_AGENT_PATH.read_text(encoding="utf-8").strip()
    return None


def set_active_agent(name: str) -> None:
    ACTIVE_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_AGENT_PATH.write_text(name, encoding="utf-8")


def load_agent(name: str, semantic: bool = False) -> AgentCartridge:
    path = _agent_path(name)
    if not path.exists():
        raise FileNotFoundError(f"agent not found: {name}")
    return AgentCartridge.load(path, semantic=semantic)


def create_agent(name: str, role: str = "", semantic: bool = False,
                 model_profile_id: Optional[str] = None,
                 secret_ref: Optional[str] = None,
                 secret_refs: Optional[list[str]] = None,
                 allowed_tools: Optional[list[str]] = None,
                 sync_policy: str = "never") -> AgentCartridge:
    path = _agent_path(name)
    return AgentCartridge.create(path, name, role, semantic,
                                 model_profile_id=model_profile_id,
                                 secret_ref=secret_ref,
                                 secret_refs=secret_refs,
                                 allowed_tools=allowed_tools,
                                 sync_policy=sync_policy)

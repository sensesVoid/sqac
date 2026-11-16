"""Cloud sync engine for SQAC cartridges."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .storage import StorageBackend

# ── config ─────────────────────────────────────────────────────────────────

CLOUD_CONFIG_PATH = Path.home() / ".sqacm" / "cloud.json"
SYNC_STATE_PATH = Path.home() / ".sqacm" / ".sync-state.json"
CLOUD_LOG_PATH = Path.home() / ".sqacm" / "cloud.log"

# Sync allowlist (relative paths)
SYNC_ALWAYS = {"facts", "skills", "docs", "agents", "vaults"}
SYNC_OPTIN = {"continuity.json", "archives"}
SYNC_NEVER = {
    "session.sqac", "checkpoint.sqac", "vault-audit.jsonl",
    "cloud.json", ".sync-state.json", "cloud.log", "agent_active",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SyncObject:
    local_path: Path
    remote_key: str
    sha256: str
    etag: str = ""
    synced_at: float = 0.0
    status: str = "unknown"  # in_sync | local_ahead | remote_ahead | conflict


@dataclass
class SyncState:
    objects: dict[str, dict] = field(default_factory=dict)  # remote_key -> {sha256, etag, synced_at}

    @classmethod
    def load(cls, path: Path = SYNC_STATE_PATH) -> "SyncState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(objects=data.get("objects", {}))
        except Exception:
            return cls()

    def save(self, path: Path = SYNC_STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"objects": self.objects}, indent=2), encoding="utf-8")

    def get(self, key: str) -> dict | None:
        return self.objects.get(key)

    def set(self, key: str, sha256: str, etag: str) -> None:
        self.objects[key] = {"sha256": sha256, "etag": etag, "synced_at": time.time()}


# ── config ──────────────────────────────────────────────────────────────────

def load_cloud_config(path: Path = CLOUD_CONFIG_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cloud_config(config: dict, path: Path = CLOUD_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ── sync engine ────────────────────────────────────────────────────────────

class SyncEngine:
    """Push/pull/status sync for SQAC cartridges."""

    def __init__(self, backend: StorageBackend, sqac_dir: Path, config: dict):
        self.backend = backend
        self.sqac_dir = sqac_dir
        self.config = config
        self.state = SyncState.load()
        self.sync_set: set[str] = set(config.get("sync_set", [])) or SYNC_ALWAYS
        self.sync_continuity = config.get("sync_continuity", False)
        self.sync_archives = config.get("sync_archives", False)

    def _is_syncable(self, rel_path: str) -> bool:
        """Check if a relative path is in the sync allowlist."""
        parts = Path(rel_path).parts
        if not parts:
            return False
        top = parts[0]
        name = Path(rel_path).name

        # Never sync these
        if name in SYNC_NEVER:
            return False
        if name.endswith(".tmp") or name.endswith(".lock"):
            return False
        if name == "agent_active":
            return False

        # Always sync
        if top in SYNC_ALWAYS:
            return True

        # Opt-in
        if self.sync_continuity and rel_path == "continuity.json":
            return True
        if self.sync_archives and top == "archives":
            return True

        return False

    def _collect_local_objects(self) -> list[SyncObject]:
        """Collect all syncable local files."""
        objects = []
        for root, dirs, files in os.walk(self.sqac_dir):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                full = Path(root) / fname
                rel = full.relative_to(self.sqac_dir).as_posix()
                if not self._is_syncable(rel):
                    continue
                # Keychain-mode vault check
                if rel.startswith("vaults/") and self._is_keychain_vault(full):
                    continue
                sha = _sha256_file(full)
                objects.append(SyncObject(local_path=full, remote_key=f"v1/{rel}", sha256=sha))
        return objects

    def _is_keychain_vault(self, path: Path) -> bool:
        """Check if a vault uses keychain mode (local-only)."""
        try:
            from .format import read_cartridge
            cart = read_cartridge(str(path))
            vinfo = cart.header.extra.get("vault", {})
            return vinfo.get("kdf") == "keychain"
        except Exception:
            return False

    def status(self) -> list[SyncObject]:
        """Compare local, synced, and remote state."""
        local_objects = self._collect_local_objects()
        for obj in local_objects:
            synced = self.state.get(obj.remote_key)
            if synced is None:
                obj.status = "local_ahead"
                continue
            if synced["sha256"] == obj.sha256:
                # Check remote
                info = self.backend.head(obj.remote_key)
                if info is None:
                    obj.status = "local_ahead"
                elif info.etag == synced["etag"]:
                    obj.status = "in_sync"
                else:
                    obj.status = "remote_ahead"
            else:
                info = self.backend.head(obj.remote_key)
                if info is None:
                    obj.status = "local_ahead"
                elif info.etag == synced["etag"]:
                    obj.status = "local_ahead"
                else:
                    obj.status = "conflict"
            obj.etag = synced.get("etag", "")
            obj.synced_at = synced.get("synced_at", 0)
        return local_objects

    def push(self, names: list[str] | None = None, dry_run: bool = False,
            force_local: bool = False, force_remote: bool = False) -> list[dict]:
        """Push local changes to remote."""
        results = []
        objects = self._collect_local_objects()
        if names:
            # Filter to named cartridges (base name match)
            filtered = []
            for obj in objects:
                rel = obj.remote_key.replace("v1/", "", 1)
                base = Path(rel).stem
                if base in names or any(n in rel for n in names):
                    filtered.append(obj)
            objects = filtered

        for obj in objects:
            synced = self.state.get(obj.remote_key)
            if synced and synced["sha256"] == obj.sha256 and not force_local:
                # Already synced
                results.append({"key": obj.remote_key, "action": "skip", "reason": "already synced"})
                continue

            # Check remote state for conflicts
            info = self.backend.head(obj.remote_key)
            if info is not None and synced is not None and info.etag != synced.get("etag", "") and not force_local:
                if not force_remote:
                    results.append({"key": obj.remote_key, "action": "conflict", "reason": "remote changed since last sync"})
                    continue

            if dry_run:
                results.append({"key": obj.remote_key, "action": "would_push", "sha256": obj.sha256})
                continue

            # Upload
            data = obj.local_path.read_bytes()
            if_match = synced["etag"] if (synced and not force_local) else None
            if_none_match = (synced is None and not force_local)
            try:
                etag = self.backend.put(obj.remote_key, data, if_match=if_match, if_none_match=if_none_match)
                self.state.set(obj.remote_key, obj.sha256, etag)
                results.append({"key": obj.remote_key, "action": "pushed", "etag": etag})
            except ValueError as exc:
                if "412" in str(exc) or "Precondition" in str(exc):
                    results.append({"key": obj.remote_key, "action": "conflict", "reason": str(exc)})
                else:
                    results.append({"key": obj.remote_key, "action": "error", "reason": str(exc)})

        self.state.save()
        return results

    def pull(self, names: list[str] | None = None, dry_run: bool = False,
             force_local: bool = False, force_remote: bool = False) -> list[dict]:
        """Pull remote changes to local."""
        results = []
        remote_objects = self.backend.list("v1/")
        for info in remote_objects:
            key = info.key
            if not key.startswith("v1/"):
                continue
            rel = key[3:]  # strip "v1/"
            if not self._is_syncable(rel):
                continue
            local_path = self.sqac_dir / rel

            # Keychain vault check
            if rel.startswith("vaults/") and local_path.exists():
                if self._is_keychain_vault(local_path):
                    results.append({"key": key, "action": "skip", "reason": "keychain vault is local-only"})
                    continue

            synced = self.state.get(key)
            local_sha = _sha256_file(local_path) if local_path.exists() else None

            if local_sha is not None and synced and synced["sha256"] == local_sha and synced["etag"] == info.etag:
                results.append({"key": key, "action": "skip", "reason": "already in sync"})
                continue

            # Conflict check
            if local_sha is not None and synced and local_sha != synced["sha256"] and info.etag != synced.get("etag", ""):
                if not force_local and not force_remote:
                    results.append({"key": key, "action": "conflict", "reason": "both local and remote changed"})
                    continue

            if dry_run:
                results.append({"key": key, "action": "would_pull", "etag": info.etag})
                continue

            # Download
            data = self.backend.get(key)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            new_sha = hashlib.sha256(data).hexdigest()
            self.state.set(key, new_sha, info.etag)
            results.append({"key": key, "action": "pulled", "size": len(data)})

        self.state.save()
        return results


# ── cloud init ──────────────────────────────────────────────────────────────

def cloud_init(provider: str = "r2", bucket: str = "", prefix: str = "v1",
               endpoint_url: str | None = None, region: str | None = None,
               access_key_id: str | None = None, secret_access_key: str | None = None,
               account_id: str | None = None) -> dict:
    """Initialize cloud config. Returns config dict."""
    from .storage import _build_endpoint, PROVIDER_PRESETS

    preset = PROVIDER_PRESETS.get(provider, {})
    ep = endpoint_url or _build_endpoint(provider, account_id, endpoint_override=endpoint_url)
    reg = region or preset.get("region", "us-east-1")

    config = {
        "provider": provider,
        "endpoint_url": ep,
        "bucket": bucket,
        "prefix": prefix,
        "region": reg,
        "sync_mode": "off",
        "sync_continuity": False,
        "sync_archives": False,
        "exclude": [],
    }
    save_cloud_config(config)
    return config


def create_backend_from_config(config: dict) -> "StorageBackend":
    """Create a storage backend from cloud config."""
    from .storage import create_backend, _resolve_credentials

    provider = config.get("provider", "r2")
    bucket = config.get("bucket", "")
    prefix = config.get("prefix", "v1")
    endpoint_url = config.get("endpoint_url")
    region = config.get("region")

    access_key, secret_key, account_id = _resolve_credentials(provider)

    return create_backend(
        provider=provider,
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint_url,
        region=region,
        access_key_id=access_key,
        secret_access_key=secret_key,
        account_id=account_id,
    )

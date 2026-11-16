"""Storage backend abstraction for cloud sync (S3-compatible)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

# ── data classes ───────────────────────────────────────────────────────────

@dataclass
class ObjectInfo:
    key: str
    size: int
    etag: str
    last_modified: str = ""


# ── protocol ───────────────────────────────────────────────────────────────

@runtime_checkable
class StorageBackend(Protocol):
    def get(self, key: str) -> bytes: ...
    def put(self, key: str, data: bytes, *, if_match: str | None = None,
            if_none_match: bool = False) -> str: ...  # returns etag
    def head(self, key: str) -> ObjectInfo | None: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str = "") -> list[ObjectInfo]: ...


# ── provider presets ───────────────────────────────────────────────────────

PROVIDER_PRESETS = {
    "r2": {
        "endpoint_url": "https://{account_id}.r2.cloudflarestorage.com",
        "region": "auto",
        "signature_version": "s3v4",
    },
    "s3": {
        "endpoint_url": "https://s3.{region}.amazonaws.com",
        "region": "us-east-1",
        "signature_version": "s3v4",
    },
    "supabase": {
        "endpoint_url": "https://{project_ref}.supabase.co/storage/v1/s3",
        "region": "auto",
        "signature_version": "s3v4",
    },
    "minio": {
        "endpoint_url": None,  # user-provided
        "region": "us-east-1",
        "signature_version": "s3v4",
    },
    "b2": {
        "endpoint_url": "https://s3.{region}.backblazeb2.com",
        "region": "us-west-004",
        "signature_version": "s3v4",
    },
    "spaces": {
        "endpoint_url": "https://{region}.digitaloceanspaces.com",
        "region": "nyc3",
        "signature_version": "s3v4",
    },
}


# ── credential resolution ──────────────────────────────────────────────────

def _resolve_credentials(provider: str) -> tuple[str | None, str | None, str | None]:
    """Resolve (access_key_id, secret_access_key, account_id/project_ref) from env.

    Checks provider-specific aliases first, then generic SQAC_CLOUD_*.
    """
    # Provider-specific env var prefixes (in priority order)
    prefixes = {
        "r2": ["SQAC_R2", "SQAC_CLOUD"],
        "s3": ["AWS", "SQAC_S3", "SQAC_CLOUD"],
        "supabase": ["SUPABASE", "SQAC_CLOUD"],
        "minio": ["MINIO", "SQAC_CLOUD"],
        "b2": ["B2", "SQAC_CLOUD"],
        "spaces": ["DO", "SPACES", "SQAC_CLOUD"],
    }.get(provider, ["SQAC_CLOUD"])

    access_key = None
    secret_key = None
    account_id = None

    for prefix in prefixes:
        if access_key is None:
            access_key = os.environ.get(f"{prefix}_ACCESS_KEY_ID") or os.environ.get(f"{prefix}_ACCESS_KEY")
        if secret_key is None:
            secret_key = os.environ.get(f"{prefix}_SECRET_ACCESS_KEY") or os.environ.get(f"{prefix}_SECRET_KEY")
        if account_id is None:
            account_id = (
                os.environ.get(f"{prefix}_ACCOUNT_ID")
                or os.environ.get(f"{prefix}_PROJECT_REF")
                or os.environ.get(f"{prefix}_PROJECT_URL")
            )

    # Generic fallback
    if access_key is None:
        access_key = os.environ.get("SQAC_CLOUD_ACCESS_KEY_ID")
    if secret_key is None:
        secret_key = os.environ.get("SQAC_CLOUD_SECRET_ACCESS_KEY")
    if account_id is None:
        account_id = os.environ.get("SQAC_CLOUD_ACCOUNT_ID")

    return access_key, secret_key, account_id


def _build_endpoint(provider: str, account_id: str | None, endpoint_override: str | None = None) -> str:
    """Build the endpoint URL from provider preset and account info."""
    if endpoint_override:
        return endpoint_override

    preset = PROVIDER_PRESETS.get(provider, {})
    template = preset.get("endpoint_url")
    if template is None:
        raise ValueError(f"Provider {provider!r} requires an explicit endpoint_url")
    if "{account_id}" in template:
        if not account_id:
            raise ValueError(f"Provider {provider!r} requires an account_id")
        return template.format(account_id=account_id)
    if "{project_ref}" in template:
        if not account_id:
            raise ValueError(f"Provider {provider!r} requires a project_ref (account_id)")
        return template.format(project_ref=account_id)
    if "{region}" in template:
        region = preset.get("region", "us-east-1")
        return template.format(region=region)
    return template


# ── LocalBackend ───────────────────────────────────────────────────────────

class LocalBackend:
    """Local filesystem backend (for tests and offline use)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def get(self, key: str) -> bytes:
        p = self._path(key)
        if not p.exists():
            raise FileNotFoundError(f"object not found: {key}")
        return p.read_bytes()

    def put(self, key: str, data: bytes, *, if_match: str | None = None,
            if_none_match: bool = False) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Simple etag = md5 of content
        etag = hashlib.md5(data).hexdigest()
        if if_none_match and p.exists():
            raise FileExistsError(f"object already exists: {key}")
        if if_match is not None and p.exists():
            current_etag = hashlib.md5(p.read_bytes()).hexdigest()
            if current_etag != if_match:
                raise ValueError(f"etag mismatch for {key}: expected {if_match}, got {current_etag}")
        p.write_bytes(data)
        return etag

    def head(self, key: str) -> ObjectInfo | None:
        p = self._path(key)
        if not p.exists():
            return None
        data = p.read_bytes()
        return ObjectInfo(key=key, size=len(data), etag=hashlib.md5(data).hexdigest())

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def list(self, prefix: str = "") -> list[ObjectInfo]:
        results = []
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return results
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                if not rel.startswith(prefix):
                    continue
                data = p.read_bytes()
                results.append(ObjectInfo(key=rel, size=len(data), etag=hashlib.md5(data).hexdigest()))
        return results


# ── FakeBackend (tests, supports conditional put) ──────────────────────────

class FakeBackend:
    """In-memory backend for tests. Supports conditional put."""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._etags: dict[str, str] = {}

    def get(self, key: str) -> bytes:
        if key not in self._store:
            raise KeyError(f"object not found: {key}")
        return self._store[key]

    def put(self, key: str, data: bytes, *, if_match: str | None = None,
            if_none_match: bool = False) -> str:
        etag = hashlib.md5(data).hexdigest()
        if if_none_match and key in self._store:
            raise ValueError(f"object already exists: {key} (412 Precondition Failed)")
        if if_match is not None:
            current = self._etags.get(key)
            if current is not None and current != if_match:
                raise ValueError(f"etag mismatch for {key}: expected {if_match}, got {current} (412 Precondition Failed)")
        self._store[key] = data
        self._etags[key] = etag
        return etag

    def head(self, key: str) -> ObjectInfo | None:
        if key not in self._store:
            return None
        data = self._store[key]
        return ObjectInfo(key=key, size=len(data), etag=self._etags[key])

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._etags.pop(key, None)

    def list(self, prefix: str = "") -> list[ObjectInfo]:
        return [ObjectInfo(key=k, size=len(v), etag=self._etags[k])
                for k, v in self._store.items() if k.startswith(prefix)]


# ── S3Backend (provider-agnostic) ─────────────────────────────────────────

class S3Backend:
    """S3-compatible backend. Works with R2, AWS S3, Supabase, MinIO, B2, Spaces."""

    def __init__(
        self,
        provider: str = "r2",
        bucket: str = "",
        prefix: str = "",
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        account_id: str | None = None,
    ):
        self.provider = provider
        self.bucket = bucket
        self.prefix = prefix

        # Resolve credentials
        if access_key_id is None or secret_access_key is None:
            ak, sk, aid = _resolve_credentials(provider)
            if access_key_id is None:
                access_key_id = ak
            if secret_access_key is None:
                secret_access_key = sk
            if account_id is None:
                account_id = aid

        self.endpoint_url = _build_endpoint(provider, account_id, endpoint_override=endpoint_url)
        self.region = region or PROVIDER_PRESETS.get(provider, {}).get("region", "us-east-1")

        if not access_key_id or not secret_access_key:
            raise ValueError(
                f"No credentials found for provider {provider!r}. "
                f"Set SQAC_CLOUD_ACCESS_KEY_ID/SQAC_CLOUD_SECRET_ACCESS_KEY "
                f"or provider-specific env vars."
            )

        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError("boto3 is required for S3Backend: pip install boto3") from exc

        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name=self.region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def _full_key(self, key: str) -> str:
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{key}"
        return key

    def get(self, key: str) -> bytes:
        resp = self._client.get_object(Bucket=self.bucket, Key=self._full_key(key))
        return resp["Body"].read()

    def put(self, key: str, data: bytes, *, if_match: str | None = None,
            if_none_match: bool = False) -> str:
        fk = self._full_key(key)
        kwargs = {"Bucket": self.bucket, "Key": fk, "Body": data}
        if if_match is not None:
            kwargs["IfMatch"] = if_match
        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        resp = self._client.put_object(**kwargs)
        return resp.get("ETag", "").strip('"')

    def head(self, key: str) -> ObjectInfo | None:
        try:
            resp = self._client.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return ObjectInfo(
                key=key,
                size=resp.get("ContentLength", 0),
                etag=resp.get("ETag", "").strip('"'),
                last_modified=str(resp.get("LastModified", "")),
            )
        except Exception:
            return None

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=self._full_key(key))

    def list(self, prefix: str = "") -> list[ObjectInfo]:
        fk = self._full_key(prefix) if prefix else self._full_key("")
        resp = self._client.list_objects_v2(Bucket=self.bucket, Prefix=fk)
        results = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if self.prefix and key.startswith(self.prefix.rstrip("/") + "/"):
                key = key[len(self.prefix.rstrip("/") + "/"):]
            results.append(ObjectInfo(
                key=key,
                size=obj.get("Size", 0),
                etag=obj.get("ETag", "").strip('"'),
                last_modified=str(obj.get("LastModified", "")),
            ))
        return results


# ── factory ─────────────────────────────────────────────────────────────────

def create_backend(provider: str = "r2", **kwargs) -> StorageBackend:
    """Factory for creating storage backends."""
    if provider == "local":
        return LocalBackend(kwargs.get("root", "."))
    if provider == "fake":
        return FakeBackend()
    return S3Backend(provider=provider, **kwargs)

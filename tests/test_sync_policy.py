"""Sync policy enforcement tests (v1.2).

Every test uses a FakeBackend + tmp sqac_dir; the module-level state path is
redirected to tmp so the real ~/.sqacm is never touched.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from sqac import cloud
from sqac.cloud import SyncEngine, _effective_sync_policy
from sqac.storage import FakeBackend
from sqac.format import WRITE_VERSION, CartridgeHeader, write_cartridge


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    p = tmp_path / ".sync-state.json"
    monkeypatch.setattr(cloud, "SYNC_STATE_PATH", p)
    return p


def _make_cart(path: Path, cartridge_type: str | None, sync_policy: str | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if cartridge_type is not None:
        extra["cartridge_type"] = cartridge_type
    if sync_policy is not None:
        extra["sync_policy"] = sync_policy
    hdr = CartridgeHeader(version=WRITE_VERSION, dims=1024, extra=extra)
    write_cartridge(path, hdr, {}, [])


def _engine(tmp_path, state_path):
    backend = FakeBackend()
    cfg = {"sync_set": ["facts", "skills", "docs", "agents", "vaults"], "sync_continuity": False}
    return SyncEngine(backend, tmp_path, cfg), backend


# -- effective policy resolution ---------------------------------------------

def test_policy_default_agent_is_never(tmp_path):
    cart = tmp_path / "agents" / "a.sqac"
    cart.parent.mkdir(parents=True)
    _make_cart(cart, "agent", None)
    assert _effective_sync_policy(cart) == "never"


def test_policy_default_secret_is_never(tmp_path):
    cart = tmp_path / "secrets" / "s.sqac"
    cart.parent.mkdir(parents=True)
    _make_cart(cart, "secret", None)
    assert _effective_sync_policy(cart) == "never"


def test_policy_default_vault_is_never(tmp_path):
    cart = tmp_path / "vaults" / "v.sqac"
    cart.parent.mkdir(parents=True)
    _make_cart(cart, "vault", None)
    assert _effective_sync_policy(cart) == "never"


def test_policy_default_memory_is_write_through(tmp_path):
    cart = tmp_path / "facts" / "f.sqac"
    cart.parent.mkdir(parents=True)
    _make_cart(cart, None, None)
    assert _effective_sync_policy(cart) == "write_through"


def test_policy_explicit_write_through_agent(tmp_path):
    cart = tmp_path / "agents" / "a.sqac"
    cart.parent.mkdir(parents=True)
    _make_cart(cart, "agent", "write_through")
    assert _effective_sync_policy(cart) == "write_through"


def test_policy_local_only_treated_as_never(tmp_path):
    cart = tmp_path / "facts" / "f.sqac"
    cart.parent.mkdir(parents=True)
    _make_cart(cart, None, "local_only")
    assert _effective_sync_policy(cart) == "never"


def test_policy_corrupt_cartridge_fails_closed(tmp_path):
    cart = tmp_path / "facts" / "bad.sqac"
    cart.parent.mkdir(parents=True)
    cart.write_bytes(b"this is not a cartridge")
    assert _effective_sync_policy(cart) == "never"


# -- collection / push / pull -------------------------------------------------

def test_collect_excludes_agent_and_vault(tmp_path, state_path):
    eng, _ = _engine(tmp_path, state_path)
    _make_cart(tmp_path / "facts" / "f.sqac", None, None)
    _make_cart(tmp_path / "agents" / "jarvis.sqac", "agent", None)
    _make_cart(tmp_path / "vaults" / "v.sqac", "vault", None)
    _make_cart(tmp_path / "secrets" / "d.sqac", "secret", None)
    names = _collect_names(eng, tmp_path)
    assert "facts/f.sqac" in names
    assert "agents/jarvis.sqac" not in names
    assert "vaults/v.sqac" not in names
    assert "secrets/d.sqac" not in names


def _collect_names(eng, tmp_path):
    return {o.local_path.relative_to(tmp_path).as_posix() for o in eng._collect_local_objects()}


def test_push_skips_policy_denied(tmp_path, state_path):
    eng, backend = _engine(tmp_path, state_path)
    _make_cart(tmp_path / "agents" / "jarvis.sqac", "agent", None)
    _make_cart(tmp_path / "facts" / "ok.sqac", None, None)
    results = eng.push()
    keys = [r["key"] for r in results]
    assert "v1/facts/ok.sqac" in keys
    assert not any("agents" in k for k in keys)
    assert "v1/facts/ok.sqac" in backend._store
    assert not any("agents" in k for k in backend._store)


def test_push_allows_explicit_write_through_agent(tmp_path, state_path):
    eng, backend = _engine(tmp_path, state_path)
    _make_cart(tmp_path / "agents" / "jarvis.sqac", "agent", "write_through")
    results = eng.push()
    assert any(r["key"] == "v1/agents/jarvis.sqac" and r["action"] == "pushed" for r in results)


def test_pull_refuses_to_overwrite_policy_denied_local(tmp_path, state_path):
    eng, backend = _engine(tmp_path, state_path)
    local = tmp_path / "agents" / "jarvis.sqac"
    local.parent.mkdir(parents=True)
    _make_cart(local, "agent", None)
    original = local.read_bytes()
    # Remote claims a *different* cartridge at the same key (even forced)
    remote = tmp_path / "remote-src.sqac"
    _make_cart(remote, "agent", "write_through")
    backend.put("v1/agents/jarvis.sqac", remote.read_bytes())
    results = eng.pull(force_remote=True)
    assert any(r["key"] == "v1/agents/jarvis.sqac" and r["reason"] == "sync_policy_never" for r in results)
    assert local.read_bytes() == original


def test_pull_refuses_policy_denied_remote(tmp_path, state_path):
    eng, backend = _engine(tmp_path, state_path)
    remote = tmp_path / "remote-src.sqac"
    _make_cart(remote, "vault", None)  # vault -> never by default
    backend.put("v1/vaults/v.sqac", remote.read_bytes())
    results = eng.pull()
    assert any(r["key"] == "v1/vaults/v.sqac" and r["reason"] == "sync_policy_never" for r in results)
    assert not (tmp_path / "vaults" / "v.sqac").exists()


def test_pull_allows_write_through_memory_cartridge(tmp_path, state_path):
    eng, backend = _engine(tmp_path, state_path)
    remote = tmp_path / "remote-src.sqac"
    _make_cart(remote, None, None)
    payload = remote.read_bytes()
    backend.put("v1/facts/shared.sqac", payload)
    results = eng.pull()
    pulled = tmp_path / "facts" / "shared.sqac"
    assert pulled.exists()
    assert pulled.read_bytes() == payload
    assert any(r["key"] == "v1/facts/shared.sqac" and r["action"] == "pulled" for r in results)


def test_pull_denies_secret_domain_even_when_forced(tmp_path, state_path):
    eng, backend = _engine(tmp_path, state_path)
    remote = tmp_path / "remote-src.sqac"
    _make_cart(remote, "secret", "write_through")  # even explicit ALLOW is pointless: secrets/ never syncs
    backend.put("v1/secrets/d.sqac", remote.read_bytes())
    results = eng.pull(force_remote=True)
    assert any(r["key"] == "v1/secrets/d.sqac" and r["action"] == "skip" for r in results)
    assert not (tmp_path / "secrets" / "d.sqac").exists()


def test_pull_denies_non_sqac_material_in_sensitive_dir(tmp_path, state_path):
    eng, backend = _engine(tmp_path, state_path)
    backend.put("v1/agents/notes.txt", b"hello")
    results = eng.pull(force_remote=True)
    assert any(r.get("reason") in ("sensitive_directory_default_deny", "sync_policy_never") for r in results)
    assert not (tmp_path / "agents" / "notes.txt").exists()


def test_denied_objects_reports_policy_denied(tmp_path, state_path):
    eng, _ = _engine(tmp_path, state_path)
    _make_cart(tmp_path / "agents" / "a.sqac", "agent", None)
    _make_cart(tmp_path / "facts" / "ok.sqac", None, "write_through")
    denied = eng.denied_objects()
    paths = {d["path"] for d in denied}
    assert "agents/a.sqac" in paths
    assert "facts/ok.sqac" not in paths
    assert all(d["reason"] == "sync_policy_never" for d in denied)

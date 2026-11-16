"""External secret provider tests."""
from __future__ import annotations

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from sqac import providers


def test_available_providers_empty_when_no_clis(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert providers.available_providers() == []


def test_available_providers_op(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/op" if x == "op" else None)
    assert providers.available_providers() == ["1password"]


def test_fetch_missing_cli_raises(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(providers.ProviderNotFoundError):
        providers.fetch("1password", "op://x")


def test_fetch_unknown_provider_raises():
    with pytest.raises(providers.ProviderNotFoundError):
        providers.fetch("unknown", "ref")


def test_fetch_1password_calls_op(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        m = MagicMock()
        m.returncode = 0
        m.stdout = b"secret-bytes\n"
        m.stderr = b""
        return m

    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/op" if x == "op" else None)
    monkeypatch.setattr("subprocess.run", fake_run)
    result = providers.fetch("1password", "op://vault/item/field")
    assert result == b"secret-bytes"
    assert calls[0][0] == "op"
    assert calls[0][1] == "read"


def test_fetch_locked_raises(monkeypatch):
    def fake_run(cmd, **kw):
        m = MagicMock()
        m.returncode = 1
        m.stdout = b""
        m.stderr = b"error: not signed in"
        return m

    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/op" if x == "op" else None)
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(providers.ProviderLockedError):
        providers.fetch("1password", "op://x")


def test_fetch_cli_error_raises(monkeypatch):
    def fake_run(cmd, **kw):
        m = MagicMock()
        m.returncode = 1
        m.stdout = b""
        m.stderr = b"some other error"
        return m

    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/op" if x == "op" else None)
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(providers.ProviderCLIError):
        providers.fetch("1password", "op://x")


def test_fetch_timeout_raises(monkeypatch):
    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/op" if x == "op" else None)
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(providers.ProviderCLIError):
        providers.fetch("1password", "op://x")


def test_vault_init_provider_stores_ref(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    monkeypatch.setattr("sqac.providers.fetch", lambda p, r: b"master-key-bytes-32-chars-long!!")
    from sqac.vault import Vault, read_cartridge
    v = Vault.create(tmp_path / "v.sqac", mode="provider", provider="1password",
                     provider_ref="op://vault/item/field")
    cart = read_cartridge(str(tmp_path / "v.sqac"))
    assert cart.header.extra["vault"]["kdf"] == "provider"
    assert cart.header.extra["vault"]["provider"] == "1password"
    assert cart.header.extra["vault"]["provider_ref"] == "op://vault/item/field"
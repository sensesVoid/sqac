"""Multi-vault support tests."""
from __future__ import annotations

import pytest
from pathlib import Path

from sqac.vault_registry import VaultRegistry, VaultNotFoundError
from sqac.vault import parse_vault_ref, canonical_ref


def test_list_vaults_empty(tmp_path):
    reg = VaultRegistry(tmp_path / "vaults")
    assert reg.list_vaults() == []


def test_init_and_list(tmp_path):
    reg = VaultRegistry(tmp_path / "vaults")
    reg.init("work", mode="portable", passphrase="pw")
    assert "work" in reg.list_vaults()


def test_active_defaults_to_default(tmp_path):
    reg = VaultRegistry(tmp_path / "vaults")
    assert reg.active() == "default"


def test_set_active_roundtrip(tmp_path):
    reg = VaultRegistry(tmp_path / "vaults")
    reg.init("work", mode="portable", passphrase="pw")
    reg.set_active("work")
    assert reg.active() == "work"


def test_path_for_unknown_raises(tmp_path):
    reg = VaultRegistry(tmp_path / "vaults")
    with pytest.raises(VaultNotFoundError):
        reg.path_for("nonexistent")


def test_parse_vault_ref_simple():
    assert parse_vault_ref("vault:mysecret") == (None, "mysecret")


def test_parse_vault_ref_with_vault():
    assert parse_vault_ref("vault:work/mysecret") == ("work", "mysecret")


def test_parse_vault_ref_braced():
    assert parse_vault_ref("${vault:work/mysecret}") == ("work", "mysecret")


def test_parse_vault_ref_not_a_ref():
    assert parse_vault_ref("not-a-ref") is None


def test_canonical_ref_simple():
    assert canonical_ref("mysecret") == "vault:mysecret"


def test_canonical_ref_with_vault():
    assert canonical_ref("mysecret", vault="work") == "vault:work/mysecret"


def test_cross_vault_isolation(tmp_path, monkeypatch):
    reg = VaultRegistry(tmp_path / "vaults")
    # Portable mode + env passphrase: works headless (no OS keychain on CI)
    reg.init("work", mode="portable", passphrase="pw")
    reg.init("personal", mode="portable", passphrase="pw")
    monkeypatch.setenv("SQAC_VAULT_PASSPHRASE", "pw")
    # Store a secret in personal vault
    v = reg.open("personal")
    v.put("api_key", "secret-value")
    # Resolve via explicit vault ref
    vault_name, name = parse_vault_ref("vault:personal/api_key")
    assert vault_name == "personal"
    assert name == "api_key"
    val = reg.resolve(name, vault=vault_name)
    assert val == "secret-value"
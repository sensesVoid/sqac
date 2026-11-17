"""Agent cartridge encryption tests."""
from __future__ import annotations

import pytest
from pathlib import Path

from sqac.agent import AgentCartridge


def test_create_plaintext_by_default(tmp_path):
    ac = AgentCartridge.create(tmp_path / "a.sqac", "agent1")
    raw = (tmp_path / "a.sqac").read_bytes()
    assert not raw.startswith(b"SQAE")


def test_create_encrypted(tmp_path):
    ac = AgentCartridge.create(tmp_path / "a.sqac", "agent1", encrypt=True, passphrase="pw")
    raw = (tmp_path / "a.sqac").read_bytes()
    assert raw.startswith(b"SQAE")


def test_load_encrypted_roundtrip(tmp_path):
    AgentCartridge.create(tmp_path / "a.sqac", "agent1", encrypt=True, passphrase="pw")
    ac = AgentCartridge.load(tmp_path / "a.sqac", passphrase="pw")
    ac.set("note", "hello")
    assert "hello" in ac.bootstrap()


def test_load_encrypted_wrong_passphrase_raises(tmp_path):
    from sqac.vault_crypto import VaultCryptoError
    AgentCartridge.create(tmp_path / "a.sqac", "agent1", encrypt=True, passphrase="pw")
    with pytest.raises(VaultCryptoError):
        AgentCartridge.load(tmp_path / "a.sqac", passphrase="wrong")


def test_encrypt_existing_plaintext(tmp_path):
    ac = AgentCartridge.create(tmp_path / "a.sqac", "agent1")
    ac.encrypt("pw")
    raw = (tmp_path / "a.sqac").read_bytes()
    assert raw.startswith(b"SQAE")


def test_decrypt_to_plaintext(tmp_path):
    ac = AgentCartridge.create(tmp_path / "a.sqac", "agent1", encrypt=True, passphrase="pw")
    ac.decrypt("pw")
    raw = (tmp_path / "a.sqac").read_bytes()
    assert not raw.startswith(b"SQAE")


def test_plaintext_agent_still_loads_without_passphrase(tmp_path):
    ac = AgentCartridge.create(tmp_path / "a.sqac", "agent1")
    ac2 = AgentCartridge.load(tmp_path / "a.sqac")
    assert ac2.name == "agent1"
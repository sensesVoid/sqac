"""Secret Domain cartridge tests: refs/metadata only, never raw values."""
from __future__ import annotations

import pytest
from pathlib import Path

from sqac.secret_domain import (
    SecretDomain, create_domain, open_domain, list_domains,
    classify_ref, parse_domain_ref,
)
from sqac.secret_resolver import SecretResolver, SecretResolutionError
from sqac.format import read_cartridge

PASS = "test-pw-123"


def _make_domain(tmp_path: Path, name: str = "default", encrypted: bool = True) -> SecretDomain:
    return SecretDomain.create(tmp_path / f"{name}.sqac", name,
                               encrypted=encrypted, passphrase=(PASS if encrypted else None))


# -- encryption envelope -----------------------------------------------------

def test_encrypted_roundtrip(tmp_path):
    d = _make_domain(tmp_path)
    d.put_ref("openrouter", "vault:prod/openrouter", description="key")
    d2 = SecretDomain.open(tmp_path / "default.sqac", passphrase=PASS)
    assert d2.get_ref_meta("openrouter")["ref"] == "vault:prod/openrouter"


def test_file_is_encrypted(tmp_path):
    d = _make_domain(tmp_path)
    d.put_ref("openrouter", "vault:prod/openrouter")
    raw = (tmp_path / "default.sqac").read_bytes()
    assert raw.startswith(b"SQSE")
    assert b"vault:prod/openrouter" not in raw


def test_wrong_passphrase_fails(tmp_path):
    d = _make_domain(tmp_path)
    with pytest.raises(Exception):
        SecretDomain.open(tmp_path / "default.sqac", passphrase="wrong-pw")


def test_unencrypted_domain_works(tmp_path):
    d = _make_domain(tmp_path, encrypted=False)
    d.put_ref("x", "vault:a")
    d2 = SecretDomain.open(tmp_path / "default.sqac")
    assert d2.get_ref_meta("x")["ref"] == "vault:a"


def test_encrypted_domain_bad_passphrase_no_plaintext_written(tmp_path):
    d = _make_domain(tmp_path)
    before = (tmp_path / "default.sqac").read_bytes()
    with pytest.raises(Exception):
        SecretDomain.open(tmp_path / "default.sqac", passphrase="nope")
    assert (tmp_path / "default.sqac").read_bytes() == before


# -- header / invariants ------------------------------------------------------

def test_header_never_syncs(tmp_path):
    _make_domain(tmp_path)  # encrypted by default
    plain = SecretDomain.create(tmp_path / "plain.sqac", "plain",
                                encrypted=False, passphrase=None)
    cart = read_cartridge(str(tmp_path / "plain.sqac"))
    assert cart.header.extra["cartridge_type"] == "secret"
    assert cart.header.extra["sync_policy"] == "never"
    assert cart.header.extra["secret_domain"]["schema"] == 1


# -- CRUD ----------------------------------------------------------------------

def test_put_and_list(tmp_path):
    d = _make_domain(tmp_path)
    d.put_ref("a", "vault:x", description="first", tags=["t1"], scopes=["inference"])
    d.put_ref("b", "op://Private/Item/field", description="second")
    rows = d.list()
    assert {r["name"] for r in rows} == {"a", "b"}
    assert rows[0]["ref_kind"] in ("vault", "provider")


def test_put_upsert_preserves_created(tmp_path):
    d = _make_domain(tmp_path)
    d.put_ref("a", "vault:x", description="v1")
    created = d.get_ref_meta("a")["created"]
    d.put_ref("a", "vault:y", description="v2")
    meta = d.get_ref_meta("a")
    assert meta["ref"] == "vault:y"
    assert meta["created"] == created


def test_rename_and_remove(tmp_path):
    d = _make_domain(tmp_path)
    d.put_ref("a", "vault:x")
    assert d.rename("a", "b") is True
    with pytest.raises(KeyError):
        d.get_ref_meta("a")
    assert d.get_ref_meta("b")["ref"] == "vault:x"
    assert d.remove("b") is True
    assert d.remove("b") is False


def test_rename_conflict_rejected(tmp_path):
    d = _make_domain(tmp_path)
    d.put_ref("a", "vault:x")
    d.put_ref("b", "vault:y")
    with pytest.raises(ValueError):
        d.rename("a", "b")


def test_get_meta_missing_raises(tmp_path):
    d = _make_domain(tmp_path)
    with pytest.raises(KeyError):
        d.get_ref_meta("nope")


# -- validation -----------------------------------------------------------------

@pytest.mark.parametrize("ref", [
    "secret:default/x",          # no indirection chains
    "notathing",
    "",
    "vault:",
    "provider:azure:x",          # unsupported provider
])
def test_put_rejects_bad_refs(tmp_path, ref):
    d = _make_domain(tmp_path)
    with pytest.raises(ValueError):
        d.put_ref("x", ref)


def test_put_rejects_raw_secret_in_description(tmp_path):
    d = _make_domain(tmp_path)
    with pytest.raises(ValueError):
        d.put_ref("x", "vault:a",
                  description="sk-proj-abcdefghijklmnopqrstuvwxyz1234567890abcdef")


def test_put_rejects_high_entropy_token(tmp_path):
    d = _make_domain(tmp_path)
    with pytest.raises(ValueError):
        d.put_ref("x", "vault:a",
                  description="token Abc123XyzPqr456StuVwx789YzAbCde012FghIjKlMnoPqr")


def test_put_rejects_bad_name(tmp_path):
    d = _make_domain(tmp_path)
    for name in ["", "a b", "a/b", "x" * 129]:
        with pytest.raises(ValueError):
            d.put_ref(name, "vault:a")


def test_classify_ref():
    assert classify_ref("vault:a") == "vault"
    assert classify_ref("vault:prod/a") == "vault"
    assert classify_ref("${vault:prod/a}") == "vault"
    assert classify_ref("op://Private/Item/field") == "provider"
    assert classify_ref("bw://item123") == "provider"
    assert classify_ref("provider:1password:op://Private/Item/field") == "provider"
    assert classify_ref("provider:bitwarden:abc123") == "provider"
    with pytest.raises(ValueError):
        classify_ref("secret:default/x")


def test_parse_domain_ref():
    assert parse_domain_ref("secret:default/openrouter") == ("default", "openrouter")
    assert parse_domain_ref("vault:x") is None


# -- module-level helpers (isolated home) --------------------------------------

@pytest.fixture
def sqac_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    import sqac.secret_domain as sd
    monkeypatch.setattr(sd, "SECRET_DIR", home / ".sqacm" / "secrets")
    return home


def test_create_open_list_roundtrip(sqac_home, tmp_path, monkeypatch):
    import sqac.secret_domain as sd
    d = sd.create_domain("default", encrypted=False)
    d.put_ref("a", "vault:x")
    assert sd.list_domains() == ["default"]
    d2 = sd.open_domain("default")
    assert d2.get_ref_meta("a")["ref"] == "vault:x"


# -- resolver chain (isolated home) ---------------------------------------------

def test_resolver_direct_vault_ref(sqac_home, monkeypatch):
    import sqac.secret_domain as sd
    from sqac.vault_registry import VaultRegistry
    monkeypatch.setenv("SQAC_VAULT_PASSPHRASE", PASS)
    reg = VaultRegistry(sqac_home / ".sqacm" / "vaults")
    reg.init("work", mode="portable", passphrase=PASS)
    v = reg.open("work")
    v.put("openrouter", "sk-test-123")
    sd.create_domain("default", encrypted=False)
    d = sd.open_domain("default")
    d.put_ref("openrouter", "vault:work/openrouter")
    r = SecretResolver(sqac_dir=sqac_home / ".sqacm")
    assert r.resolve("secret:default/openrouter") == "sk-test-123"
    assert r.resolve("vault:work/openrouter") == "sk-test-123"


def test_resolver_enforces_agent_allowlist(sqac_home, monkeypatch):
    import sqac.secret_domain as sd
    from sqac.vault_registry import VaultRegistry
    monkeypatch.setenv("SQAC_VAULT_PASSPHRASE", PASS)
    reg = VaultRegistry(sqac_home / ".sqacm" / "vaults")
    reg.init("work", mode="portable", passphrase=PASS)
    v = reg.open("work")
    v.put("openrouter", "sk-test-123")
    sd.create_domain("default", encrypted=False)
    d = sd.open_domain("default")
    d.put_ref("openrouter", "vault:work/openrouter")
    r = SecretResolver(sqac_dir=sqac_home / ".sqacm")
    with pytest.raises(SecretResolutionError):
        r.resolve("secret:default/openrouter",
                  agent_context={"secret_refs": ["secret:default/other"]})
    assert r.resolve("secret:default/openrouter",
                     agent_context={"secret_refs": ["secret:default/openrouter"]}) == "sk-test-123"


def test_resolver_missing_domain_fails_closed(sqac_home):
    r = SecretResolver(sqac_dir=sqac_home / ".sqacm")
    with pytest.raises(SecretResolutionError):
        r.resolve("secret:nonexistent/x")


def test_resolver_never_logs_values(sqac_home):
    events = []
    r = SecretResolver(sqac_dir=sqac_home / ".sqacm",
                       audit_callback=events.append,
                       allow_providers=False)
    try:
        r.resolve("vault:work/definitely-not-there")
    except SecretResolutionError:
        pass
    assert events, "resolution failure must be audited"
    for e in events:
        blob = str(e)
        assert "sk-" not in blob
        assert "definitely-not-there" in blob
        assert e.get("event") in ("secret_resolve_error", "secret_resolve_denied")


# -- agent schema v2 -------------------------------------------------------------

def test_agent_schema_v2_roundtrip(tmp_path):
    from sqac.agent import AgentCartridge
    p = tmp_path / "jarvis.sqac"
    ac = AgentCartridge.create(p, "jarvis", role="assistant",
                               model_profile_id="openrouter-gpt-5",
                               secret_ref="vault:prod/openrouter",
                               allowed_tools=["mem_search"])
    ac2 = AgentCartridge.load(p)
    assert ac2.model_profile_id == "openrouter-gpt-5"
    assert ac2.secret_ref == "vault:prod/openrouter"
    assert ac2.sync_policy == "never"
    assert ac2.allowed_tools == ["mem_search"]
    cart = read_cartridge(str(p))
    assert cart.header.extra["agent"]["schema"] == 2
    assert cart.header.extra["sync_policy"] == "never"


def test_agent_schema_v1_upgrades_on_save(tmp_path):
    from sqac.agent import AgentCartridge
    from sqac.format import WRITE_VERSION, CartridgeHeader, write_cartridge
    p = tmp_path / "old.sqac"
    hdr = CartridgeHeader(version=WRITE_VERSION, dims=1024, extra={
        "cartridge_type": "agent",
        "agent": {"schema": 1, "name": "old", "role": "legacy",
                  "created": "2020-01-01T00:00:00Z", "updated": "2020-01-01T00:00:00Z"},
        "encrypted": False,
    })
    write_cartridge(p, hdr, {}, [])
    ag = AgentCartridge.load(p)
    ag.set("identity", "legacy agent")
    cart = read_cartridge(str(p))
    assert cart.header.extra["agent"]["schema"] == 2
    assert cart.header.extra["agent"]["model_profile_id"] is None
    assert cart.header.extra["sync_policy"] == "never"


def test_agent_set_binding(tmp_path):
    from sqac.agent import AgentCartridge
    p = tmp_path / "jarvis.sqac"
    ac = AgentCartridge.create(p, "jarvis")
    ac.set_binding(model_profile_id="m1", secret_ref="vault:a",
                   allowed_tools=["t1", "t2"], sync_policy="never")
    assert ac.model_profile_id == "m1"
    assert ac.secret_ref == "vault:a"
    assert ac.allowed_tools == ["t1", "t2"]
    assert ac.secret_ref in ac.secret_refs
    with pytest.raises(ValueError):
        ac.set_binding(sync_policy="bogus")


def test_agent_binding_consistency(tmp_path):
    from sqac.agent import AgentCartridge
    p = tmp_path / "jarvis.sqac"
    ac = AgentCartridge.create(p, "jarvis")
    ac.set_binding(secret_ref="vault:a")
    assert ac.secret_refs == ["vault:a"]
    ac.set_binding(secret_refs=["vault:x", "vault:y"])
    assert ac.secret_refs == ["vault:x", "vault:y"]
    assert ac.secret_ref is None  # primary dropped: new allowlist excludes it
    # both given, inconsistent -> rejected (spec: must be consistent)
    with pytest.raises(ValueError):
        ac.set_binding(secret_ref="vault:a", secret_refs=["vault:b"])
    # both given, consistent -> accepted
    ac.set_binding(secret_ref="vault:x", secret_refs=["vault:x", "vault:y"])
    assert ac.secret_ref == "vault:x"
    assert ac.secret_refs == ["vault:x", "vault:y"]

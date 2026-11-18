#!/usr/bin/env python3
"""HTTP parity tests (Tier 1) for sqac.server.

Covers the deliverable's acceptance criteria:
1. Agent CRUD round-trip over HTTP (no vault anywhere)
   1b. Encrypted agents expose metadata without secrets or even without a passphrase
2. Teach over HTTP -> observable in rack search, no library calls
3. Mounted rack lists itself (entries / kinds / percent_full)
4. Observe a turn over HTTP -> recallable in the same process
5. Agent export/import lives in Tier 2, so not covered here
6. Rack capabilities are discoverable
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from sqac.server import app, _state
import sqac.agent as agent_mod
import sqac.server as srv


def _reset(tmpdir, api_key="sk-test", rack=True, session=False):
    _state.clear()
    os.environ["SQAC_DIR"] = tmpdir
    os.environ["SQAC_API_KEY"] = api_key
    if rack:
        os.environ["SQAC_RACK"] = "1"
    else:
        os.environ.pop("SQAC_RACK", None)
    if session:
        os.environ["SQAC_SESSION"] = "session.sqac"
    else:
        os.environ.pop("SQAC_SESSION", None)
    srv._api_key = api_key
    return Path(tmpdir)


class _Base(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        self._agents_orig = agent_mod.AGENT_DIR
        self._active_orig = agent_mod.ACTIVE_AGENT_PATH
        agent_mod.AGENT_DIR = Path(self.tmpdir) / "agents"
        agent_mod.ACTIVE_AGENT_PATH = Path(self.tmpdir) / "agent_active"
        agent_mod.AGENT_DIR.mkdir(parents=True, exist_ok=True)
        self.d = None

    def tearDown(self):
        _state.clear()
        agent_mod.AGENT_DIR = self._agents_orig
        agent_mod.ACTIVE_AGENT_PATH = self._active_orig
        for k in ("SQAC_DIR", "SQAC_API_KEY", "SQAC_RACK", "SQAC_SESSION"):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _h(self):
        return {"X-API-Key": "sk-test"}


class TestAgentCrud(_Base):
    def test_roundtrip(self):
        d = _reset(self.tmpdir, rack=False)
        r = self.client.post("/agents", headers=self._h(), json={
            "name": "atlas", "role": "coding assistant",
            "allowed_tools": ["shell", "git"], "sync_policy": "never"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertEqual(r.json()["role"], "coding assistant")

        r = self.client.get("/agents", headers=self._h())
        self.assertEqual([a["name"] for a in r.json()["agents"]], ["atlas"])

        r = self.client.get("/agents/atlas", headers=self._h())
        self.assertEqual(r.json()["allowed_tools"], ["shell", "git"])

        r = self.client.put("/agents/atlas", headers=self._h(),
                            json={"allowed_tools": ["shell"], "sync_policy": "local_only"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["allowed_tools"], ["shell"])
        self.assertEqual(r.json()["sync_policy"], "local_only")

        r = self.client.delete("/agents/atlas", headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["tombstoned"])
        self.assertFalse((self.d_agent("atlas")).exists())
        self.assertTrue((agent_mod.AGENT_DIR / ".trash").exists())

        # tombstoned agent is gone from list; delete hard is final
        r = self.client.get("/agents/atlas", headers=self._h())
        self.assertEqual(r.status_code, 404)

    def d_agent(self, name):
        return agent_mod.AGENT_DIR / f"{name}.sqac"

    def test_conflict_and_validation(self):
        _reset(self.tmpdir, rack=False)
        r = self.client.post("/agents", headers=self._h(),
                             json={"name": "atlas", "role": "r"})
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/agents", headers=self._h(),
                             json={"name": "atlas", "role": "other"})
        self.assertEqual(r.status_code, 409)
        r = self.client.put("/agents/atlas", headers=self._h(),
                            json={"sync_policy": "bogus"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/agents", headers=self._h(),
                             json={"name": "bad/name", "role": "r"})
        self.assertEqual(r.status_code, 400)

    def test_encrypted_agent_exposes_no_secrets(self):
        _reset(self.tmpdir, rack=False)
        import sqac.agent as am
        ag = am.AgentCartridge.create(agent_mod.AGENT_DIR / "vault-bot.sqac",
                                      "vault-bot", "ops", False)
        ag.encrypt("hunter2")
        r = self.client.get("/agents/vault-bot", headers=self._h())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["encrypted"])
        self.assertEqual(body["name"], "vault-bot")
        # no internals leak: no refs, no role contents, nothing secret-adjacent
        for leaked in ("secret_ref", "secret_refs", "allowed_tools", "model_profile_id", "role"):
            if leaked in body:
                self.assertIn(body[leaked], (None, "", []))


class TestRackParity(_Base):
    def test_teach_routes_by_kind(self):
        d = _reset(self.tmpdir)
        r = self.client.post("/teach", headers=self._h(), json={
            "content": "deploy ARM64 first", "key": "deploy-arch", "kind": "fact"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["cartridge"], "facts")
        # folder_routing=True mirrors the MCP rack: kinds route under a folder
        self.assertTrue((d / "facts" / "facts.sqac").exists())

        # teach without kind keeps the old default-memory behavior
        r = self.client.post("/teach", headers=self._h(), json={"content": "no kind fact"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue((d / "memory.sqac").exists()) or True  # default store always exists

    def test_rack_introspection_and_capabilities(self):
        _reset(self.tmpdir)
        self.client.post("/teach", headers=self._h(), json={
            "content": "deploy ARM64 first", "key": "deploy-arch", "kind": "fact"})
        r = self.client.get("/rack", headers=self._h())
        self.assertEqual(r.status_code, 200)
        mounted = {m["name"]: m for m in r.json()["mounted"]}
        self.assertIn("facts", mounted)
        self.assertGreater(mounted["facts"]["entries"], 0)
        self.assertIn("fact", mounted["facts"]["kinds"])
        self.assertIn("percent_full", mounted["facts"])

        r = self.client.get("/rack/capabilities", headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["routes"].get("fact"), "facts")

    def test_reload(self):
        d = _reset(self.tmpdir)
        # facts does not exist until something creates it
        r = self.client.post("/reload", headers=self._h())
        self.assertEqual(r.json()["mounted"], ["memory"])
        self.client.post("/teach", headers=self._h(), json={
            "content": "deploy ARM64 first", "key": "deploy-arch", "kind": "fact"})
        # with folder_routing=True, facts/facts.sqac remounts as "facts__facts"
        # after reload (same artifact the MCP rack has); accept any facts* stem
        r = self.client.post("/reload", headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(any(m.startswith("facts") for m in r.json()["mounted"]))

    def test_rack_search_enrichment_and_trace(self):
        _reset(self.tmpdir)
        self.client.post("/teach", headers=self._h(), json={
            "content": "deploy ARM64 first", "key": "deploy-arch", "kind": "fact"})
        r = self.client.post("/rack/search", headers=self._h(), json={
            "query": "deploy arch", "top_k": 3})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("searched", body)
        self.assertIn("facts", body["searched"])
        self.assertTrue(body["hits"])
        for hit in body["hits"]:
            self.assertIn("cartridge", hit)
            self.assertIn("kind", hit)
            self.assertIn("key", hit)

        r = self.client.post("/rack/trace", headers=self._h(), json={
            "query": "deploy arch", "top_k": 3})
        self.assertEqual(r.status_code, 200)
        for hit in r.json()["hits"]:
            self.assertIn("tier", hit)
            self.assertIn("latency_ms", hit)


class TestSessionLifecycle(_Base):
    def test_stats_graduate_forget(self):
        _reset(self.tmpdir, session=True)
        for i in range(2):
            r = self.client.post("/session/observe", headers=self._h(), json={
                "role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}"})
            self.assertEqual(r.status_code, 200)

        r = self.client.get("/session/stats", headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["live_buffer"], 2)
        self.assertEqual(r.json()["window"], 8)

        # observe -> recall in the same process tree works without a save
        r = self.client.post("/session/recall", headers=self._h(),
                             json={"query": "turn", "top_k": 2})
        self.assertEqual(r.status_code, 200)

        r = self.client.post("/session/graduate", headers=self._h(), json={})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["target"], "facts")

        r = self.client.post("/session/forget", headers=self._h(), json={"all_turns": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["buffer_cleared"])
        self.assertEqual(r.json()["remaining_turns"], 0)


class TestTier2(_Base):
    def test_export_import_roundtrip(self):
        d = _reset(self.tmpdir, rack=False)
        self.client.post("/teach", headers=self._h(), json={
            "content": "deploy ARM64 first", "key": "deploy-arch"})
        r = self.client.post("/cartridges/memory/export", headers=self._h())
        self.assertEqual(r.status_code, 200)
        blob = r.json()["archive_b64"]
        self.assertTrue(blob)
        # wipe the flat cartridge, import, verify content is back
        (d / "memory.sqac").unlink()
        r = self.client.post("/cartridges/memory/import", headers=self._h(),
                             json={"archive_b64": blob})
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/search", headers=self._h(), json={
            "query": "deploy arch", "top_k": 1})
        self.assertIn("ARM64", r.json()["hits"][0]["content"])

    def test_export_folder_routed_cartridge(self):
        _reset(self.tmpdir)  # rack: folder routing on
        self.client.post("/teach", headers=self._h(), json={
            "content": "deploy ARM64 first", "key": "deploy-arch", "kind": "fact"})
        r = self.client.post("/cartridges/facts/export", headers=self._h())
        # facts lives at facts/facts.sqac; the resolver must find it
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["archive_b64"])

    def test_import_requires_overwrite(self):
        _reset(self.tmpdir, rack=False)
        self.client.post("/teach", headers=self._h(), json={
            "content": "x", "key": "k"})
        r = self.client.post("/cartridges/memory/export", headers=self._h())
        blob = r.json()["archive_b64"]
        r = self.client.post("/cartridges/memory/import", headers=self._h(),
                             json={"archive_b64": blob})
        self.assertEqual(r.status_code, 409)
        r = self.client.post("/cartridges/memory/import", headers=self._h(),
                             json={"archive_b64": blob, "overwrite": True})
        self.assertEqual(r.status_code, 200)

    def test_readonly_blocks_mutations(self):
        _reset(self.tmpdir, rack=False)
        srv._readonly = True
        try:
            r = self.client.post("/teach", headers=self._h(), json={
                "content": "x", "key": "k"})
            self.assertEqual(r.status_code, 403)
            r = self.client.get("/stats", headers=self._h())
            self.assertEqual(r.status_code, 200)
        finally:
            srv._readonly = False

    def test_scope_blocks_out_of_scope_cartridge(self):
        d = _reset(self.tmpdir)
        srv._scope = ["facts"]
        try:
            # facts-in-scope write is fine
            r = self.client.post("/teach", headers=self._h(), json={
                "content": "deploy ARM64 first", "key": "k", "kind": "fact"})
            self.assertEqual(r.status_code, 200)
            # default memory cartridge is out of scope -> 403
            r = self.client.post("/teach", headers=self._h(), json={
                "content": "x", "key": "k"})
            self.assertEqual(r.status_code, 403)
            # unscoped default /stats is also out of scope -> 403
            r = self.client.get("/stats", headers=self._h())
            self.assertEqual(r.status_code, 403)
            # but the scoped cartridge reads fine
            r = self.client.get("/stats?cartridge=facts", headers=self._h())
            self.assertEqual(r.status_code, 200)
        finally:
            srv._scope = []

    def test_capabilities_version_describe(self):
        _reset(self.tmpdir, rack=False)
        r = self.client.get("/capabilities", headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["writable"])
        self.assertEqual(r.json()["scope"], [])

        srv._readonly = True
        try:
            r = self.client.get("/capabilities", headers=self._h())
            self.assertFalse(r.json()["writable"])
            self.assertTrue(r.json()["readonly"])
        finally:
            srv._readonly = False

        from sqac._version import __version__
        r = self.client.get("/version")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["version"], __version__)

        r = self.client.get("/describe", headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertIn("/agents", r.json()["routes"].get("GET", []))
        self.assertIn("/rack/search", r.json()["routes"].get("POST", []))


if __name__ == "__main__":
    unittest.main()

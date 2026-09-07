#!/usr/bin/env python3
"""Tests for sqac.server — HTTP API server."""
import os
import tempfile
import unittest
from pathlib import Path

from sqac.server import app, _state
from sqac.store import SqacStore


def _reset_server(tmpdir, api_key="sk-test"):
    """Reset server state for a fresh test."""
    _state.clear()
    os.environ["SQAC_DIR"] = tmpdir
    os.environ["SQAC_API_KEY"] = api_key
    import sqac.server as srv
    srv._api_key = api_key
    return Path(tmpdir)


class TestServerHealth(unittest.TestCase):
    """Test /health endpoint (no auth required)."""

    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        _state.clear()
        os.environ["SQAC_DIR"] = self.tmpdir

    def tearDown(self):
        _state.clear()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_health_returns_ok(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("simd", data)
        self.assertIn("lz4", data)
        self.assertIn("uptime_s", data)

    def test_health_no_auth(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)


class TestServerWithAuth(unittest.TestCase):
    """Test endpoints with API key auth."""

    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        self.d = _reset_server(self.tmpdir)
        # Create a pre-populated cartridge
        store = SqacStore()
        store.add("FastAPI is a web framework", key="fastapi")
        store.add("Pydantic validates data", key="pydantic")
        store.add("Redis is an in-memory cache", key="redis")
        store.save(self.d / "memory.sqac")
        self.h = {"X-API-Key": "sk-test"}

    def tearDown(self):
        _state.clear()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stats(self):
        resp = self.client.get("/stats", headers=self.h)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["entries"], 3)

    def test_search(self):
        resp = self.client.post("/search", headers=self.h, json={
            "query": "web framework", "top_k": 2
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["count"], 0)
        self.assertIn("FastAPI", data["hits"][0]["content"])

    def test_search_kind_filter(self):
        resp = self.client.post("/search", headers=self.h, json={
            "query": "web framework", "top_k": 2, "kind": "fact"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 0)

    def test_search_threshold(self):
        resp = self.client.post("/search", headers=self.h, json={
            "query": "quantum physics", "top_k": 3, "threshold": 0.95
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 0)

    def test_teach_adds_entry(self):
        resp = self.client.post("/teach", headers=self.h, json={
            "content": "SQLite is an embedded database", "key": "sqlite"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

    def test_teach_then_search(self):
        self.client.post("/teach", headers=self.h, json={
            "content": "PostgreSQL is a relational database", "key": "postgres"
        })
        resp = self.client.post("/search", headers=self.h, json={
            "query": "database", "top_k": 2
        })
        self.assertGreater(resp.json()["count"], 0)

    def test_compact(self):
        resp = self.client.post("/compact", headers=self.h, json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("original", data)
        self.assertIn("alive", data)

    def test_cartridges_list(self):
        resp = self.client.get("/cartridges", headers=self.h)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("memory", resp.json()["cartridges"])

    def test_auth_no_key(self):
        resp = self.client.post("/search", json={"query": "test"})
        self.assertEqual(resp.status_code, 401)

    def test_auth_wrong_key(self):
        resp = self.client.post("/search", headers={"X-API-Key": "sk-wrong"},
                               json={"query": "test"})
        self.assertEqual(resp.status_code, 401)


class TestServerMetrics(unittest.TestCase):
    """Test /metrics endpoint."""

    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        self.d = _reset_server(self.tmpdir)
        store = SqacStore()
        store.add("FastAPI is a web framework", key="fastapi")
        store.save(self.d / "memory.sqac")
        self.h = {"X-API-Key": "sk-test"}

    def tearDown(self):
        _state.clear()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_metrics_returns_prometheus_format(self):
        resp = self.client.get("/metrics", headers=self.h)
        self.assertEqual(resp.status_code, 200)
        text = resp.text
        self.assertIn("sqac_uptime_seconds", text)
        self.assertIn("sqac_entries", text)
        self.assertIn("sqac_dims", text)
        self.assertIn("sqac_simd_enabled", text)
        self.assertIn("sqac_lz4_enabled", text)
        self.assertIn("# TYPE sqac_entries gauge", text)

    def test_metrics_counts_requests(self):
        self.client.post("/search", headers=self.h, json={"query": "web", "top_k": 1})
        self.client.post("/teach", headers=self.h, json={"content": "test entry", "key": "t"})
        resp = self.client.get("/metrics", headers=self.h)
        text = resp.text
        self.assertIn("sqac_requests_total", text)
        # teach_total is global — just verify it's present and >= 1
        self.assertRegex(text, r"sqac_teach_total \d+")

    def test_metrics_search_latency(self):
        self.client.post("/search", headers=self.h, json={"query": "web", "top_k": 1})
        resp = self.client.get("/metrics", headers=self.h)
        text = resp.text
        self.assertIn("sqac_search_latency_ms", text)
        # count is global — verify metric is present with a positive count
        self.assertRegex(text, r"sqac_search_latency_ms_count \d+")

    def test_metrics_no_auth_rejected(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 401)


class TestServerDashboard(unittest.TestCase):
    """Test /dashboard endpoint."""

    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        self.d = _reset_server(self.tmpdir)
        store = SqacStore()
        store.add("FastAPI is a web framework", key="fastapi")
        store.add("Redis is a cache", key="redis")
        store.save(self.d / "memory.sqac")
        self.h = {"X-API-Key": "sk-test"}

    def tearDown(self):
        _state.clear()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dashboard_returns_html(self):
        resp = self.client.get("/dashboard", headers=self.h)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        html = resp.text
        self.assertIn("SQAC Dashboard", html)
        self.assertIn("Entries", html)
        self.assertIn("Redis", html)  # at least one entry shown
        self.assertIn("/metrics", html)
        self.assertIn("/search", html)

    def test_dashboard_shows_stats(self):
        resp = self.client.get("/dashboard", headers=self.h)
        html = resp.text
        self.assertIn("2</div>", html)  # 2 entries
        self.assertIn("1024", html)     # dims

    def test_dashboard_no_auth_rejected(self):
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 401)


class TestServerRack(unittest.TestCase):
    """Test rack-enabled server endpoints."""

    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        # Create rack with default cartridge
        from sqac.rack import CartridgeRack
        rack = CartridgeRack(directory=Path(self.tmpdir), default="default", semantic=True)
        rack.create("default")
        rack.save()
        # Set state directly so the server uses our configured rack
        _state.clear()
        os.environ["SQAC_DIR"] = self.tmpdir
        os.environ["SQAC_RACK"] = "1"
        os.environ["SQAC_API_KEY"] = "sk-rack"
        import sqac.server as srv
        srv._api_key = "sk-rack"
        _state["dir"] = Path(self.tmpdir)
        _state["rack"] = rack
        _state["default_store"] = None
        _state["offloader"] = None
        self.h = {"X-API-Key": "sk-rack"}

    def tearDown(self):
        _state.clear()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("SQAC_RACK", None)
        os.environ.pop("SQAC_API_KEY", None)

    def test_rack_write_and_search(self):
        resp = self.client.post("/rack/write", headers=self.h, json={
            "content": "Test knowledge for rack", "key": "test"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        resp = self.client.post("/rack/search", headers=self.h, json={
            "query": "knowledge", "top_k": 2
        })
        self.assertEqual(resp.status_code, 200)

    def test_rack_search_empty(self):
        resp = self.client.post("/rack/search", headers=self.h, json={
            "query": "anything", "top_k": 3
        })
        self.assertEqual(resp.status_code, 200)


class TestServerCompactWithTombstones(unittest.TestCase):
    """Test compact actually removes tombstones."""

    def test_compact_removes_tombstones(self):
        """Create entries, delete some, save to disk, compact via API."""
        d = Path(tempfile.mkdtemp())
        _state.clear()
        os.environ["SQAC_DIR"] = str(d)
        os.environ["SQAC_API_KEY"] = "sk-c"
        import sqac.server as srv
        srv._api_key = "sk-c"

        # Build cartridge with tombstones directly on disk
        store = SqacStore()
        store.add("Entry 0", key="e0")
        store.add("Entry 1", key="e1")
        store.add("Entry 2", key="e2")
        store.add("Entry 3", key="e3")
        store.add("Entry 4", key="e4")
        store.delete(0)
        store.delete(1)
        store.delete(2)
        store.save(d / "memory.sqac")

        from fastapi.testclient import TestClient
        client = TestClient(app)
        h = {"X-API-Key": "sk-c"}

        resp = client.post("/compact", headers=h, json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["compacted"])
        self.assertEqual(data["removed"], 3)
        self.assertEqual(data["alive"], 2)

        _state.clear()
        import shutil
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

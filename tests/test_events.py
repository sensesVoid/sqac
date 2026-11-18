#!/usr/bin/env python3
"""Tests for the sqac event bus (deliverable proposal 3).

Covers bus semantics, the webhook (mocked — no real network), and the
GET /events SSE endpoint. The full mem_* wiring is exercised in the MCP
integration suite instead of here.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sqac.server import app, _state
import sqac.server as srv
import sqac.events as events


class TestEventBus(unittest.TestCase):
    def test_emit_backlog_and_seq(self):
        e1 = events.emit("test_probe", marker="alpha-1")
        e2 = events.emit("test_probe", marker="beta-2")
        self.assertIsInstance(e1["seq"], int)
        self.assertGreater(e2["seq"], e1["seq"])
        backlog = events.backlog()
        markers = [e.get("marker") for e in backlog]
        self.assertIn("alpha-1", markers)
        self.assertIn("beta-2", markers)

    def test_subscribe_receives_live_events(self):
        q = events.subscribe()
        try:
            events.emit("test_probe", marker="live-3")
            evt = q.get(timeout=2)
            self.assertEqual(evt["marker"], "live-3")
        finally:
            events.unsubscribe(q)

    def test_webhook_posts_json(self):
        posted = {}

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=-1):
                return b""

        def fake_urlopen(req, timeout=0):
            posted["url"] = req.full_url
            posted["body"] = req.data
            posted["content_type"] = req.get_header("Content-type")
            return FakeResp()

        with mock.patch.object(events, "urlopen", fake_urlopen):
            os.environ["SQAC_WEBHOOK_URL"] = "http://127.0.0.1:9/hook"
            try:
                events.emit("test_probe", marker="hooked-4")
            finally:
                os.environ.pop("SQAC_WEBHOOK_URL", None)
        import time
        time.sleep(0.3)  # webhook fires on a daemon thread
        self.assertEqual(posted.get("url"), "http://127.0.0.1:9/hook")
        self.assertIn(b"hooked-4", posted.get("body", b""))
        self.assertEqual(posted.get("content_type"), "application/json")

    def test_webhook_failure_is_swallowed(self):
        with mock.patch.object(events, "urlopen", side_effect=OSError("refused")):
            os.environ["SQAC_WEBHOOK_URL"] = "http://127.0.0.1:9/hook"
            try:
                events.emit("test_probe", marker="hook-fail-5")  # must not raise
            finally:
                os.environ.pop("SQAC_WEBHOOK_URL", None)
        import time
        time.sleep(0.2)

    def test_emit_never_raises(self):
        events.emit("test_probe", weird=object())  # non-JSON-able payload


class TestEventsSse(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        _state.clear()
        os.environ["SQAC_DIR"] = self.tmpdir
        os.environ["SQAC_API_KEY"] = "sk-test"
        os.environ.pop("SQAC_RACK", None)
        os.environ.pop("SQAC_SESSION", None)
        srv._api_key = "sk-test"

    def tearDown(self):
        _state.clear()
        for k in ("SQAC_DIR", "SQAC_API_KEY", "SQAC_RACK", "SQAC_SESSION"):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _h(self):
        return {"X-API-Key": "sk-test"}

    def test_events_requires_auth(self):
        r = self.client.get("/events?once=true")
        self.assertEqual(r.status_code, 401)

    def test_events_once_streams_backlog(self):
        marker = "sse-marker-7"
        events.emit("mem_write", key=marker, kind="fact", cartridge="facts", source="http")
        r = self.client.get("/events?once=true", headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.headers.get("content-type", ""))
        self.assertIn("event: mem_write", r.text)
        self.assertIn(marker, r.text)

    def test_events_allowed_in_readonly(self):
        srv._readonly = True
        try:
            r = self.client.get("/events?once=true", headers=self._h())
            self.assertEqual(r.status_code, 200)
        finally:
            srv._readonly = False


if __name__ == "__main__":
    unittest.main()

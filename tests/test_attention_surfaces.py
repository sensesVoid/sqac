import os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqac.attention import render_block
from sqac.integrations import SqacClient
from sqac.tasks import TaskManager
from sqac.rack import CartridgeRack
from sqac.server import app, _state
import sqac.server as srv


class _FakeRespMsg:
    def __init__(self, content): self.content = content


class _FakeRespChoice:
    def __init__(self, content): self.message = _FakeRespMsg(content)


class _FakeResp:
    def __init__(self): self.choices = [_FakeRespChoice("ok")]


class _FakeCompletions:
    def __init__(self): self.calls = []
    def create(self, model, messages, **kwargs):
        self.calls.append(messages)
        return _FakeResp()


class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()


class _FakeUpstream:
    def __init__(self): self.chat = _FakeChat()


class TestMiddlewareAttention(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.mkdtemp()
        os.environ["SQAC_MEM_DIR"] = self._td

    def tearDown(self):
        for k in ("SQAC_MEM_DIR", "SQAC_ATTENTION_TURNS"):
            os.environ.pop(k, None)
        shutil.rmtree(self._td, ignore_errors=True)

    def _make_client(self, turns):
        up = _FakeUpstream()
        c = SqacClient(up, memory_dir=self._td)
        c.attention_turns = turns
        return c

    def _sys_msgs(self, c):
        return [m for m in c._upstream.chat.completions.calls[-1]
                if m.get("role") == "system"]

    def test_injected_at_cadence(self):
        c = self._make_client(turns=2)
        tm = TaskManager(c.rack)
        tm.add("alpha task", status="current")
        c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        self.assertFalse(any("[SQAC attention" in (m.get("content") or "") for m in self._sys_msgs(c)))
        c.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        self.assertTrue(any("[SQAC attention" in (m.get("content") or "") for m in self._sys_msgs(c)))

    def test_render_attention(self):
        c = self._make_client(turns=2)
        tm = TaskManager(c.rack)
        tm.add("beta task", status="current")
        block = c.chat.completions._render_attention()
        self.assertIn("[SQAC attention", block)
        self.assertIn("beta task", block)


def _reset(tmpdir):
    _state.clear()
    os.environ["SQAC_DIR"] = tmpdir
    os.environ["SQAC_API_KEY"] = "sk-test"
    os.environ["SQAC_RACK"] = "1"
    os.environ["SQAC_SESSION"] = "session.sqac"
    srv._api_key = "sk-test"


class TestServerAttention(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        _reset(self.tmpdir)

    def tearDown(self):
        _state.clear()
        for k in ("SQAC_DIR", "SQAC_API_KEY", "SQAC_RACK", "SQAC_SESSION", "SQAC_ATTENTION_TURNS"):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _h(self):
        return {"X-API-Key": "sk-test"}

    def test_attention_endpoint_returns_block(self):
        tm = TaskManager(CartridgeRack(directory=self.tmpdir))
        tm.add("ship v2", status="current")
        r = self.client.get("/attention", headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertIn("[SQAC attention | turn 0]", r.json()["attention"])
        self.assertIn("ship v2", r.json()["attention"])

    def test_observe_returns_attention_when_due(self):
        srv.ATTENTION_TURNS = 1
        r = self.client.post("/session/observe", json={"role": "user", "text": "hello"}, headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertIn("attention", r.json())


if __name__ == "__main__":
    unittest.main()

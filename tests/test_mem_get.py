import json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqac.store import SqacStore
from sqac.rack import CartridgeRack
from sqac.server import app, _state
import sqac.server as srv


def _reset(tmpdir, api_key="sk-test"):
    _state.clear()
    os.environ["SQAC_DIR"] = tmpdir
    os.environ["SQAC_API_KEY"] = api_key
    os.environ["SQAC_RACK"] = "1"
    os.environ.pop("SQAC_SESSION", None)
    srv._api_key = api_key


class TestMemGetStoreRack(unittest.TestCase):
    def test_store_get(self):
        st = SqacStore()
        idx = st.add("hello", key="k1")
        mid = st._entries[idx]["meta"]["mem_id"]
        self.assertEqual(st.get(mid)["content"], "hello")
        self.assertIsNone(st.get("nope"))

    def test_rack_get_by_mem_id(self):
        d = tempfile.mkdtemp()
        try:
            rack = CartridgeRack(directory=d, routes={"fact": "facts"}, default="facts", semantic=False)
            rack.create("facts")
            rack.write_routed("alpha", key="a", kind="fact")
            mid = rack._stores["facts"]._entries[-1]["meta"]["mem_id"]
            res = rack.get_by_mem_id(mid)
            self.assertIsNotNone(res)
            name, e = res
            self.assertEqual(name, "facts")
            self.assertEqual(e["content"], "alpha")
            self.assertIsNone(rack.get_by_mem_id("missing"))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestMemGetHTTP(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
        self.tmpdir = tempfile.mkdtemp()
        _reset(self.tmpdir)

    def tearDown(self):
        _state.clear()
        for k in ("SQAC_DIR","SQAC_API_KEY","SQAC_RACK","SQAC_SESSION"):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _h(self):
        return {"X-API-Key": "sk-test"}

    def test_get_entry_found_and_not_found(self):
        # create a fact via teach
        r = self.client.post("/teach", json={"content": "deploy to prod", "key": "deploy", "kind": "fact"}, headers=self._h())
        self.assertEqual(r.status_code, 200)
        # find mem_id via rack search
        sr = self.client.post("/rack/search", json={"query": "deploy", "top_k": 5}, headers=self._h())
        self.assertEqual(sr.status_code, 200)
        hits = sr.json().get("hits", [])
        self.assertTrue(hits)
        mid = hits[0].get("mem_id") or hits[0].get("meta", {}).get("mem_id")
        self.assertTrue(mid)
        gr = self.client.get(f"/entries/{mid}", headers=self._h())
        self.assertEqual(gr.status_code, 200)
        body = gr.json()
        self.assertTrue(body["found"])
        self.assertEqual(body["mem_id"], mid)
        self.assertIn("deploy", body["content"])
        nf = self.client.get("/entries/NOPE123", headers=self._h())
        self.assertEqual(nf.status_code, 404)
        self.assertFalse(nf.json()["found"])


class TestMemGetCLI(unittest.TestCase):
    def test_cli_get(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "mem.sqac")
            st = SqacStore()
            st.add("cli fact", key="k")
            mid = st._entries[0]["meta"]["mem_id"]
            st.save(path, name="mem")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            r = subprocess.run([sys.executable, "-m", "sqac.cli", "get", mid, "--db", path],
                               capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parents[1]))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            out = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertTrue(out["found"])
            self.assertEqual(out["mem_id"], mid)
        finally:
            shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()

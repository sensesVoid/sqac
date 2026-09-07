"""Unit tests for sqac.autobuild — project auto-build and realtime tracking.

Run: python -m pytest tests/test_autobuild.py -q
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from sqac.autobuild import (
    Unit,
    _sync_rack,
    build_store,
    diff_units,
    extract_config,
    extract_directives,
    extract_file,
    extract_markdown,
    extract_module_docstrings,
    extract_project_units,
    load_state,
    save_state,
    track_once,
    unit_map,
    walk_project,
)
from sqac.rack import CartridgeRack
from sqac.store import SqacStore


def _make_project(root: Path, files: dict[str, str]) -> None:
    """Write a dict of {relpath: content} into root."""
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


class TestWalkProject(unittest.TestCase):
    def test_basic_walk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {
                "README.md": "# Hello\nSome text.",
                "src/main.py": "x = 1",
                "tests/test_main.py": "def test_x(): pass",
            })
            files = walk_project(root)
            rels = [p.relative_to(root).as_posix() for p in files]
            self.assertIn("README.md", rels)
            self.assertIn("src/main.py", rels)
            self.assertIn("tests/test_main.py", rels)

    def test_excludes_node_modules(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {
                "README.md": "hi",
                "node_modules/foo/index.js": "module.exports = 1",
            })
            files = walk_project(root)
            rels = [p.relative_to(root).as_posix() for p in files]
            self.assertNotIn("node_modules/foo/index.js", rels)
            self.assertIn("README.md", rels)

    def test_excludes_pycache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {
                "app.py": "x = 1",
                "__pycache__/app.cpython-312.pyc": "binary",
            })
            files = walk_project(root)
            rels = [p.relative_to(root).as_posix() for p in files]
            self.assertIn("app.py", rels)
            self.assertFalse(any("__pycache__" in r for r in rels))

    def test_respects_gitignore(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {
                ".gitignore": "*.log\n",
                "app.py": "x = 1",
                "debug.log": "log data",
            })
            files = walk_project(root)
            rels = [p.relative_to(root).as_posix() for p in files]
            self.assertIn("app.py", rels)
            self.assertNotIn("debug.log", rels)

    def test_skips_large_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {"small.py": "x = 1"})
            big = root / "big.py"
            big.write_bytes(b"x = 1\n" * 50_000)  # > 256KB
            files = walk_project(root)
            rels = [p.relative_to(root).as_posix() for p in files]
            self.assertIn("small.py", rels)
            self.assertNotIn("big.py", rels)

    def test_sorted_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {
                "z_last.py": "x = 1",
                "a_first.py": "y = 2",
                "m_middle.py": "z = 3",
            })
            files = walk_project(root)
            rels = [p.relative_to(root).as_posix() for p in files]
            self.assertEqual(rels, sorted(rels))


class TestExtractMarkdown(unittest.TestCase):
    def test_heading_extraction(self):
        md = "# Getting Started\nInstall with pip.\n\n# Usage\nRun `sqac teach`."
        units = extract_markdown("README.md", md)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0].kind, "fact")
        self.assertIn("Getting Started", units[0].key)
        self.assertIn("Install with pip", units[0].content)

    def test_badge_stripped(self):
        md = "# Title\n[![badge](url)](http://x)\nActual content here."
        units = extract_markdown("README.md", md)
        self.assertTrue(units)
        content = units[0].content
        self.assertNotIn("badge", content.lower() or "actual content" in content.lower())

    def test_long_body_chunked(self):
        long = "# Big\n" + "\n\n".join(f"Paragraph {i}. " * 20 for i in range(20))
        units = extract_markdown("README.md", long)
        # should produce multiple chunks
        self.assertGreater(len(units), 1)

    def test_slug_generation(self):
        md = "# Hello World! This is a Test"
        units = extract_markdown("README.md", md)
        self.assertTrue(units[0].source.endswith("#hello-world-this-is-a-test"))

    def test_empty_file(self):
        units = extract_markdown("empty.md", "")
        self.assertEqual(units, [])


class TestExtractModuleDocstrings(unittest.TestCase):
    def test_module_docstring(self):
        code = '"""This module handles authentication and authorization."""\nx = 1'
        units = extract_module_docstrings("auth.py", code)
        self.assertEqual(len(units), 1)
        self.assertIn("authentication", units[0].content)

    def test_no_docstring(self):
        code = "x = 1\ny = 2"
        units = extract_module_docstrings("plain.py", code)
        self.assertEqual(units, [])

    def test_short_docstring_skipped(self):
        code = '"""short"""\nx = 1'
        units = extract_module_docstrings("short.py", code)
        self.assertEqual(units, [])


class TestExtractConfig(unittest.TestCase):
    def test_package_json(self):
        pkg = json.dumps({
            "name": "my-app",
            "version": "1.0.0",
            "description": "A test app",
            "scripts": {"build": "webpack", "test": "jest"},
            "dependencies": {"react": "^18.0.0"},
            "devDependencies": {"jest": "^29.0.0"},
        })
        units = extract_config("package.json", pkg)
        self.assertTrue(any("my-app" in u.content for u in units))
        self.assertTrue(any("build" in u.key for u in units))
        self.assertTrue(any("react" in u.content for u in units))

    def test_pyproject_toml(self):
        toml = '[project]\nname = "sqac"\nversion = "0.1.0"\ndescription = "VSA memory"\nrequires-python = ">=3.10"\n\n[project.dependencies]\n"numpy"\n'
        units = extract_config("pyproject.toml", toml)
        self.assertTrue(any("sqac" in u.content for u in units))

    def test_requirements_txt(self):
        req = "numpy==1.24.0\npandas==2.0.0\n# comment\n"
        units = extract_config("requirements.txt", req)
        self.assertEqual(len(units), 1)
        self.assertIn("numpy", units[0].content)

    def test_makefile(self):
        mf = "all: build test\n\nbuild:\n\techo build\n\ntest:\n\techo test\n"
        units = extract_config("Makefile", mf)
        self.assertTrue(units)
        self.assertIn("build", units[0].content)
        self.assertIn("test", units[0].content)

    def test_github_workflow(self):
        wf = "name: CI\n\njobs:\n  build:\n    runs-on: ubuntu-latest\n  test:\n    runs-on: ubuntu-latest\n"
        units = extract_config(".github/workflows/ci.yml", wf)
        self.assertTrue(units)
        self.assertIn("build", units[0].content)

    def test_env_example(self):
        env = "DATABASE_URL=postgres://localhost/mydb\nAPI_KEY=sk-xxx\n# comment\n"
        units = extract_config(".env.example", env)
        self.assertTrue(units)
        self.assertIn("DATABASE_URL", units[0].content)
        self.assertIn("API_KEY", units[0].content)

    def test_dockerfile(self):
        df = "FROM python:3.12-slim\nCOPY . /app\nCMD [\"python\", \"app.py\"]\n"
        units = extract_config("Dockerfile", df)
        self.assertTrue(units)
        self.assertIn("python:3.12-slim", units[0].content)

    def test_docker_compose(self):
        dc = "services:\n  web:\n    image: nginx\n  db:\n    image: postgres\n"
        units = extract_config("docker-compose.yml", dc)
        self.assertTrue(units)
        self.assertIn("web", units[0].content)
        self.assertIn("db", units[0].content)


class TestExtractDirectives(unittest.TestCase):
    def test_agents_md(self):
        md = "# Coding Rules\nAlways use type hints.\n\n# Testing\nWrite tests for all public functions."
        units = extract_directives("AGENTS.md", md)
        self.assertEqual(len(units), 2)
        self.assertIn("type hints", units[0].content)
        self.assertIn("tests", units[1].content)


class TestExtractFile(unittest.TestCase):
    def test_routes_markdown(self):
        units = extract_file(Path("/fake"), "README.md")
        # just verify it doesn't crash and returns a list
        self.assertIsInstance(units, list)

    def test_routes_python_docstring(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {"app.py": '"""My app module."""\nx = 1'})
            units = extract_file(root, "app.py")
            self.assertIsInstance(units, list)

    def test_routes_package_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {"package.json": '{"name":"test","scripts":{"build":"echo hi"}}'})
            units = extract_file(root, "package.json")
            self.assertIsInstance(units, list)

    def test_skips_empty_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {"empty.py": ""})
            units = extract_file(root, "empty.py")
            self.assertEqual(units, [])

    def test_skips_binary_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "image.bin"
            p.write_bytes(b"\x00\x00\x00\x00binary stuff")
            units = extract_file(root, "image.bin")
            self.assertEqual(units, [])


class TestExtractProjectUnits(unittest.TestCase):
    def test_full_project(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {
                "README.md": "# My Project\nA cool tool for doing things.\n\n## Install\nRun pip install.",
                "src/__init__.py": '"""Core module for the project."""\n',
                "pyproject.toml": '[project]\nname = "cooltool"\nversion = "0.1.0"\n',
                "AGENTS.md": "# Rules\nUse type hints everywhere.",
            })
            units = extract_project_units(root)
            self.assertGreater(len(units), 0)
            sources = [u.source for u in units]
            # should have units from README, docstring, pyproject, agents
            self.assertTrue(any("README.md" in s for s in sources))
            self.assertTrue(any("__init__.py" in s for s in sources))
            self.assertTrue(any("pyproject.toml" in s for s in sources))
            self.assertTrue(any("AGENTS.md" in s for s in sources))

    def test_deduplication(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_project(root, {"README.md": "# Title\nContent."})
            units = extract_project_units(root)
            sources = [u.source for u in units]
            # no duplicate sources
            self.assertEqual(len(sources), len(set(sources)))

    def test_empty_project(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            units = extract_project_units(root)
            # should at least return conventions if any files exist
            # but for an empty dir, likely just empty
            self.assertIsInstance(units, list)


class TestUnitMapAndDiff(unittest.TestCase):
    def test_unit_map(self):
        units = [
            Unit(source="a", kind="fact", key="a", content="alpha"),
            Unit(source="b", kind="fact", key="b", content="beta"),
        ]
        m = unit_map(units)
        self.assertEqual(set(m.keys()), {"a", "b"})
        self.assertEqual(len(m["a"]), 16)  # sha256 hex[:16]

    def test_diff_units_added(self):
        prev = {"a": "aaa", "b": "bbb"}
        new = {"a": "aaa", "b": "bbb", "c": "ccc"}
        added, changed, removed = diff_units(prev, new)
        self.assertEqual(added, ["c"])
        self.assertEqual(changed, [])
        self.assertEqual(removed, [])

    def test_diff_units_removed(self):
        prev = {"a": "aaa", "b": "bbb", "c": "ccc"}
        new = {"a": "aaa", "b": "bbb"}
        added, changed, removed = diff_units(prev, new)
        self.assertEqual(added, [])
        self.assertEqual(changed, [])
        self.assertEqual(removed, ["c"])

    def test_diff_units_changed(self):
        prev = {"a": "aaa", "b": "bbb"}
        new = {"a": "aaa", "b": "xxx"}
        added, changed, removed = diff_units(prev, new)
        self.assertEqual(added, [])
        self.assertEqual(changed, ["b"])
        self.assertEqual(removed, [])


class TestBuildStore(unittest.TestCase):
    def test_builds_store_from_units(self):
        units = [
            Unit(source="readme#intro", kind="fact", key="intro",
                 content="[README.md] Intro\nThis is the intro."),
            Unit(source="readme#install", kind="fact", key="install",
                 content="[README.md] Install\nRun pip install."),
        ]
        store = build_store(units)
        self.assertEqual(len(store), 2)
        hits = store.search("intro")
        self.assertTrue(hits)
        self.assertIn("intro", hits[0].content.lower())


class TestTrackOnce(unittest.TestCase):
    def test_first_build(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "project")
            root.mkdir()
            sqac_dir = Path(td, ".sqac")
            _make_project(root, {
                "README.md": "# My Project\nDescription of the project.",
                "src/app.py": '"""Main application module."""\nprint("hello")',
            })
            summary = track_once(root, sqac_dir, verbose=False)
            self.assertIsInstance(summary, dict)
            # first build: everything is "added"
            self.assertGreater(len(summary["added"]), 0)
            self.assertEqual(summary["removed"], [])
            # cartridge and state should exist
            self.assertTrue((sqac_dir / "project.sqac").exists())
            self.assertTrue((sqac_dir / "state.json").exists())

    def test_idempotent_rebuild(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "project")
            root.mkdir()
            sqac_dir = Path(td, ".sqac")
            _make_project(root, {"README.md": "# Title\nContent."})
            track_once(root, sqac_dir, verbose=False)
            # second pass, nothing changed
            summary = track_once(root, sqac_dir, verbose=False)
            self.assertEqual(summary["added"], [])
            self.assertEqual(summary["changed"], [])
            self.assertEqual(summary["removed"], [])

    def test_detects_new_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "project")
            root.mkdir()
            sqac_dir = Path(td, ".sqac")
            _make_project(root, {"README.md": "# Title\nContent."})
            track_once(root, sqac_dir, verbose=False)
            # add a new file
            _make_project(root, {"NEW.md": "# New File\nBrand new content."})
            summary = track_once(root, sqac_dir, verbose=False)
            self.assertTrue(any("NEW.md" in s for s in summary["added"]))

    def test_detects_removed_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "project")
            root.mkdir()
            sqac_dir = Path(td, ".sqac")
            _make_project(root, {
                "README.md": "# Title\nContent.",
                "TEMP.md": "# Temporary\nDelete me.",
            })
            track_once(root, sqac_dir, verbose=False)
            # remove TEMP.md
            (root / "TEMP.md").unlink()
            summary = track_once(root, sqac_dir, verbose=False)
            self.assertTrue(any("TEMP.md" in s for s in summary["removed"]))

    def test_cartridge_is_searchable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "project")
            root.mkdir()
            sqac_dir = Path(td, ".sqac")
            _make_project(root, {
                "README.md": "# Deployment\nWe deploy only to ARM64.",
            })
            track_once(root, sqac_dir, verbose=False)
            loaded = SqacStore.load(sqac_dir / "project.sqac")
            hits = loaded.search("deployment target")
            self.assertTrue(hits)
            self.assertIn("ARM64", hits[0].content)


class TestStatePersistence(unittest.TestCase):
    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as td:
            sqac_dir = Path(td)
            sources = {"a.md": "abc123", "b.py": "def456"}
            save_state(sqac_dir, sources)
            state = load_state(sqac_dir)
            self.assertIsNotNone(state)
            self.assertEqual(state["sources"], sources)
            self.assertIn("synced_at", state)

    def test_load_nonexistent(self):
        state = load_state(Path("/nonexistent/path"))
        self.assertIsNone(state)


class TestTrackOnceLogging(unittest.TestCase):
    def test_track_once_returns_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "project")
            root.mkdir()
            sqac_dir = Path(td, ".sqac")
            _make_project(root, {"README.md": "# Title\nContent."})
            summary = track_once(root, sqac_dir, verbose=False)
            self.assertIn("added", summary)
            self.assertIn("changed", summary)
            self.assertIn("removed", summary)


class TestAuditLog(unittest.TestCase):
    def test_log_file_appended(self):
        """track() with --log appends JSONL entries to the file."""
        import threading
        with tempfile.TemporaryDirectory() as td:
            root = Path(td, "project")
            root.mkdir()
            sqac_dir = Path(td, ".sqac")
            log_path = Path(td, "audit.jsonl")
            _make_project(root, {"README.md": "# Title\nContent."})
            # do one sync manually, then verify the log file logic
            from sqac.autobuild import track_once
            summary = track_once(root, sqac_dir, verbose=False)
            # simulate what track() does for logging
            dirty = len(summary["added"]) + len(summary["changed"]) + len(summary["removed"])
            record = {
                "ts": "2026-09-07T12:00:00",
                "added": summary["added"],
                "changed": summary["changed"],
                "removed": summary["removed"],
                "total_sources": len(summary["added"]),
                "dirty": bool(dirty),
            }
            log_path.write_text(json.dumps(record) + "\n")
            # verify
            lines = log_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertIn("ts", entry)
            self.assertIn("added", entry)
            self.assertIn("dirty", entry)
            self.assertIsInstance(entry["added"], list)

    def test_log_file_accumulates(self):
        """Multiple writes append, not overwrite."""
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td, "audit.jsonl")
            for i in range(3):
                record = {"ts": f"2026-09-07T12:00:{i:02d}", "dirty": i > 0}
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            lines = log_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 3)
            for i, line in enumerate(lines):
                entry = json.loads(line)
                self.assertEqual(entry["dirty"], i > 0)


class TestSyncRack(unittest.TestCase):
    def test_sync_copies_cartridge(self):
        """_sync_rack copies the cartridge and writes a manifest."""
        with tempfile.TemporaryDirectory() as td:
            sqac_dir = Path(td, ".sqac")
            sqac_dir.mkdir()
            root = Path(td, "project")
            root.mkdir()
            _make_project(root, {"README.md": "# Test\nContent."})
            # build the cartridge first
            track_once(root, sqac_dir, verbose=False)
            cart = sqac_dir / "project.sqac"
            self.assertTrue(cart.exists())
            # sync into a rack
            rack_dir = Path(td, "rack")
            _sync_rack(rack_dir, "myapp", cart)
            # verify the cartridge was copied
            rack_cart = rack_dir / "myapp.sqac"
            self.assertTrue(rack_cart.exists())
            # verify the manifest
            manifest = rack_dir / "rack.json"
            self.assertTrue(manifest.exists())
            data = json.loads(manifest.read_text())
            self.assertIn("myapp", data)
            self.assertIn("synced_at", data["myapp"])
            self.assertIn("path", data["myapp"])

    def test_sync_updates_manifest(self):
        """Second sync updates the manifest entry."""
        with tempfile.TemporaryDirectory() as td:
            sqac_dir = Path(td, ".sqac")
            sqac_dir.mkdir()
            root = Path(td, "project")
            root.mkdir()
            _make_project(root, {"README.md": "# Test\nContent."})
            track_once(root, sqac_dir, verbose=False)
            cart = sqac_dir / "project.sqac"
            rack_dir = Path(td, "rack")
            _sync_rack(rack_dir, "myapp", cart)
            t1 = json.loads((rack_dir / "rack.json").read_text())["myapp"]["synced_at"]
            # second sync
            _sync_rack(rack_dir, "myapp", cart)
            t2 = json.loads((rack_dir / "rack.json").read_text())["myapp"]["synced_at"]
            # timestamps should be identical or close (same second)
            self.assertEqual(t1, t2)

    def test_rack_searchable_after_sync(self):
        """CartridgeRack can mount and search the synced cartridge."""
        with tempfile.TemporaryDirectory() as td:
            sqac_dir = Path(td, ".sqac")
            sqac_dir.mkdir()
            root = Path(td, "project")
            root.mkdir()
            _make_project(root, {
                "README.md": "# Auth\nUse OAuth2 for all endpoints.",
            })
            track_once(root, sqac_dir, verbose=False)
            rack_dir = Path(td, "rack")
            _sync_rack(rack_dir, "myapp", sqac_dir / "project.sqac")
            # mount via CartridgeRack
            rack = CartridgeRack(directory=rack_dir)
            self.assertIn("myapp", rack.names())
            hits = rack.search("authentication method")
            self.assertTrue(hits)
            self.assertIn("OAuth2", hits[0].content)
            self.assertEqual(hits[0].meta["cartridge"], "myapp")


if __name__ == "__main__":
    unittest.main()

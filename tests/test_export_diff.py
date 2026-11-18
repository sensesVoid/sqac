#!/usr/bin/env python3
"""Tests for sqac export / sqac diff (deliverable proposal 1)."""
import json
import tempfile
import unittest
from pathlib import Path

from sqac.export import export_cartridge, diff_cartridges, ExportError
from sqac.store import SqacStore


def _make_cartridge(path: Path, entries: list[tuple[str, str, str]]) -> None:
    """entries: (content, key, kind)"""
    store = SqacStore(semantic=False)
    for content, key, kind in entries:
        store.add(content, key=key, kind=kind, source="test")
    store.save(path, name=path.stem)


class TestExport(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)
        self.cart = self.d / "facts.sqac"
        _make_cartridge(self.cart, [
            ("deploy ARM64 first", "deploy-arch", "fact"),
            ("prefer tests before implementation", "tests-first", "skill"),
        ])

    def tearDown(self):
        self._td.cleanup()

    def test_export_md_contains_entries(self):
        out = export_cartridge(self.cart, fmt="md")
        self.assertIn("# SQAC Export — facts", out)
        self.assertIn("deploy ARM64 first", out)
        self.assertIn("prefer tests before implementation", out)
        self.assertIn("entries: 2", out)

    def test_export_json_roundtrip(self):
        out = export_cartridge(self.cart, fmt="json")
        data = json.loads(out)
        self.assertEqual(data["name"], "facts")
        self.assertEqual(data["entry_count"], 2)
        self.assertEqual(data["format"], "sqac-export/1")
        contents = {e["content"] for e in data["entries"]}
        self.assertIn("deploy ARM64 first", contents)

    def test_export_rejects_unknown_format(self):
        with self.assertRaises(ExportError):
            export_cartridge(self.cart, fmt="yaml")

    def test_export_refuses_encrypted(self):
        enc = self.d / "vault.sqac"
        enc.write_bytes(b"SQAE" + b"\x00" * 32)
        with self.assertRaises(ExportError) as ctx:
            export_cartridge(enc, fmt="md")
        self.assertIn("encrypted", str(ctx.exception))


class TestDiff(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)
        self.old = self.d / "old.sqac"
        self.new = self.d / "new.sqac"

    def tearDown(self):
        self._td.cleanup()

    def test_diff_detects_all_changes(self):
        _make_cartridge(self.old, [
            ("deploy ARM64 first", "deploy-arch", "fact"),
            ("use blue-green deploys", "blue-green", "fact"),
            ("run migrations nightly", "migrations", "fact"),
        ])
        _make_cartridge(self.new, [
            ("deploy ARM64 first", "deploy-arch", "fact"),      # unchanged
            ("use rolling deploys instead", "blue-green", "fact"),  # updated
            ("run migrations nightly", "migrations", "skill"),   # reclassified
            ("ship canary first", "canary", "fact"),             # added
        ])
        out = json.loads(diff_cartridges(self.old, self.new, fmt="json"))
        self.assertEqual(out["summary"]["added"], 1)
        self.assertEqual(out["summary"]["removed"], 0)
        self.assertEqual(out["summary"]["updated"], 1)
        self.assertEqual(out["summary"]["reclassified"], 1)
        self.assertEqual(out["added"][0]["content"], "ship canary first")
        self.assertEqual(out["updated"][0]["old"]["content"], "use blue-green deploys")
        self.assertEqual(out["reclassified"][0]["old"]["kind"], "fact")
        self.assertEqual(out["reclassified"][0]["new"]["kind"], "skill")

    def test_diff_counts_removals(self):
        _make_cartridge(self.old, [
            ("fact a", "a", "fact"),
            ("fact b", "b", "fact"),
        ])
        _make_cartridge(self.new, [("fact a", "a", "fact")])
        out = json.loads(diff_cartridges(self.old, self.new, fmt="json"))
        self.assertEqual(out["summary"]["removed"], 1)
        self.assertEqual(out["removed"][0]["content"], "fact b")

    def test_diff_md_output(self):
        _make_cartridge(self.old, [("fact a", "a", "fact")])
        _make_cartridge(self.new, [("fact a plus", "a", "fact"), ("new fact", "b", "fact")])
        out = diff_cartridges(self.old, self.new, fmt="md")
        self.assertIn("# SQAC Diff — old.sqac → new.sqac", out)
        self.assertIn("Added (1)", out)
        self.assertIn("Updated", out)

    def test_diff_rejects_encrypted(self):
        enc = self.d / "vault.sqac"
        enc.write_bytes(b"SQSE" + b"\x00" * 32)
        with self.assertRaises(ExportError):
            diff_cartridges(self.old, enc, fmt="json")


class TestExportCli(unittest.TestCase):
    def test_cli_export_and_diff(self):
        import io
        import contextlib
        from sqac.cli import main
        with tempfile.TemporaryDirectory() as td:
            cart = Path(td) / "facts.sqac"
            _make_cartridge(cart, [("deploy ARM64 first", "deploy-arch", "fact")])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(["export", str(cart), "--format", "md"])
            self.assertEqual(rc, 0)
            self.assertIn("deploy ARM64 first", buf.getvalue())

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(["diff", str(cart), str(cart), "--format", "json"])
            self.assertEqual(rc, 0)
            self.assertIn('"added": []', buf.getvalue())


if __name__ == "__main__":
    unittest.main()

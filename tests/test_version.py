"""The distribution version has exactly one source."""

import re
import unittest


class TestVersion(unittest.TestCase):
    def test_version_is_semver(self):
        from sqac._version import __version__

        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")

    def test_init_reexports_version(self):
        import sqac
        from sqac._version import __version__

        self.assertEqual(sqac.__version__, __version__)

    def test_pyproject_uses_dynamic_attr(self):
        from pathlib import Path

        text = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('dynamic = ["version"]', text)
        self.assertIn('version = {attr = "sqac._version.__version__"}', text)
        self.assertNotRegex(text, re.compile(r'(?m)^version = "\d'))

"""Tests for the SQAC code graph.

Run: python -m pytest tests/test_ast.py -q
"""

import tempfile
import unittest
from pathlib import Path

from sqac.graph import CodeGraph
from sqac.graph.models import EdgeKind, SymbolKind
from sqac.graph.parser import CodeParser, detect_language


class TestParser(unittest.TestCase):
    """Test tree-sitter based code parsing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_python_function_extraction(self):
        """Extract function definitions from Python files."""
        (self.root / "app.py").write_text('''
def handle_request():
    """Handle an incoming request."""
    process_data()

def process_data():
    pass
''')
        parser = CodeParser(self.root)
        symbols, edges, info = parser.parse_file("app.py")
        names = [s.name for s in symbols]
        self.assertIn("handle_request", names)
        self.assertIn("process_data", names)
        self.assertEqual(info.language, "python")

    def test_python_class_extraction(self):
        """Extract class definitions with methods."""
        (self.root / "models.py").write_text('''
class User:
    """A user model."""

    def __init__(self, name):
        self.name = name

    def save(self):
        pass
''')
        parser = CodeParser(self.root)
        symbols, edges, info = parser.parse_file("models.py")
        classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
        methods = [s for s in symbols if s.kind == SymbolKind.METHOD]
        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0].name, "User")
        self.assertEqual(classes[0].docstring, "A user model.")
        self.assertGreaterEqual(len(methods), 1)

    def test_python_inheritance(self):
        """Extract inheritance edges."""
        (self.root / "views.py").write_text('''
class BaseView:
    pass

class ApiView(BaseView):
    def get(self):
        pass
''')
        parser = CodeParser(self.root)
        symbols, edges, info = parser.parse_file("views.py")
        inherit_edges = [e for e in edges if e.kind == EdgeKind.INHERITS]
        self.assertGreater(len(inherit_edges), 0)

    def test_python_imports(self):
        """Extract import edges."""
        (self.root / "main.py").write_text('''
import os
from pathlib import Path
from . import utils
''')
        parser = CodeParser(self.root)
        symbols, edges, info = parser.parse_file("main.py")
        import_edges = [e for e in edges if e.kind == EdgeKind.IMPORTS]
        self.assertGreater(len(import_edges), 0)

    def test_python_calls(self):
        """Extract function call edges."""
        (self.root / "service.py").write_text('''
def handle():
    process()
    validate()

def process():
    pass

def validate():
    pass
''')
        parser = CodeParser(self.root)
        symbols, edges, info = parser.parse_file("service.py")
        call_edges = [e for e in edges if e.kind == EdgeKind.CALLS]
        self.assertGreater(len(call_edges), 0)

    def test_javascript_extraction(self):
        """Extract JS/TS functions and classes."""
        (self.root / "index.js").write_text('''
function handleRequest() {
    processData();
}

class UserService {
    constructor(name) {
        this.name = name;
    }
}

const API_URL = "https://api.example.com";
''')
        parser = CodeParser(self.root)
        symbols, edges, info = parser.parse_file("index.js")
        names = [s.name for s in symbols]
        self.assertIn("handleRequest", names)
        self.assertIn("UserService", names)
        self.assertIn("API_URL", names)
        self.assertEqual(info.language, "javascript")

    def test_language_detection(self):
        """Detect language from file extension."""
        self.assertEqual(detect_language(Path("app.py")), "python")
        self.assertEqual(detect_language(Path("index.ts")), "typescript")
        self.assertEqual(detect_language(Path("main.go")), "go")
        self.assertEqual(detect_language(Path("lib.rs")), "rust")
        self.assertIsNone(detect_language(Path("readme.md")))


class TestCodeGraph(unittest.TestCase):
    """Test the full code graph build + query pipeline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _create_project(self):
        """Create a sample Python project."""
        (self.root / "app.py").write_text('''
"""Main application module."""

def handle_request():
    """Handle incoming requests."""
    user = get_user()
    return process(user)

def get_user():
    """Fetch the current user."""
    return User("admin")

def process(user):
    """Process user data."""
    return user.name

class User:
    """User model."""

    def __init__(self, name):
        self.name = name

    def save(self):
        """Save user to database."""
        pass
''')
        (self.root / "utils.py").write_text('''
"""Utility functions."""

def format_name(name):
    """Format a name for display."""
    return name.strip().title()

def validate_email(email):
    """Validate an email address."""
    return "@" in email
''')

    def test_build_and_stats(self):
        """Build a graph and check stats."""
        self._create_project()
        graph = CodeGraph(self.root)
        summary = graph.build()
        self.assertGreater(summary["symbols"], 0)
        self.assertGreater(summary["files"], 0)
        stats = graph.stats()
        self.assertIn("python", stats["languages"])

    def test_explore(self):
        """Explore a symbol to get callers and callees."""
        self._create_project()
        graph = CodeGraph(self.root)
        graph.build()
        info = graph.explore("handle_request")
        self.assertTrue(info["found"])
        self.assertEqual(info["symbol"]["name"], "handle_request")
        # handle_request calls get_user and process
        callees = [c["name"] for c in info["callees"]]
        self.assertIn("get_user", callees)
        self.assertIn("process", callees)

    def test_callers(self):
        """Find who calls a function."""
        self._create_project()
        graph = CodeGraph(self.root)
        graph.build()
        callers = graph.callers("process")
        caller_names = [c["name"] for c in callers]
        self.assertIn("handle_request", caller_names)

    def test_blast_radius(self):
        """Find what breaks if we change a function."""
        self._create_project()
        graph = CodeGraph(self.root)
        graph.build()
        result = graph.blast_radius("get_user")
        self.assertTrue(result["found"])
        self.assertGreater(result["total_affected"], 0)
        affected_names = [a["name"] for a in result["affected"]]
        self.assertIn("handle_request", affected_names)

    def test_search(self):
        """Search symbols by name."""
        self._create_project()
        graph = CodeGraph(self.root)
        graph.build()
        results = graph.search("user")
        self.assertGreater(len(results), 0)
        names = [r["name"] for r in results]
        self.assertIn("User", names)

    def test_save_and_load(self):
        """Persist and reload the graph."""
        self._create_project()
        graph = CodeGraph(self.root)
        graph.build()
        stats_before = graph.stats()
        graph.save()

        # Reload
        graph2 = CodeGraph(self.root)
        graph2.load()
        stats_after = graph2.stats()
        self.assertEqual(stats_before["symbols"], stats_after["symbols"])
        self.assertEqual(stats_before["edges"], stats_after["edges"])

    def test_fuzzy_search(self):
        """Fuzzy search matches partial names."""
        self._create_project()
        graph = CodeGraph(self.root)
        graph.build()
        results = graph.search("handle")
        self.assertGreater(len(results), 0)
        names = [r["name"] for r in results]
        self.assertIn("handle_request", names)

    def test_class_docstring(self):
        """Docstrings are extracted from classes."""
        self._create_project()
        graph = CodeGraph(self.root)
        graph.build()
        user = graph.explore("User")
        self.assertTrue(user["found"])
        self.assertEqual(user["symbol"]["docstring"], "User model.")


if __name__ == "__main__":
    unittest.main()

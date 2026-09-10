"""Tree-sitter based code parser.

Extracts symbols (functions, classes, methods, variables) and edges
(calls, imports, inheritance) from source files using tree-sitter ASTs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from .models import (
    Edge,
    EdgeKind,
    FileInfo,
    Symbol,
    SymbolKind,
)

# Language detection by file extension
_EXT_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
}

# Files/dirs to skip
_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    ".sqac", ".codegraph", "dist", "build", ".tox", ".mypy_cache",
    ".pytest_cache", "egg-info", ".eggs",
}
_SKIP_EXTS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".lock", ".sum",
}


def detect_language(path: Path) -> Optional[str]:
    """Detect programming language from file extension."""
    return _EXT_MAP.get(path.suffix.lower())


def should_skip(path: Path) -> bool:
    """Check if a path should be skipped during traversal."""
    for part in path.parts:
        if part in _SKIP_DIRS:
            return True
    if path.suffix.lower() in _SKIP_EXTS:
        return True
    return False


def content_hash(data: bytes) -> str:
    """Fast content hash for change detection."""
    return hashlib.md5(data).hexdigest()[:16]


class CodeParser:
    """Parse source files and extract symbols + edges.

    Uses tree-sitter when available, falls back to regex-based extraction
    for unsupported languages.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self._parsers: dict[str, object] = {}

    def _get_parser(self, language: str):
        """Get or create a tree-sitter parser for the given language."""
        if language in self._parsers:
            return self._parsers[language]
        try:
            from tree_sitter_language_pack import get_parser
            parser = get_parser(language)
            self._parsers[language] = parser
            return parser
        except Exception:
            return None

    def parse_file(self, rel_path: str) -> tuple[list[Symbol], list[Edge], FileInfo]:
        """Parse a single source file and return symbols, edges, and file info."""
        abs_path = self.project_root / rel_path
        if not abs_path.exists():
            return [], [], FileInfo(path=rel_path, language="unknown")

        raw = abs_path.read_bytes()
        language = detect_language(abs_path)
        if language is None:
            return [], [], FileInfo(
                path=rel_path, language="unknown",
                size_bytes=len(raw),
            )

        stat = abs_path.stat()
        checksum = content_hash(raw)
        info = FileInfo(
            path=rel_path,
            language=language,
            size_bytes=len(raw),
            last_modified=stat.st_mtime,
            checksum=checksum,
        )

        parser = self._get_parser(language)
        if parser is None:
            # Fallback: regex-based extraction
            return self._parse_regex(rel_path, raw.decode(errors="replace"), language)

        tree = parser.parse(raw)
        symbols, edges = self._walk_tree(tree, rel_path, language, raw)
        info.symbol_count = len(symbols)
        info.edge_count = len(edges)
        return symbols, edges, info

    def _walk_tree(self, tree, rel_path: str, language: str, raw: bytes) -> tuple[list[Symbol], list[Edge]]:
        """Walk a tree-sitter AST and extract symbols + edges."""
        symbols: list[Symbol] = []
        edges: list[Edge] = []

        if language == "python":
            symbols, edges = self._walk_python(tree.root_node, rel_path, raw)
        elif language in ("javascript", "typescript"):
            symbols, edges = self._walk_js_ts(tree.root_node, rel_path, raw, language)
        elif language == "go":
            symbols, edges = self._walk_go(tree.root_node, rel_path, raw)
        elif language == "rust":
            symbols, edges = self._walk_rust(tree.root_node, rel_path, raw)
        else:
            # Generic fallback: extract any function/class definitions
            symbols, edges = self._walk_generic(tree.root_node, rel_path, raw, language)

        return symbols, edges

    # ── Python extraction ───────────────────────────────────────────────

    def _walk_python(self, node, rel_path: str, raw: bytes) -> tuple[list[Symbol], list[Edge]]:
        symbols: list[Symbol] = []
        edges: list[Edge] = []
        current_class: Optional[str] = None

        def _extract(node, parent_class: Optional[str] = None):
            nonlocal current_class

            if node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    sym_id = f"{rel_path}::{name}"
                    line_start = node.start_point[0] + 1
                    line_end = node.end_point[0] + 1
                    docstring = self._extract_docstring(node)
                    sym = Symbol(
                        id=sym_id, name=name, kind=SymbolKind.CLASS,
                        file_path=rel_path, line_start=line_start,
                        line_end=line_end, language="python",
                        docstring=docstring,
                    )
                    symbols.append(sym)
                    current_class = sym_id

                    # Extract inheritance
                    for base in self._get_bases(node):
                        base_id = self._resolve_name(base, rel_path)
                        edges.append(Edge(
                            source=sym_id, target=base_id,
                            kind=EdgeKind.INHERITS, file_path=rel_path,
                            line=line_start,
                        ))

                    # Walk children
                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            _extract(child, parent_class=sym_id)
                    current_class = None

            elif node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    prefix = f"{current_class}." if current_class else ""
                    sym_id = f"{rel_path}::{prefix}{name}"
                    kind = SymbolKind.METHOD if current_class else SymbolKind.FUNCTION
                    line_start = node.start_point[0] + 1
                    line_end = node.end_point[0] + 1
                    docstring = self._extract_docstring(node)
                    signature = self._extract_signature(node)
                    sym = Symbol(
                        id=sym_id, name=name, kind=kind,
                        file_path=rel_path, line_start=line_start,
                        line_end=line_end, language="python",
                        docstring=docstring, signature=signature,
                        parent=current_class,
                    )
                    symbols.append(sym)

                    # Walk body for calls
                    body = node.child_by_field_name("body")
                    if body:
                        self._extract_calls(body, sym_id, rel_path, edges)

            elif node.type == "decorated_definition":
                for child in node.children:
                    if child.type != "decorator":
                        _extract(child, parent_class=parent_class)

            # Recurse into children (skip already-processed nodes)
            if node.type not in ("function_definition", "class_definition", "decorated_definition"):
                for child in node.children:
                    _extract(child, parent_class=parent_class)

        _extract(node)

        # Extract imports
        self._extract_imports_py(node, rel_path, edges)

        return symbols, edges

    def _extract_docstring(self, func_node) -> Optional[str]:
        """Extract the docstring from a function/class node.

        Python docstrings appear as:
        - expression_statement > string (inside function body block)
        - string (direct child of class body block)
        """
        try:
            body = func_node.child_by_field_name("body")
            if body is None:
                return None
            for child in body.children:
                # Function docstring: expression_statement > string
                if child.type == "expression_statement":
                    expr = child.children[0] if child.children else None
                    if expr and expr.type == "string":
                        return self._clean_docstring(expr.text.decode())
                # Class docstring: string is a direct child of block
                elif child.type == "string":
                    return self._clean_docstring(child.text.decode())
        except Exception:
            pass
        return None

    def _clean_docstring(self, raw: str) -> str:
        """Strip triple quotes and whitespace from a docstring."""
        return raw.strip().strip('"').strip("'").strip()

    def _extract_signature(self, func_node) -> Optional[str]:
        """Extract the function signature."""
        try:
            params = func_node.child_by_field_name("parameters")
            if params:
                return params.text.decode()
        except Exception:
            pass
        return None

    def _get_bases(self, class_node) -> list[str]:
        """Get base class names from a class definition."""
        bases = []
        try:
            for child in class_node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type == "identifier":
                            bases.append(arg.text.decode())
                        elif arg.type == "attribute":
                            bases.append(arg.text.decode())
        except Exception:
            pass
        return bases

    def _resolve_name(self, name: str, rel_path: str) -> str:
        """Best-effort resolve a name to a symbol ID."""
        # If it looks like a module.Class, try to resolve
        if "." in name:
            parts = name.rsplit(".", 1)
            return f"{rel_path}::{parts[-1]}"
        return f"{rel_path}::{name}"

    def _extract_calls(self, node, caller_id: str, rel_path: str, edges: list[Edge]):
        """Walk a tree-sitter node and extract call edges recursively."""
        if node.type == "call":
            func = node.child_by_field_name("function")
            if func:
                name = func.text.decode().split("(")[0].split(".")[-1]
                edges.append(Edge(
                    source=caller_id,
                    target=f"{rel_path}::{name}",
                    kind=EdgeKind.CALLS,
                    file_path=rel_path,
                    line=node.start_point[0] + 1,
                ))
        # Recurse into all children
        for child in node.children:
            self._extract_calls(child, caller_id, rel_path, edges)

    def _extract_imports_py(self, root, rel_path: str, edges: list[Edge]):
        """Extract Python import statements as edges."""
        for child in root.children:
            if child.type == "import_statement":
                for alias in child.children:
                    if alias.type == "dotted_name":
                        name = alias.text.decode()
                        edges.append(Edge(
                            source=f"{rel_path}::(imports)",
                            target=f"{name}",
                            kind=EdgeKind.IMPORTS,
                            file_path=rel_path,
                            line=child.start_point[0] + 1,
                        ))
            elif child.type == "import_from_statement":
                module_node = child.children[0] if child.children else None
                if module_node and module_node.type == "dotted_name":
                    module = module_node.text.decode()
                    for name_node in child.children:
                        if name_node.type == "dotted_name" or name_node.type == "identifier":
                            name = name_node.text.decode()
                            edges.append(Edge(
                                source=f"{rel_path}::(imports)",
                                target=f"{module}.{name}",
                                kind=EdgeKind.IMPORTS,
                                file_path=rel_path,
                                line=child.start_point[0] + 1,
                            ))

    # ── JavaScript/TypeScript extraction ─────────────────────────────────

    def _walk_js_ts(self, node, rel_path: str, raw: bytes, language: str) -> tuple[list[Symbol], list[Edge]]:
        """Extract symbols from JS/TS AST."""
        symbols: list[Symbol] = []
        edges: list[Edge] = []

        def _extract(node, parent_class: Optional[str] = None):
            if node.type in ("function_declaration", "function"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    prefix = f"{parent_class}." if parent_class else ""
                    sym_id = f"{rel_path}::{prefix}{name}"
                    kind = SymbolKind.METHOD if parent_class else SymbolKind.FUNCTION
                    sym = Symbol(
                        id=sym_id, name=name, kind=kind,
                        file_path=rel_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language=language,
                        parent=parent_class,
                    )
                    symbols.append(sym)
                    # Extract calls from body
                    body = node.child_by_field_name("body")
                    if body:
                        self._extract_calls_js(body, sym_id, rel_path, edges)

            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    sym_id = f"{rel_path}::{name}"
                    sym = Symbol(
                        id=sym_id, name=name, kind=SymbolKind.CLASS,
                        file_path=rel_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        language=language,
                    )
                    symbols.append(sym)
                    # Walk class body
                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            _extract(child, parent_class=sym_id)

            elif node.type == "lexical_declaration":
                # const/let/var
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            name = name_node.text.decode()
                            sym = Symbol(
                                id=f"{rel_path}::{name}", name=name,
                                kind=SymbolKind.VARIABLE,
                                file_path=rel_path,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                language=language,
                            )
                            symbols.append(sym)

            # Recurse
            if node.type not in ("function_declaration", "class_declaration"):
                for child in node.children:
                    _extract(child, parent_class=parent_class)

        _extract(node)

        # Extract imports
        self._extract_imports_js(node, rel_path, edges)

        return symbols, edges

    def _extract_calls_js(self, body_node, caller_id: str, rel_path: str, edges: list[Edge]):
        """Extract function calls from JS/TS body."""
        for child in body_node.children:
            if child.type == "expression_statement":
                expr = child.children[0] if child.children else None
                if expr and expr.type == "call_expression":
                    func = expr.child_by_field_name("function")
                    if func:
                        name = func.text.decode().split("(")[0].split(".")[-1]
                        edges.append(Edge(
                            source=caller_id,
                            target=f"{rel_path}::{name}",
                            kind=EdgeKind.CALLS,
                            file_path=rel_path,
                            line=child.start_point[0] + 1,
                        ))
            elif hasattr(child, "children"):
                self._extract_calls_js(child, caller_id, rel_path, edges)

    def _extract_imports_js(self, root, rel_path: str, edges: list[Edge]):
        """Extract JS/TS import statements."""
        for child in root.children:
            if child.type == "import_statement":
                source = child.child_by_field_name("source")
                if source:
                    module = source.text.decode().strip("'\"")
                    edges.append(Edge(
                        source=f"{rel_path}::(imports)",
                        target=module,
                        kind=EdgeKind.IMPORTS,
                        file_path=rel_path,
                        line=child.start_point[0] + 1,
                    ))

    # ── Go extraction ───────────────────────────────────────────────────

    def _walk_go(self, node, rel_path: str, raw: bytes) -> tuple[list[Symbol], list[Edge]]:
        """Extract symbols from Go AST."""
        symbols: list[Symbol] = []
        edges: list[Edge] = []

        for child in node.children:
            if child.type == "function_declaration":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    sym = Symbol(
                        id=f"{rel_path}::{name}", name=name,
                        kind=SymbolKind.FUNCTION,
                        file_path=rel_path,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language="go",
                    )
                    symbols.append(sym)
            elif child.type == "type_declaration":
                # struct/interface
                for type_spec in child.children:
                    if type_spec.type == "type_spec":
                        name_node = type_spec.child_by_field_name("name")
                        if name_node:
                            name = name_node.text.decode()
                            kind = SymbolKind.CLASS  # struct ≈ class
                            sym = Symbol(
                                id=f"{rel_path}::{name}", name=name,
                                kind=kind,
                                file_path=rel_path,
                                line_start=child.start_point[0] + 1,
                                line_end=child.end_point[0] + 1,
                                language="go",
                            )
                            symbols.append(sym)
            elif child.type == "import_declaration":
                for spec in child.children:
                    if spec.type == "import_spec":
                        path_node = spec.child_by_field_name("path")
                        if path_node:
                            module = path_node.text.decode().strip('"')
                            edges.append(Edge(
                                source=f"{rel_path}::(imports)",
                                target=module,
                                kind=EdgeKind.IMPORTS,
                                file_path=rel_path,
                                line=child.start_point[0] + 1,
                            ))

        return symbols, edges

    # ── Rust extraction ─────────────────────────────────────────────────

    def _walk_rust(self, node, rel_path: str, raw: bytes) -> tuple[list[Symbol], list[Edge]]:
        """Extract symbols from Rust AST."""
        symbols: list[Symbol] = []
        edges: list[Edge] = []

        for child in node.children:
            if child.type == "function_item":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode()
                    sym = Symbol(
                        id=f"{rel_path}::{name}", name=name,
                        kind=SymbolKind.FUNCTION,
                        file_path=rel_path,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language="rust",
                    )
                    symbols.append(sym)
            elif child.type == "impl_item":
                # Extract methods from impl blocks
                for method in child.children:
                    if method.type == "function_item":
                        name_node = method.child_by_field_name("name")
                        if name_node:
                            name = name_node.text.decode()
                            sym = Symbol(
                                id=f"{rel_path}::{name}", name=name,
                                kind=SymbolKind.METHOD,
                                file_path=rel_path,
                                line_start=method.start_point[0] + 1,
                                line_end=method.end_point[0] + 1,
                                language="rust",
                            )
                            symbols.append(sym)
            elif child.type == "use_declaration":
                # Extract use statements
                use_path = child.child_by_field_name("path")
                if use_path:
                    edges.append(Edge(
                        source=f"{rel_path}::(imports)",
                        target=use_path.text.decode(),
                        kind=EdgeKind.IMPORTS,
                        file_path=rel_path,
                        line=child.start_point[0] + 1,
                    ))

        return symbols, edges

    # ── Generic fallback ────────────────────────────────────────────────

    def _walk_generic(self, node, rel_path: str, raw: bytes, language: str) -> tuple[list[Symbol], list[Edge]]:
        """Generic extraction for unsupported languages — find any
        function/class-like nodes."""
        symbols: list[Symbol] = []
        edges: list[Edge] = []

        _FUNC_TYPES = {"function_definition", "function_declaration", "method_definition",
                       "function_item", "funcdef"}
        _CLASS_TYPES = {"class_definition", "class_declaration", "class_definition",
                        "struct_declaration", "struct_item"}

        for child in node.children:
            if child.type in _FUNC_TYPES:
                # Try to find a name child
                name = self._find_name_child(child)
                if name:
                    sym = Symbol(
                        id=f"{rel_path}::{name}", name=name,
                        kind=SymbolKind.FUNCTION,
                        file_path=rel_path,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language=language,
                    )
                    symbols.append(sym)
            elif child.type in _CLASS_TYPES:
                name = self._find_name_child(child)
                if name:
                    sym = Symbol(
                        id=f"{rel_path}::{name}", name=name,
                        kind=SymbolKind.CLASS,
                        file_path=rel_path,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        language=language,
                    )
                    symbols.append(sym)
            elif hasattr(child, "children"):
                sub_syms, sub_edges = self._walk_generic(child, rel_path, raw, language)
                symbols.extend(sub_syms)
                edges.extend(sub_edges)

        return symbols, edges

    def _find_name_child(self, node) -> Optional[str]:
        """Try to find a name identifier in a node's children."""
        for child in node.children:
            if child.type in ("identifier", "name", "field_identifier"):
                return child.text.decode()
        return None

    # ── Regex fallback ──────────────────────────────────────────────────

    def _parse_regex(self, rel_path: str, text: str, language: str) -> tuple[list[Symbol], list[Edge], FileInfo]:
        """Fallback regex-based extraction when tree-sitter is unavailable."""
        import re
        symbols: list[Symbol] = []
        edges: list[Edge] = []
        info = FileInfo(path=rel_path, language=language, size_bytes=len(text.encode()))

        if language == "python":
            # Match def/class
            for m in re.finditer(r'^(class|def)\s+(\w+)', text, re.MULTILINE):
                kind_str, name = m.group(1), m.group(2)
                line = text[:m.start()].count("\n") + 1
                kind = SymbolKind.CLASS if kind_str == "class" else SymbolKind.FUNCTION
                symbols.append(Symbol(
                    id=f"{rel_path}::{name}", name=name, kind=kind,
                    file_path=rel_path, line_start=line, line_end=line,
                    language=language,
                ))
        elif language in ("javascript", "typescript"):
            for m in re.finditer(r'^(function|class|const|let|var)\s+(\w+)', text, re.MULTILINE):
                kind_str, name = m.group(1), m.group(2)
                line = text[:m.start()].count("\n") + 1
                if kind_str == "class":
                    kind = SymbolKind.CLASS
                elif kind_str == "function":
                    kind = SymbolKind.FUNCTION
                else:
                    kind = SymbolKind.VARIABLE
                symbols.append(Symbol(
                    id=f"{rel_path}::{name}", name=name, kind=kind,
                    file_path=rel_path, line_start=line, line_end=line,
                    language=language,
                ))

        info.symbol_count = len(symbols)
        info.edge_count = len(edges)
        return symbols, edges, info

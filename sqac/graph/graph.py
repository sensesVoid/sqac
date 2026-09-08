"""CodeGraph — build, query, and persist a code knowledge graph.

The graph stores symbols (functions, classes, methods) and edges (calls,
imports, inheritance) extracted from source files via tree-sitter. It
supports:

- build(): parse all source files and populate the graph
- explore(symbol): get definition + callers + callees
- blast_radius(symbol): what breaks if I change this?
- callers(symbol) / callees(symbol): direct relationships
- search(query): fuzzy search across symbol names and docstrings
- save() / load(): persist to .sqac-graph/graph.json
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .models import Edge, EdgeKind, FileInfo, Symbol, SymbolKind
from .parser import CodeParser, detect_language, should_skip

logger = logging.getLogger(__name__)

GRAPH_VERSION = 1


class CodeGraph:
    """A queryable knowledge graph of a codebase.

    Usage:
        graph = CodeGraph("/path/to/project")
        graph.build()
        graph.save()

        info = graph.explore("handle_request")
        broken = graph.blast_radius("handle_request")
    """

    def __init__(self, project_root: str | Path, graph_dir: Optional[str | Path] = None):
        self.project_root = Path(project_root).resolve()
        if graph_dir:
            self.graph_dir = Path(graph_dir)
        else:
            # Default: store in the project's .sqac-graph/ directory
            # When used via MCP, graph_dir is set to <mem_dir>/ast/
            self.graph_dir = self.project_root / ".sqac-graph"
        self._symbols: dict[str, Symbol] = {}     # id -> Symbol
        self._edges: list[Edge] = []
        self._files: dict[str, FileInfo] = {}     # path -> FileInfo
        # Indexes for fast queries
        self._callers: dict[str, set[str]] = defaultdict(set)  # target -> {sources}
        self._callees: dict[str, set[str]] = defaultdict(set)  # source -> {targets}
        self._name_index: dict[str, list[str]] = defaultdict(list)  # name -> [symbol_ids]
        self._file_symbols: dict[str, list[str]] = defaultdict(list)  # file -> [symbol_ids]
        self._parser = CodeParser(self.project_root)

    # ── build ───────────────────────────────────────────────────────────

    def build(self, include_patterns: Optional[list[str]] = None,
              exclude_patterns: Optional[list[str]] = None) -> dict[str, Any]:
        """Parse all source files and populate the graph.

        Returns a summary dict with counts.
        """
        self._symbols.clear()
        self._edges.clear()
        self._files.clear()
        self._callers.clear()
        self._callees.clear()
        self._name_index.clear()
        self._file_symbols.clear()

        files_scanned = 0
        files_parsed = 0
        errors = 0

        for rel_path in self._walk_project():
            files_scanned += 1
            try:
                symbols, edges, info = self._parser.parse_file(rel_path)
                self._files[rel_path] = info
                for sym in symbols:
                    self._symbols[sym.id] = sym
                    self._name_index[sym.name].append(sym.id)
                    self._file_symbols[rel_path].append(sym.id)
                for edge in edges:
                    self._edges.append(edge)
                    if edge.kind == EdgeKind.CALLS:
                        self._callers[edge.target].add(edge.source)
                        self._callees[edge.source].add(edge.target)
                files_parsed += 1
            except Exception as exc:
                errors += 1
                logger.warning("sqac.graph: failed to parse %s: %s", rel_path, exc)

        summary = {
            "project": str(self.project_root),
            "files_scanned": files_scanned,
            "files_parsed": files_parsed,
            "files": files_parsed,
            "symbols": len(self._symbols),
            "edges": len(self._edges),
            "errors": errors,
        }
        logger.info("sqac.graph: built — %s", summary)
        return summary

    def _walk_project(self):
        """Yield relative paths of source files in the project."""
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden dirs and common non-source dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in (
                "node_modules", "__pycache__", ".venv", "venv",
                ".sqac", ".sqac-graph", "dist", "build", ".tox",
                ".mypy_cache", ".pytest_cache",
            )]
            for fname in files:
                fpath = Path(root) / fname
                if should_skip(fpath.relative_to(self.project_root)):
                    continue
                lang = detect_language(fpath)
                if lang is not None:
                    yield fpath.relative_to(self.project_root).as_posix()

    # ── query ───────────────────────────────────────────────────────────

    def explore(self, name: str) -> dict[str, Any]:
        """Get full info about a symbol: definition, callers, callees."""
        sym_ids = self._name_index.get(name, [])
        if not sym_ids:
            # Try fuzzy match
            for sym_name, ids in self._name_index.items():
                if name.lower() in sym_name.lower():
                    sym_ids = ids
                    break
        if not sym_ids:
            return {"found": False, "query": name}

        sym = self._symbols[sym_ids[0]]
        caller_ids = self._callers.get(sym.id, set())
        callee_ids = self._callees.get(sym.id, set())

        return {
            "found": True,
            "symbol": sym.to_dict(),
            "callers": [self._symbols[cid].to_dict() for cid in caller_ids if cid in self._symbols],
            "callees": [self._symbols[cid].to_dict() for cid in callee_ids if cid in self._symbols],
            "caller_count": len(caller_ids),
            "callee_count": len(callee_ids),
        }

    def blast_radius(self, name: str, max_depth: int = 3) -> dict[str, Any]:
        """Find all symbols that would be affected by changing this symbol.

        Traverses the call graph upstream (callers of callers) up to
        max_depth levels.
        """
        sym_ids = self._name_index.get(name, [])
        if not sym_ids:
            for sym_name, ids in self._name_index.items():
                if name.lower() in sym_name.lower():
                    sym_ids = ids
                    break
        if not sym_ids:
            return {"found": False, "query": name, "affected": []}

        affected: dict[str, int] = {}  # sym_id -> depth
        queue = [(sym_ids[0], 0)]
        visited = set()

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)
            affected[current_id] = depth

            # Find callers of this symbol
            for caller_id in self._callers.get(current_id, set()):
                if caller_id not in visited:
                    queue.append((caller_id, depth + 1))

        # Also include direct callees (what this function depends on)
        for callee_id in self._callees.get(sym_ids[0], set()):
            if callee_id not in affected:
                affected[callee_id] = -1  # -1 = dependency

        return {
            "found": True,
            "symbol": self._symbols[sym_ids[0]].to_dict(),
            "affected": [
                {**self._symbols[sid].to_dict(), "depth": depth}
                for sid, depth in sorted(affected.items(), key=lambda x: x[1])
                if sid in self._symbols
            ],
            "total_affected": len(affected),
        }

    def callers(self, name: str) -> list[dict[str, Any]]:
        """Get all symbols that call the named symbol."""
        sym_ids = self._name_index.get(name, [])
        if not sym_ids:
            return []
        caller_ids = self._callers.get(sym_ids[0], set())
        return [self._symbols[cid].to_dict() for cid in caller_ids if cid in self._symbols]

    def callees(self, name: str) -> list[dict[str, Any]]:
        """Get all symbols called by the named symbol."""
        sym_ids = self._name_index.get(name, [])
        if not sym_ids:
            return []
        callee_ids = self._callees.get(sym_ids[0], set())
        return [self._symbols[cid].to_dict() for cid in callee_ids if cid in self._symbols]

    def search(self, query: str, kind: Optional[str] = None, top_k: int = 20) -> list[dict[str, Any]]:
        """Search symbols by name or docstring content."""
        results: list[tuple[float, Symbol]] = []
        query_lower = query.lower()

        for sym in self._symbols.values():
            if kind and sym.kind.value != kind:
                continue
            score = 0.0
            if query_lower in sym.name.lower():
                score = 1.0 if sym.name.lower() == query_lower else 0.8
            elif sym.docstring and query_lower in sym.docstring.lower():
                score = 0.5
            elif query_lower in sym.file_path.lower():
                score = 0.3
            if score > 0:
                results.append((score, sym))

        results.sort(key=lambda x: (-x[0], x[1].name))
        return [sym.to_dict() for _, sym in results[:top_k]]

    def file_info(self, path: str) -> Optional[dict[str, Any]]:
        """Get metadata about a source file."""
        info = self._files.get(path)
        if info is None:
            return None
        sym_ids = self._file_symbols.get(path, [])
        return {
            "file": info.to_dict(),
            "symbols": [self._symbols[sid].to_dict() for sid in sym_ids if sid in self._symbols],
        }

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: Optional[str | Path] = None) -> None:
        """Persist the graph to disk."""
        save_path = Path(path) if path else self.graph_dir / "graph.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": GRAPH_VERSION,
            "project": str(self.project_root),
            "symbols": [s.to_dict() for s in self._symbols.values()],
            "edges": [e.to_dict() for e in self._edges],
            "files": {p: fi.to_dict() for p, fi in self._files.items()},
        }
        tmp = save_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(save_path)
        logger.info("sqac.graph: saved to %s (%d symbols, %d edges)",
                     save_path, len(self._symbols), len(self._edges))

    def load(self, path: Optional[str | Path] = None) -> bool:
        """Load the graph from disk. Returns True if successful."""
        load_path = Path(path) if path else self.graph_dir / "graph.json"
        if not load_path.exists():
            return False
        try:
            data = json.loads(load_path.read_text(encoding="utf-8"))
            self._symbols.clear()
            self._edges.clear()
            self._files.clear()
            self._callers.clear()
            self._callees.clear()
            self._name_index.clear()
            self._file_symbols.clear()

            for sd in data.get("symbols", []):
                sym = Symbol.from_dict(sd)
                self._symbols[sym.id] = sym
                self._name_index[sym.name].append(sym.id)
                self._file_symbols[sym.file_path].append(sym.id)
            for ed in data.get("edges", []):
                edge = Edge.from_dict(ed)
                self._edges.append(edge)
                if edge.kind == EdgeKind.CALLS:
                    self._callers[edge.target].add(edge.source)
                    self._callees[edge.source].add(edge.target)
            for fp, fid in data.get("files", {}).items():
                self._files[fp] = FileInfo.from_dict(fid)

            logger.info("sqac.graph: loaded %d symbols, %d edges from %s",
                         len(self._symbols), len(self._edges), load_path)
            return True
        except Exception as exc:
            logger.warning("sqac.graph: failed to load %s: %s", load_path, exc)
            return False

    # ── stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        kind_counts: dict[str, int] = defaultdict(int)
        for sym in self._symbols.values():
            kind_counts[sym.kind.value] += 1
        edge_counts: dict[str, int] = defaultdict(int)
        for edge in self._edges:
            edge_counts[edge.kind.value] += 1
        lang_counts: dict[str, int] = defaultdict(int)
        for sym in self._symbols.values():
            lang_counts[sym.language] += 1
        return {
            "symbols": len(self._symbols),
            "edges": len(self._edges),
            "files": len(self._files),
            "symbol_kinds": dict(kind_counts),
            "edge_kinds": dict(edge_counts),
            "languages": dict(lang_counts),
        }

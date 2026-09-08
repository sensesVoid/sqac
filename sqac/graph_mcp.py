"""MCP tools for sqac ast — AST-level code intelligence.

Adds ast_init, ast_explore, ast_blast, ast_callers, ast_callees,
ast_search, and ast_stats tools to the SQAC MCP server.

Graph data is persisted to <mem_dir>/ast/ within the SQAC memory
directory, so it lives alongside facts/, skills/, docs/ in the rack.
"""

from __future__ import annotations

import functools
import json
import os
import threading
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import Field

from .graph import CodeGraph

# Shared graph state — lazily initialized per project
_GRAPH_LOCK = threading.RLock()
_graphs: dict[str, CodeGraph] = {}


def _get_graph(project_root: Optional[str] = None) -> CodeGraph:
    """Get or create a CodeGraph for the given project root.

    Stores graph data in <mem_dir>/ast/ so it lives within the rack.
    Falls back to <project>/.sqac-graph/ if no SQAC_MEM_DIR is set.
    """
    root = Path(project_root or os.getcwd()).resolve()
    key = str(root)
    if key not in _graphs:
        # Determine where to store the graph
        mem_dir = os.environ.get("SQAC_MEM_DIR")
        if mem_dir:
            graph_dir = Path(mem_dir) / "ast"
        else:
            graph_dir = root / ".sqac-graph"
        graph = CodeGraph(root, graph_dir=graph_dir)
        # Try to load existing graph
        if not graph.load():
            graph.build()
            graph.save()
        _graphs[key] = graph
    return _graphs[key]


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def _err(exc: Exception) -> str:
    return f"Error: {type(exc).__name__}: {exc}"


def graph_tool(fn):
    """Decorator for AST MCP tools — thread-safe."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _GRAPH_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def register_graph_tools(mcp_server):
    """Register all AST tools on the given MCP server."""

    @mcp_server.tool(
        name="ast_init",
        annotations={
            "title": "Build Code Structure Index",
            "read_only_hint": False,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    @graph_tool
    def ast_init(
        project: Annotated[Optional[str], Field(description="Project root directory (default: cwd)")] = None,
    ) -> str:
        """Build or rebuild the structural index for a codebase.

        Parses all source files, extracts symbols (functions, classes,
        methods) and edges (calls, imports, inheritance), and persists
        to <mem_dir>/ast/. Returns summary stats.
        """
        try:
            root = Path(project or os.getcwd()).resolve()
            mem_dir = os.environ.get("SQAC_MEM_DIR")
            if mem_dir:
                graph_dir = Path(mem_dir) / "ast"
            else:
                graph_dir = root / ".sqac-graph"
            graph = CodeGraph(root, graph_dir=graph_dir)
            summary = graph.build()
            graph.save()
            _graphs[str(root)] = graph
            return _ok({"status": "built", "graph_dir": str(graph_dir), **summary})
        except Exception as e:
            return _err(e)

    @mcp_server.tool(
        name="ast_explore",
        annotations={
            "title": "Explore a Code Symbol",
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": True,
        },
    )
    @graph_tool
    def ast_explore(
        symbol: Annotated[str, Field(description="Symbol name to explore (e.g. 'handle_request', 'User')")],
        project: Annotated[Optional[str], Field(description="Project root directory (default: cwd)")] = None,
    ) -> str:
        """Explore a code symbol: get its definition, docstring, callers, and callees.

        Returns the symbol's file, line range, parent class (if method),
        and the full list of who calls it and what it calls.
        """
        try:
            graph = _get_graph(project)
            result = graph.explore(symbol)
            if not result["found"]:
                return _ok({"found": False, "query": symbol, "hint": "Try ast_search to find similar symbols"})
            return _ok(result)
        except Exception as e:
            return _err(e)

    @mcp_server.tool(
        name="ast_blast",
        annotations={
            "title": "Blast Radius Analysis",
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": True,
        },
    )
    @graph_tool
    def ast_blast(
        symbol: Annotated[str, Field(description="Symbol to analyze blast radius for")],
        depth: Annotated[int, Field(ge=1, le=10, description="Max traversal depth (default: 3)")] = 3,
        project: Annotated[Optional[str], Field(description="Project root directory (default: cwd)")] = None,
    ) -> str:
        """Find all symbols affected by changing this symbol.

        Traverses the call graph upstream (callers of callers) up to
        `depth` levels. Returns the list of affected symbols with their
        depth from the change point.
        """
        try:
            graph = _get_graph(project)
            result = graph.blast_radius(symbol, max_depth=depth)
            if not result["found"]:
                return _ok({"found": False, "query": symbol})
            return _ok(result)
        except Exception as e:
            return _err(e)

    @mcp_server.tool(
        name="ast_callers",
        annotations={
            "title": "Find Callers of a Symbol",
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": True,
        },
    )
    @graph_tool
    def ast_callers(
        symbol: Annotated[str, Field(description="Symbol name to find callers for")],
        project: Annotated[Optional[str], Field(description="Project root directory (default: cwd)")] = None,
    ) -> str:
        """Find all symbols that call the named symbol."""
        try:
            graph = _get_graph(project)
            callers = graph.callers(symbol)
            return _ok({"symbol": symbol, "callers": callers, "count": len(callers)})
        except Exception as e:
            return _err(e)

    @mcp_server.tool(
        name="ast_callees",
        annotations={
            "title": "Find Callees of a Symbol",
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": True,
        },
    )
    @graph_tool
    def ast_callees(
        symbol: Annotated[str, Field(description="Symbol name to find callees for")],
        project: Annotated[Optional[str], Field(description="Project root directory (default: cwd)")] = None,
    ) -> str:
        """Find all symbols called by the named symbol."""
        try:
            graph = _get_graph(project)
            callees = graph.callees(symbol)
            return _ok({"symbol": symbol, "callees": callees, "count": len(callees)})
        except Exception as e:
            return _err(e)

    @mcp_server.tool(
        name="ast_search",
        annotations={
            "title": "Search Code Symbols",
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": True,
        },
    )
    @graph_tool
    def ast_search(
        query: Annotated[str, Field(description="Search query (matches symbol names and docstrings)")],
        kind: Annotated[Optional[str], Field(description="Filter by kind: function, class, method, variable")] = None,
        top_k: Annotated[int, Field(ge=1, le=100, description="Max results (default: 20)")] = 20,
        project: Annotated[Optional[str], Field(description="Project root directory (default: cwd)")] = None,
    ) -> str:
        """Search across all code symbols by name or docstring content."""
        try:
            graph = _get_graph(project)
            results = graph.search(query, kind=kind, top_k=top_k)
            return _ok({"query": query, "results": results, "count": len(results)})
        except Exception as e:
            return _err(e)

    @mcp_server.tool(
        name="ast_stats",
        annotations={
            "title": "Code Structure Statistics",
            "read_only_hint": True,
            "destructive_hint": False,
            "idempotent_hint": True,
            "open_world_hint": False,
        },
    )
    @graph_tool
    def ast_stats(
        project: Annotated[Optional[str], Field(description="Project root directory (default: cwd)")] = None,
    ) -> str:
        """Get statistics about the code structure index: symbol counts, edge counts, languages."""
        try:
            graph = _get_graph(project)
            return _ok(graph.stats())
        except Exception as e:
            return _err(e)

"""SQAC Graph — AST-level code intelligence.

Parses source files with tree-sitter, extracts symbols and call edges,
and builds a queryable knowledge graph. Answers "what calls X?", "what
does X call?", "what breaks if I change X?" — the structural questions
that fuzzy text search can't handle.

Usage:
    from sqac.graph import CodeGraph

    graph = CodeGraph("/path/to/project")
    graph.build()                          # parse all source files
    graph.save()                           # persist to .sqac-graph/

    # Query
    graph.explore("my_function")           # definition + callers + callees
    graph.blast_radius("my_function")      # what breaks if I change this
    graph.callers("handle_request")        # who calls this?
    graph.callees("process_data")          # what does this call?
"""

from .models import Symbol, Edge, SymbolKind, EdgeKind, FileInfo
from .graph import CodeGraph

__all__ = [
    "Symbol",
    "Edge",
    "SymbolKind",
    "EdgeKind",
    "FileInfo",
    "CodeGraph",
]

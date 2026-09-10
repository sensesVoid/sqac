"""Data models for the code graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SymbolKind(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    VARIABLE = "variable"
    CONSTANT = "constant"
    MODULE = "module"
    PARAMETER = "parameter"
    DECORATOR = "decorator"
    MIXIN = "mixin"
    INTERFACE = "interface"


class EdgeKind(str, Enum):
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    DECORATES = "decorates"
    USES = "uses"           # variable/constant reference
    DEFINES = "defines"     # module defines symbol
    RETURNS = "returns"
    YIELDS = "yields"
    RAISES = "raises"


@dataclass
class Symbol:
    """A code symbol (function, class, method, variable, etc.)."""
    id: str                          # unique: "file_path::symbol_name" or "file_path::Class.method"
    name: str                        # short name (e.g. "handle_request")
    kind: SymbolKind                 # function, class, method, etc.
    file_path: str                   # relative path from project root
    line_start: int                  # 1-indexed
    line_end: int                    # 1-indexed, inclusive
    language: str                    # "python", "typescript", etc.
    docstring: Optional[str] = None  # extracted docstring/comment
    signature: Optional[str] = None  # function signature (parameters)
    parent: Optional[str] = None     # enclosing class/method id (for methods)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "language": self.language,
            "docstring": self.docstring,
            "signature": self.signature,
            "parent": self.parent,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Symbol":
        return cls(
            id=d["id"],
            name=d["name"],
            kind=SymbolKind(d["kind"]),
            file_path=d["file_path"],
            line_start=d["line_start"],
            line_end=d["line_end"],
            language=d["language"],
            docstring=d.get("docstring"),
            signature=d.get("signature"),
            parent=d.get("parent"),
            meta=d.get("meta", {}),
        )


@dataclass
class Edge:
    """A relationship between two symbols."""
    source: str      # symbol id of the caller/importer
    target: str      # symbol id of the callee/import
    kind: EdgeKind   # calls, imports, inherits, etc.
    file_path: str   # file where the edge originates
    line: int        # line number (1-indexed)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "file_path": self.file_path,
            "line": self.line,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Edge":
        return cls(
            source=d["source"],
            target=d["target"],
            kind=EdgeKind(d["kind"]),
            file_path=d["file_path"],
            line=d["line"],
            meta=d.get("meta", {}),
        )


@dataclass
class FileInfo:
    """Metadata about a parsed source file."""
    path: str                        # relative path
    language: str                    # detected language
    size_bytes: int = 0
    last_modified: float = 0.0       # epoch timestamp
    symbol_count: int = 0
    edge_count: int = 0
    checksum: Optional[str] = None   # content hash for change detection

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "size_bytes": self.size_bytes,
            "last_modified": self.last_modified,
            "symbol_count": self.symbol_count,
            "edge_count": self.edge_count,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileInfo":
        return cls(
            path=d["path"],
            language=d["language"],
            size_bytes=d.get("size_bytes", 0),
            last_modified=d.get("last_modified", 0.0),
            symbol_count=d.get("symbol_count", 0),
            edge_count=d.get("edge_count", 0),
            checksum=d.get("checksum"),
        )

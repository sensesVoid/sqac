"""File watcher for sqac.graph auto-sync.

Polls the project directory for changes and rebuilds the code graph
automatically. Uses a polling approach (no external dependencies) that
works on all platforms.

Usage:
    from sqac.graph.watcher import GraphWatcher

    watcher = GraphWatcher("/path/to/project")
    watcher.start()  # background thread
    # ... work ...
    watcher.stop()
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .graph import CodeGraph
from .parser import detect_language, should_skip

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    """Fast content hash for change detection."""
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()[:16]
    except (OSError, IOError):
        return ""


class GraphWatcher:
    """Watch a project directory for file changes and rebuild the graph.

    Polls every `interval` seconds. When source files are added, modified,
    or deleted, the graph is rebuilt automatically.

    Args:
        project_root: Path to the project root directory.
        interval: Poll interval in seconds (default: 5).
        on_rebuild: Optional callback called after each rebuild with the
            summary dict. Useful for logging or UI updates.
    """

    def __init__(
        self,
        project_root: str | Path,
        interval: float = 5.0,
        on_rebuild: Optional[Callable[[dict], None]] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.interval = interval
        self.on_rebuild = on_rebuild
        self._graph: Optional[CodeGraph] = None
        self._snapshots: dict[str, str] = {}  # path -> content hash
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def graph(self) -> CodeGraph:
        """Lazily initialize and return the code graph."""
        if self._graph is None:
            self._graph = CodeGraph(self.project_root)
            if not self._graph.load():
                self._graph.build()
                self._graph.save()
            self._take_snapshot()
        return self._graph

    def _take_snapshot(self) -> None:
        """Record content hashes of all source files."""
        self._snapshots.clear()
        for rel_path in self._walk_source_files():
            abs_path = self.project_root / rel_path
            self._snapshots[rel_path] = _file_hash(abs_path)

    def _walk_source_files(self):
        """Yield relative paths of source files."""
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in (
                "node_modules", "__pycache__", ".venv", "venv",
                ".sqac", ".sqac-graph", "dist", "build", ".tox",
                ".mypy_cache", ".pytest_cache",
            )]
            for fname in files:
                fpath = Path(root) / fname
                rel = fpath.relative_to(self.project_root)
                if should_skip(rel):
                    continue
                if detect_language(fpath) is not None:
                    yield rel.as_posix()

    def _detect_changes(self) -> tuple[set[str], set[str], set[str]]:
        """Compare current files against snapshot. Returns (added, modified, removed)."""
        current: dict[str, str] = {}
        for rel_path in self._walk_source_files():
            abs_path = self.project_root / rel_path
            current[rel_path] = _file_hash(abs_path)

        added = set(current.keys()) - set(self._snapshots.keys())
        removed = set(self._snapshots.keys()) - set(current.keys())
        modified = set()
        for path in set(current.keys()) & set(self._snapshots.keys()):
            if current[path] != self._snapshots[path]:
                modified.add(path)

        return added, modified, removed

    def _poll(self) -> None:
        """Poll for changes and rebuild if needed."""
        while not self._stop_event.is_set():
            try:
                added, modified, removed = self._detect_changes()
                if added or modified or removed:
                    logger.info(
                        "graph watcher: changes detected — +%d ~%d -%d files",
                        len(added), len(modified), len(removed),
                    )
                    self.graph.build()
                    self.graph.save()
                    self._take_snapshot()
                    if self.on_rebuild:
                        self.on_rebuild(self.graph.stats())
            except Exception as exc:
                logger.warning("graph watcher: rebuild failed: %s", exc)
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        """Start the watcher in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        # Force initial build
        self.graph
        self._thread = threading.Thread(
            target=self._poll, daemon=True, name="graph-watcher"
        )
        self._thread.start()
        logger.info("graph watcher: started (interval=%.1fs)", self.interval)

    def stop(self) -> None:
        """Stop the watcher."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
            self._thread = None
        logger.info("graph watcher: stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

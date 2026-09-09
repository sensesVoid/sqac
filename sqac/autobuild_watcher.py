"""File watcher for sqac.autobuild auto-sync using watchdog.

Uses OS-native file system events (inotify on Linux, FSEvents on macOS,
ReadDirectoryChangesW on Windows) for real-time change detection.
Falls back to polling if watchdog is unavailable.

Usage:
    from sqac.autobuild_watcher import ProjectWatcher

    watcher = ProjectWatcher("/path/to/project")
    watcher.start()  # background thread
    # ... work ...
    watcher.stop()
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .autobuild import (
    EXCLUDED_DIRS, EXCLUDED_NAMES,
    extract_project_units, build_store, save_state, load_state,
    diff_units, unit_map,
)

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent, FileMovedEvent
    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False
    Observer = None
    FileSystemEventHandler = object


def _should_skip(path: Path, root: Path) -> bool:
    """Check if path should be excluded from watching."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    
    # Check excluded directories
    for part in rel.parts:
        if part in EXCLUDED_DIRS:
            return True
    
    # Check excluded names (wildcard support)
    name = path.name
    for pattern in EXCLUDED_NAMES:
        if pattern.startswith("*."):
            ext = pattern[1:]
            if name.endswith(ext):
                return True
        elif name == pattern:
            return True
    
    return False


def _file_hash(path: Path) -> str:
    """Fast content hash for change detection."""
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()[:16]
    except (OSError, IOError):
        return ""


class _ChangeHandler(FileSystemEventHandler):
    """Handle file system events and queue changes for processing."""
    
    def __init__(self, watcher: 'ProjectWatcher'):
        self.watcher = watcher
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._debounce_timer: Optional[threading.Timer] = None
    
    def _queue_change(self, path: Path, event_type: str):
        """Queue a path for processing."""
        if _should_skip(path, self.watcher.project_root):
            return
        with self._lock:
            self._pending.add(path.relative_to(self.watcher.project_root).as_posix())
        self._schedule_flush()
    
    def _schedule_flush(self):
        """Debounce: flush pending changes after 500ms of quiet."""
        if self._debounce_timer:
            self._debounce_timer.cancel()
        self._debounce_timer = threading.Timer(0.5, self._flush)
        self._debounce_timer.daemon = True
        self._debounce_timer.start()
    
    def _flush(self):
        """Process all pending changes."""
        with self._lock:
            if not self._pending:
                return
            pending = self._pending.copy()
            self._pending.clear()
        
        self.watcher._process_changes(pending)
    
    def on_modified(self, event):
        if not event.is_directory:
            self._queue_change(Path(event.src_path), "modified")
    
    def on_created(self, event):
        if not event.is_directory:
            self._queue_change(Path(event.src_path), "created")
    
    def on_deleted(self, event):
        if not event.is_directory:
            self._queue_change(Path(event.src_path), "deleted")
    
    def on_moved(self, event):
        if not event.is_directory:
            self._queue_change(Path(event.dest_path), "moved")


class ProjectWatcher:
    """Watch a project directory for file changes and rebuild the cartridge.
    
    Uses watchdog for native file system events. Falls back to polling
    if watchdog is unavailable.
    
    Args:
        project_root: Path to the project root directory.
        sqac_dir: Output directory for .sqac cartridge (default: project_root/.sqac)
        interval: Poll interval for fallback (default: 5.0s)
        on_sync: Optional callback called after each sync with the summary dict.
        use_watchdog: Whether to use watchdog (default: True if available)
    """
    
    def __init__(
        self,
        project_root: str | Path,
        sqac_dir: str | Path | None = None,
        interval: float = 5.0,
        on_sync: Optional[Callable[[dict], None]] = None,
        use_watchdog: bool = True,
    ):
        self.project_root = Path(project_root).resolve()
        self.sqac_dir = Path(sqac_dir) if sqac_dir else self.project_root / ".sqac"
        self.interval = interval
        self.on_sync = on_sync
        self.use_watchdog = use_watchdog and _HAS_WATCHDOG
        
        self._observer: Optional[Observer] = None
        self._handler: Optional[_ChangeHandler] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_sync = 0
    
    def _initial_build(self) -> dict:
        """Perform initial full build."""
        logger.info("autobuild watcher: initial build...")
        units = extract_project_units(self.project_root)
        if not units:
            logger.warning("autobuild watcher: no units extracted")
            return {"added": [], "changed": [], "removed": []}
        
        store = build_store(units)
        self.sqac_dir.mkdir(parents=True, exist_ok=True)
        store.save(self.sqac_dir / "project.sqac", name="project",
                   description=f"autobuilt from {self.project_root}")
        save_state(self.sqac_dir, unit_map(units))
        
        logger.info(f"autobuild watcher: built {len(units)} units")
        return {"added": [u.source for u in units], "changed": [], "removed": []}
    
    def _process_changes(self, changed_paths: set[str]) -> None:
        """Process a set of changed paths."""
        if time.time() - self._last_sync < 1.0:
            return  # Rate limit
        
        try:
            units = extract_project_units(self.project_root)
            new_map = unit_map(units)
            
            state = load_state(self.sqac_dir) or {}
            prev_map = state.get("sources", {})
            
            added, changed, removed = diff_units(prev_map, new_map)
            
            if added or changed or removed:
                store = build_store(units)
                store.save(self.sqac_dir / "project.sqac", name="project",
                           description=f"autobuilt from {self.project_root}")
                save_state(self.sqac_dir, new_map)
                
                summary = {"added": added, "changed": changed, "removed": removed}
                logger.info("autobuild watcher: synced — +%d ~%d -%d", 
                           len(added), len(changed), len(removed))
                
                if self.on_sync:
                    self.on_sync(summary)
            
            self._last_sync = time.time()
            
        except Exception as exc:
            logger.warning("autobuild watcher: sync failed: %s", exc)
    
    def _poll(self) -> None:
        """Fallback polling method."""
        self._initial_build()
        
        while not self._stop_event.is_set():
            try:
                units = extract_project_units(self.project_root)
                new_map = unit_map(units)
                
                state = load_state(self.sqac_dir) or {}
                prev_map = state.get("sources", {})
                
                added, changed, removed = diff_units(prev_map, new_map)
                
                if added or changed or removed:
                    store = build_store(units)
                    store.save(self.sqac_dir / "project.sqac", name="project",
                               description=f"autobuilt from {self.project_root}")
                    save_state(self.sqac_dir, new_map)
                    
                    summary = {"added": added, "changed": changed, "removed": removed}
                    logger.info("autobuild watcher: polled — +%d ~%d -%d", 
                               len(added), len(changed), len(removed))
                    
                    if self.on_sync:
                        self.on_sync(summary)
                        
            except Exception as exc:
                logger.warning("autobuild watcher: poll failed: %s", exc)
            
            self._stop_event.wait(self.interval)
    
    def start(self) -> None:
        """Start the watcher."""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._initial_build()
        
        if self.use_watchdog and _HAS_WATCHDOG:
            self._start_watchdog()
        else:
            self._start_polling()
        
        logger.info("autobuild watcher: started (watchdog=%s, interval=%.1fs)", 
                   self.use_watchdog, self.interval)
    
    def _start_watchdog(self) -> None:
        """Start watchdog-based watching."""
        self._handler = _ChangeHandler(self)
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.project_root), recursive=True)
        self._observer.start()
        logger.info("autobuild watcher: watchdog observer started")
    
    def _start_polling(self) -> None:
        """Start polling fallback."""
        self._poll_thread = threading.Thread(
            target=self._poll, daemon=True, name="autobuild-poller"
        )
        self._poll_thread.start()
        logger.info("autobuild watcher: polling fallback started")
    
    def stop(self) -> None:
        """Stop the watcher."""
        self._stop_event.set()
        
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=self.interval * 2)
            self._observer = None
            logger.info("autobuild watcher: watchdog stopped")
        
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=self.interval * 2)
            self._poll_thread = None
            logger.info("autobuild watcher: polling stopped")
        
        if self._handler and self._handler._debounce_timer:
            self._handler._debounce_timer.cancel()
    
    @property
    def is_running(self) -> bool:
        if self.use_watchdog:
            return self._observer is not None and self._observer.is_alive()
        return self._poll_thread is not None and self._poll_thread.is_alive()
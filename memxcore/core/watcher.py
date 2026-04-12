"""
ArchiveWatcher — Watch the archive/ directory for MD file changes and auto-trigger RAG reindexing.

Design principles:
- Graceful degradation when watchdog is not installed; does not affect other features
- 1.5-second debounce: waits until user stops rapid successive edits before triggering, avoiding reindexing intermediate states
- File modify/create: only reindex the affected category (incremental, fast)
- File delete: full rebuild (cannot determine which facts were removed; rebuilding is safer)
- Sync update index.json after each reindex
"""

import os
import threading
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .memory_manager import MemoryManager


class ArchiveWatcher:
    """
    Watch archive/*.md files for changes and auto-trigger RAG reindexing.

    Implemented using watchdog's Observer (background thread) + debounce timer.
    When watchdog is not installed, available=False and all methods silently no-op.
    """

    def __init__(self, manager: "MemoryManager") -> None:
        self._manager = manager
        self._observer = None
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._available = False
        self._try_import()

    def _try_import(self) -> None:
        try:
            import watchdog  # noqa: F401
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start file watching. Calling while already running is a no-op."""
        if not self._available or self._observer is not None:
            return

        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(".md"):
                    watcher._schedule(event.src_path)

            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith(".md"):
                    watcher._schedule(event.src_path)

            def on_deleted(self, event):
                if not event.is_directory and event.src_path.endswith(".md"):
                    watcher._schedule("__rebuild__")

        self._observer = Observer()
        self._observer.schedule(
            _Handler(),
            self._manager.archive_dir,
            recursive=False,
        )
        self._observer.daemon = True
        self._observer.start()

    def stop(self) -> None:
        """Stop file watching and cancel all pending debounce timers."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    # ── Internal logic ────────────────────────────────────────────────────────

    def _schedule(self, path: str, delay: float = 1.5) -> None:
        """
        Debounce: execute reindexing after a delay of `delay` seconds.
        Multiple triggers for the same file within the delay window will only execute the last one.
        """
        with self._lock:
            existing = self._timers.pop(path, None)
            if existing:
                existing.cancel()
            timer = threading.Timer(delay, self._run, args=[path])
            self._timers[path] = timer
            timer.start()

    def _run(self, path: str) -> None:
        """Actually execute reindexing (runs in the timer thread)."""
        with self._lock:
            self._timers.pop(path, None)

        if path == "__rebuild__":
            self._manager.rebuild_rag_index()
        else:
            category = os.path.basename(path)[:-3]  # strip .md
            if os.path.isfile(path):
                self._manager.rag.reindex_file(path, category)
            else:
                # File was deleted — remove stale vectors for this category
                try:
                    rag = self._manager.rag
                    if rag.available:
                        with rag._lock:
                            existing = rag._collection.get(where={"category": category})
                            if existing["ids"]:
                                rag._collection.delete(ids=existing["ids"])
                except Exception:
                    pass

        # Keep BM25 and index.json in sync after each change
        self._manager.bm25.rebuild(self._manager.archive_dir, self._manager.user_path)
        self._manager.update_index()

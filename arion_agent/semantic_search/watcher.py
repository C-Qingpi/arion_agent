from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from arion_agent.semantic_search.config import TEXT_EXTENSIONS, WATCHER_DEBOUNCE_SEC
from arion_agent.semantic_search.ignore import load_ignore_patterns, should_ignore


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(
        self,
        workspace: Path,
        on_change: Callable[[set[str]], None],
        debounce_sec: float,
    ) -> None:
        super().__init__()
        self._workspace = workspace.resolve()
        self._on_change = on_change
        self._debounce_sec = debounce_sec
        self._patterns = load_ignore_patterns(self._workspace)
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _rel_from_path(self, path: str) -> str | None:
        raw = Path(path)
        if not raw.is_absolute():
            raw = self._workspace / raw
        try:
            rel = raw.resolve().relative_to(self._workspace).as_posix()
        except ValueError:
            return None
        if should_ignore(rel, self._patterns):
            return None
        return rel

    def _rel(self, path: str) -> str | None:
        raw = Path(path)
        if not raw.is_absolute():
            raw = self._workspace / raw
        if not raw.exists():
            return self._rel_from_path(path)
        try:
            rel = raw.resolve().relative_to(self._workspace).as_posix()
        except ValueError:
            return None
        if should_ignore(rel, self._patterns):
            return None
        if raw.is_file() and raw.suffix.lower() not in TEXT_EXTENSIONS:
            return None
        return rel

    def _schedule(self, rel: str | None) -> None:
        if rel is None:
            return
        with self._lock:
            self._pending.add(rel)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_sec, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            batch = set(self._pending)
            self._pending.clear()
            self._timer = None
        if batch:
            self._on_change(batch)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule(self._rel(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule(self._rel(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        rel = self._rel(event.src_path)
        if rel is not None:
            self._schedule(rel)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_rel = self._rel(event.src_path)
        dest_rel = self._rel(event.dest_path)
        with self._lock:
            if src_rel is not None:
                self._pending.add(f"__deleted__:{src_rel}")
            if dest_rel is not None:
                self._pending.add(dest_rel)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_sec, self._flush)
            self._timer.daemon = True
            self._timer.start()


class WorkspaceWatcher:
    def __init__(
        self,
        workspace: Path,
        on_change: Callable[[set[str]], None],
        *,
        debounce_sec: float = WATCHER_DEBOUNCE_SEC,
    ) -> None:
        self._workspace = workspace.resolve()
        self._handler = _DebouncedHandler(workspace, on_change, debounce_sec)
        self._observer = Observer()

    def start(self) -> None:
        self._observer.schedule(self._handler, str(self._workspace), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=5)

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from arion_agent.semantic_search.config import INCREMENTAL_BATCH_FILES
from arion_agent.semantic_search.embedder import embedder_loaded
from arion_agent.semantic_search.indexer import (
    apply_removals,
    apply_renames,
    file_content_hash,
    index_file_batch,
    plan_sync,
    scan_manifest,
)
from arion_agent.semantic_search.ignore import load_ignore_patterns
from arion_agent.semantic_search.store import ChunkStore


@dataclass(frozen=True, slots=True)
class IndexerStatus:
    running: bool
    thread_alive: bool
    initial_sync_done: bool
    total_files: int
    indexed_files: int
    pending_files: int
    chunk_count: int
    embedding: bool
    embedder_ready: bool
    last_batch_sec: float | None
    last_error: str | None


class IncrementalIndexer:
    """Background indexer: batches files so search is usable while indexing."""

    def __init__(
        self,
        workspace: Path,
        store: ChunkStore,
        *,
        batch_size: int = INCREMENTAL_BATCH_FILES,
        extra_ignore: list[str] | None = None,
    ) -> None:
        self._workspace = workspace.resolve()
        self._store = store
        self._batch_size = batch_size
        self._patterns = load_ignore_patterns(
            self._workspace,
            extra_patterns=extra_ignore,
        )
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._initial_sync_done = False
        self._embedding = False
        self._last_batch_sec: float | None = None
        self._last_error: str | None = None
        self._vector_cache: dict[str, list[float]] = {}

    @property
    def status(self) -> IndexerStatus:
        with self._lock:
            pending = len(self._pending) + self._queue.qsize()
        thread_alive = self._thread is not None and self._thread.is_alive()
        manifest = self._store.load_manifest()
        disk_manifest = scan_manifest(self._workspace, self._patterns)
        return IndexerStatus(
            running=thread_alive,
            thread_alive=thread_alive,
            initial_sync_done=self._initial_sync_done,
            total_files=len(disk_manifest),
            indexed_files=len(manifest),
            pending_files=pending,
            chunk_count=self._store.chunk_count(),
            embedding=self._embedding,
            embedder_ready=embedder_loaded(),
            last_batch_sec=self._last_batch_sec,
            last_error=self._last_error,
        )

    def ensure_running(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._clear_queue()
        with self._lock:
            self._pending.clear()
        self._running = True
        self._last_error = None
        self._vector_cache = {
            row.get("search_text_hash") or row["content_hash"]: row["vector"]
            for row in self._store.all_rows()
        }
        self._thread = threading.Thread(target=self._run, name="semantic-indexer", daemon=True)
        self._thread.start()

    def start(self) -> None:
        self.ensure_running()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=120)

    def request_paths(self, rel_paths: set[str]) -> None:
        with self._lock:
            for rel in rel_paths:
                if rel not in self._pending:
                    self._pending.add(rel)
                    self._queue.put(rel)

    def handle_watcher_batch(self, rel_paths: set[str]) -> None:
        deletions = {
            rel[len("__deleted__:"):]
            for rel in rel_paths
            if rel.startswith("__deleted__:")
        }
        updates = {rel for rel in rel_paths if not rel.startswith("__deleted__:")}

        manifest = self._store.load_manifest()
        for old_path in list(deletions):
            for new_path in list(updates):
                candidate = self._workspace / new_path
                if not candidate.is_file():
                    continue
                old_hash = manifest.get(old_path)
                if old_hash is None:
                    continue
                if file_content_hash(candidate) != old_hash:
                    continue
                apply_renames(self._store, [(old_path, new_path)])
                deletions.discard(old_path)
                updates.discard(new_path)
                break

        if deletions:
            apply_removals(self._store, sorted(deletions))

        if updates:
            self.request_paths(updates)

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        try:
            if not self._initial_sync_done:
                self._bootstrap_sync()
            while self._running:
                batch = self._drain_batch()
                if batch:
                    self._index_batch(batch)
                    continue
                time.sleep(0.2)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._running = False
            self._embedding = False

    def _bootstrap_sync(self) -> None:
        manifest = self._store.load_manifest()
        sync = plan_sync(self._workspace, manifest, self._patterns)

        if sync.renames:
            apply_renames(self._store, sync.renames)
        if sync.removals:
            apply_removals(self._store, sync.removals)

        for path in sync.to_index:
            self.request_paths({path.relative_to(self._workspace).as_posix()})

        while self._running:
            with self._lock:
                pending = len(self._pending)
            if pending == 0:
                break
            batch = self._drain_batch()
            if batch:
                self._index_batch(batch)
            else:
                break

        self._initial_sync_done = True

    def _drain_batch(self) -> list[Path]:
        rels: list[str] = []
        while len(rels) < self._batch_size:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                if self._running:
                    self._queue.put(None)
                break
            with self._lock:
                self._pending.discard(item)
            rels.append(item)

        return [self._workspace / rel for rel in rels if (self._workspace / rel).is_file()]

    def _index_batch(self, paths: list[Path]) -> None:
        self._embedding = True
        started = time.perf_counter()
        index_file_batch(
            paths,
            self._workspace,
            self._store,
            self._vector_cache,
            persist=True,
        )
        self._last_batch_sec = time.perf_counter() - started
        self._embedding = False

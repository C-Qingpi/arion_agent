from __future__ import annotations

import json
import queue
import shutil
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
from arion_agent.semantic_search.scope import (
    ignore_config_mtime,
    is_ignore_config_rel,
    is_search_config_rel,
    resolve_index_scope,
    search_config_mtime,
)
from arion_agent.semantic_search.store import ChunkStore

BACKUP_SUFFIX = ".backup"


def _compute_and_save_stale_info(
    workspace: Path,
    backup_path: Path,
    index_scope,
) -> None:
    """Compute the diff between the backup manifest and the current workspace,
    and save it as stale_info.json alongside the backup so search can warn the agent."""
    try:
        backup_store = ChunkStore(backup_path)
        old_manifest = backup_store.load_manifest()
        disk_manifest = scan_manifest(workspace, index_scope)

        sync = plan_sync(workspace, old_manifest, index_scope)

        stale_info = {
            "stale_count": len(sync.to_index),
            "missing_count": len(sync.removals),
            "stale_files": sorted(
                str(p.relative_to(workspace)) for p in sync.to_index
            ),
            "missing_files": sorted(sync.removals),
            "total_old": len(old_manifest),
            "total_new": len(disk_manifest),
        }
        stale_path = backup_path / "stale_info.json"
        stale_path.write_text(json.dumps(stale_info, indent=2), encoding="utf-8")
    except Exception:
        pass  # non-critical — best-effort stale info


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
        self._extra_ignore = extra_ignore
        self._index_scope = resolve_index_scope(self._workspace, extra_ignore=extra_ignore)
        self._scope_mtime = search_config_mtime(self._workspace)
        self._ignore_mtime = ignore_config_mtime(self._workspace)
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
        self._watcher = None

    @property
    def status(self) -> IndexerStatus:
        if self._maybe_reload_scope():
            self._request_resync()
        with self._lock:
            pending = len(self._pending) + self._queue.qsize()
        thread_alive = self._thread is not None and self._thread.is_alive()
        manifest = self._store.load_manifest()
        disk_manifest = scan_manifest(self._workspace, self._index_scope)
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
        # Always resync on thread start — scan_manifest and _vector_cache
        # happen inside _run() so the caller is not blocked.
        self._initial_sync_done = False
        self._running = True
        self._last_error = None
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
        if any(is_search_config_rel(rel) for rel in rel_paths):
            self._maybe_reload_scope()
            self._request_resync()
            return

        if any(is_ignore_config_rel(rel) for rel in rel_paths):
            self._maybe_reload_scope()
            self._request_resync()
            if self._watcher is not None:
                self._watcher.reload_patterns()
            return

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

    def set_watcher(self, watcher) -> None:
        """Register the WorkspaceWatcher so its ignore patterns can be refreshed on config change."""
        self._watcher = watcher

    def reset_and_resync(self) -> None:
        # Atomic swap: back up the old index so search is uninterrupted
        backup_path = self._store.index_dir.parent / (self._store.index_dir.name + BACKUP_SUFFIX)
        if self._store.index_dir.exists():
            if backup_path.exists():
                shutil.rmtree(backup_path)
            self._store.index_dir.rename(backup_path)
        self._store = ChunkStore(self._index_dir)

        # Compute what changed vs the old manifest (best-effort)
        _compute_and_save_stale_info(self._workspace, backup_path, self._index_scope)

        self._index_scope = resolve_index_scope(
            self._workspace,
            extra_ignore=self._extra_ignore,
        )
        self._scope_mtime = search_config_mtime(self._workspace)
        self._ignore_mtime = ignore_config_mtime(self._workspace)
        self._initial_sync_done = False
        self._vector_cache.clear()
        self._last_error = None
        self._clear_queue()
        with self._lock:
            self._pending.clear()
        self.ensure_running()

    def _maybe_reload_scope(self) -> bool:
        changed = False

        # Check search.json
        mtime = search_config_mtime(self._workspace)
        if mtime != self._scope_mtime:
            self._scope_mtime = mtime
            changed = True

        # Check .searchignore and .gitignore
        ignore_mtime = ignore_config_mtime(self._workspace)
        if ignore_mtime != self._ignore_mtime:
            self._ignore_mtime = ignore_mtime
            changed = True

        if changed:
            self._index_scope = resolve_index_scope(
                self._workspace,
                extra_ignore=self._extra_ignore,
            )
        return changed

    def _request_resync(self) -> None:
        self._initial_sync_done = False
        self._clear_queue()
        with self._lock:
            self._pending.clear()

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        try:
            # Load existing vectors so unchanged files skip re-embedding.
            # Moved here from ensure_running() to avoid blocking the caller.
            self._vector_cache = {
                row.get("search_text_hash") or row["content_hash"]: row["vector"]
                for row in self._store.all_rows()
            }
            # Ensure index exists even if resuming after a crash/restart
            # where bootstrap already completed in a previous session
            self._store._ensure_index()
            while self._running:
                if self._maybe_reload_scope():
                    self._request_resync()
                if not self._initial_sync_done:
                    self._bootstrap_sync()
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
        sync = plan_sync(self._workspace, manifest, self._index_scope)

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
                continue
            with self._lock:
                still_pending = len(self._pending) + self._queue.qsize()
            if still_pending == 0:
                break
            time.sleep(0.1)

        self._initial_sync_done = True
        self._store._ensure_index()

        # Initial sync done — delete the backup (new index is live)
        backup_path = self._store.index_dir.parent / (self._store.index_dir.name + BACKUP_SUFFIX)
        if backup_path.exists():
            try:
                shutil.rmtree(backup_path)
            except Exception:
                pass  # non-critical cleanup

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

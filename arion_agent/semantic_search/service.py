from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

from arion_agent.semantic_search.config import EMBEDDER_WARMUP, FINAL_TOP_K, MIN_HYBRID_SCORE, resolve_index_dir
from arion_agent.semantic_search.embedder import get_embedder
from arion_agent.semantic_search.translate import warmup_mt


def _warmup_models() -> None:
    get_embedder()
    warmup_mt()
from arion_agent.semantic_search.incremental import BACKUP_SUFFIX, IncrementalIndexer, IndexerStatus
from arion_agent.semantic_search.scope import resolve_index_scope
from arion_agent.semantic_search.retriever import SearchResult, count_paths_with_filters, hybrid_search
from arion_agent.semantic_search.store import ChunkStore
from arion_agent.semantic_search.watcher import WorkspaceWatcher


@dataclass(frozen=True, slots=True)
class SearchServiceConfig:
    batch_size: int | None = None
    extra_ignore: list[str] | None = None
    warmup_embedder: bool = EMBEDDER_WARMUP
    enable_watcher: bool = True


class SearchService:
    """Agent-facing semantic search: background incremental index + workspace watcher."""

    def __init__(
        self,
        workspace: Path,
        index_dir: Path | None = None,
        *,
        config: SearchServiceConfig | None = None,
    ) -> None:
        self._workspace = workspace.resolve()
        self._index_dir = resolve_index_dir(self._workspace, index_dir)
        self._config = config or SearchServiceConfig()
        self._store = ChunkStore(self._index_dir)
        self._indexer = IncrementalIndexer(
            self._workspace,
            self._store,
            batch_size=self._config.batch_size or 8,
            extra_ignore=self._config.extra_ignore,
        )
        self._watcher: WorkspaceWatcher | None = None
        self._warmup_thread: threading.Thread | None = None
        self._started = False

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def index_dir(self) -> Path:
        return self._index_dir

    @property
    def store(self) -> ChunkStore:
        return self._store

    def start(self) -> None:
        from arion_agent.semantic_search.scope import ensure_index_config_files

        ensure_index_config_files(self._workspace)

        if self._config.warmup_embedder and self._warmup_thread is None:
            self._warmup_thread = threading.Thread(
                target=_warmup_models,
                name="search-model-warmup",
                daemon=True,
            )
            self._warmup_thread.start()

        self._indexer.ensure_running()

        if self._config.enable_watcher and self._watcher is None:
            self._watcher = WorkspaceWatcher(
                self._workspace,
                self._indexer.handle_watcher_batch,
            )
            self._indexer.set_watcher(self._watcher)
            self._watcher.start()

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        self._indexer.stop()
        self._started = False

    def status(self) -> IndexerStatus:
        return self._indexer.status

    def search(
        self,
        query: str,
        *,
        target_directories: list[str] | None = None,
        path_glob: str | None = None,
        num_results: int = FINAL_TOP_K,
        min_score: float = MIN_HYBRID_SCORE,
    ) -> list[SearchResult]:
        # Ensure vector index exists before first query of each session.
        # _ensure_index is a no-op after the first call, so subsequent
        # queries incur only a cheap boolean check.
        self._store._ensure_index()

        # If the primary store is empty (rebuild in progress), fall back to
        # the backup index so search remains uninterrupted during a reset.
        store = self._store
        if not store.has_table():
            backup_store = self._try_backup_store()
            if backup_store is not None:
                store = backup_store

        return hybrid_search(
            query,
            index_dir=store.index_dir if store is not self._store else self._index_dir,
            store=store,
            target_directories=target_directories,
            path_glob=path_glob,
            num_results=num_results,
            min_score=min_score,
        )

    def _try_backup_store(self) -> ChunkStore | None:
        """Return a ChunkStore for the backup index if it exists and has data."""
        backup_path = self._index_dir.parent / (self._index_dir.name + BACKUP_SUFFIX)
        if not backup_path.exists():
            return None
        try:
            backup_store = ChunkStore(backup_path)
            if backup_store.has_table():
                backup_store._ensure_index()
                return backup_store
        except Exception:
            pass
        return None

    @property
    def stale_info(self) -> dict | None:
        """Read stale_info.json from the backup index dir, if a rebuild is in progress.

        Returns the diff between the old and current workspace at the time the
        reset was triggered, or None if no backup (no active rebuild).
        """
        backup_path = self._index_dir.parent / (self._index_dir.name + BACKUP_SUFFIX)
        stale_path = backup_path / "stale_info.json"
        if stale_path.exists():
            try:
                return json.loads(stale_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def reset_index(self) -> None:
        self.start()
        self._indexer.reset_and_resync()

    def ignore_patterns(self) -> list[str]:
        return resolve_index_scope(
            self._workspace,
            extra_ignore=self._config.extra_ignore,
        ).patterns

    def count_filter_match_paths(
        self,
        *,
        target_directories: list[str] | None = None,
        path_glob: str | None = None,
    ) -> tuple[int, int]:
        """Return (matching_paths, total_paths) for the given filter combination.

        Useful for no-results diagnostics — tells the agent whether zero
        results came from the query or from the filters.
        """
        return count_paths_with_filters(
            self._store,
            target_directories=target_directories,
            path_glob=path_glob,
        )

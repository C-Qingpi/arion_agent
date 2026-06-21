from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from arion_agent.semantic_search.config import EMBEDDER_WARMUP, FINAL_TOP_K, MIN_HYBRID_SCORE, resolve_index_dir
from arion_agent.semantic_search.embedder import get_embedder
from arion_agent.semantic_search.translate import warmup_mt


def _warmup_models() -> None:
    get_embedder()
    warmup_mt()
from arion_agent.semantic_search.incremental import IncrementalIndexer, IndexerStatus
from arion_agent.semantic_search.scope import resolve_index_scope
from arion_agent.semantic_search.retriever import SearchResult, hybrid_search
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
        return hybrid_search(
            query,
            index_dir=self._index_dir,
            store=self._store,
            target_directories=target_directories,
            path_glob=path_glob,
            num_results=num_results,
            min_score=min_score,
        )

    def reset_index(self) -> None:
        self.start()
        self._indexer.reset_and_resync()

    def ignore_patterns(self) -> list[str]:
        return resolve_index_scope(
            self._workspace,
            extra_ignore=self._config.extra_ignore,
        ).patterns

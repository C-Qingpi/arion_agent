from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

from langchain_core.tools import tool

from arion_agent.semantic_search.embedder import embedder_loaded

if TYPE_CHECKING:
    from arion_agent.semantic_search.incremental import IndexerStatus
    from arion_agent.semantic_search.service import SearchService


def _index_status_line(st: IndexerStatus) -> str:
    """Build a one-line index status indicator."""
    if st.last_error:
        return f"⚠️ Indexer error: {st.last_error}"
    if not st.thread_alive and st.indexed_files < st.total_files:
        return f"⚠️ Indexer not running — searched {st.indexed_files}/{st.total_files} files"
    if st.indexed_files == 0 and st.chunk_count == 0:
        if not st.embedder_ready:
            return f"⏳ Index startup: loading embedding models ({st.pending_files} files queued)"
        if st.embedding:
            return f"⏳ Index startup: embedding first batch ({st.pending_files} files queued)"
        if st.total_files == 0:
            return "ℹ️ Workspace has no files to index — search will return nothing"
        return f"⏳ Index starting ({st.pending_files}/{st.total_files} files queued)"
    if not st.initial_sync_done:
        busy = " (indexing)" if st.embedding else ""
        return (
            f"⏳ Index building: searched {st.indexed_files}/{st.total_files} files, "
            f"{st.chunk_count} chunks{busy}"
        )
    if st.indexed_files < st.total_files:
        return (
            f"ℹ️ Searched {st.indexed_files}/{st.total_files} files "
            f"(index catching up after changes)"
        )
    return f"ℹ️ Searched {st.indexed_files} files — index fully built"


def format_empty_search_message(st: IndexerStatus) -> str:
    """Explain why semantic_search returned no hits (startup vs partial vs truly empty)."""
    status = _index_status_line(st)
    if st.last_error:
        return f"{status}\nTry fixing the error and retrying."

    if not st.thread_alive and st.indexed_files < st.total_files:
        return (
            f"{status}\n"
            "Send another message to restart indexing."
        )

    if st.indexed_files == 0 and st.chunk_count == 0:
        if st.embedding or (st.running and st.pending_files > 0 and not st.embedder_ready):
            return (
                f"{status}\n"
                "First run on a new workspace can take several minutes before the first file "
                "is searchable; retry shortly with a broader query."
            )
        if st.running and st.pending_files > 0:
            return f"{status}\nRetry shortly."
        if st.running and st.pending_files == 0 and st.total_files > 0:
            return (
                f"{status}\n"
                "Indexer is active but no files are queued yet — scanner may still be discovering "
                "files. Try again shortly."
            )
        return f"No results.\n{status}"

    if not st.initial_sync_done:
        return (
            f"{status}\n"
            "Try a broader query, or wait for more files to finish indexing and retry."
        )

    return f"No results.\n{status}"


def _format_hits(results, st: IndexerStatus | None) -> str:
    status = _index_status_line(st) if st is not None else ""
    lines = [status, f"({len(results)} hits)"] if status else [f"({len(results)} hits)"]
    for i, hit in enumerate(results, start=1):
        lines.append(
            f"--- [{i}] score={hit.score:.3f} ---\n"
            f"{hit.path}:{hit.start_line}-{hit.end_line} [{hit.kind}]\n"
            f"{hit.snippet}"
        )
    return "\n".join(lines)


def _format_status_detail(st: IndexerStatus) -> str:
    """Build a detailed multi-line indexer status report."""
    lines = [
        f"Indexer running: {st.running}",
        f"Background thread: {'alive' if st.thread_alive else 'stopped'}",
        f"Initial sync done: {st.initial_sync_done}",
        f"Files: {st.indexed_files} indexed / {st.total_files} total ({st.pending_files} pending)",
        f"Chunks: {st.chunk_count}",
        f"Embedder: {'ready' if st.embedder_ready else 'loading...'}",
    ]
    if st.embedding:
        lines.append("Status: embedding in progress")
    if st.last_batch_sec is not None:
        lines.append(f"Last batch time: {st.last_batch_sec:.1f}s")
    if st.last_error:
        lines.append(f"Last error: {st.last_error}")
    return "\n".join(lines)


def create_search_tools(service: SearchService, *, min_score: float, default_num_results: int) -> list:
    @tool
    def semantic_search(
        query: Annotated[str, "Natural-language search query"],
        target_directories: Annotated[
            list[str],
            "Optional path prefixes to limit search (workspace-relative, e.g. ['src', 'docs'])",
        ] = None,
        path_glob: Annotated[
            str,
            "Optional glob filter on result paths (e.g. '**/*.py', '**/*.md', 'src/**')",
        ] = None,
        num_results: Annotated[int, "Max results to return (1-25)"] = default_num_results,
    ) -> str:
        """Semantic search over workspace files by meaning, not exact text.

        Prefer narrow target_directories and path_glob before a workspace-wide query.
        Returns path, line range, and snippet for each hit, plus an index status
        line indicating how much of the workspace was searched. Indexing runs in the
        background; results improve as more files are indexed.
        """
        capped = max(1, min(num_results, 25))
        service.start()

        # Don't block on model download — return status immediately
        if not embedder_loaded():
            st = service.status()
            return (
                f"⏳ Embedding model still loading...\n{_index_status_line(st)}\n"
                "Retry in 1-2 minutes when model warmup completes."
            )

        def _run_search():
            return service.search(
                query,
                target_directories=target_directories or None,
                path_glob=path_glob,
                num_results=capped,
                min_score=min_score,
            )

        results = _run_search()
        st = service.status()
        if not results and not st.thread_alive and st.indexed_files < st.total_files:
            service.start()
            if embedder_loaded():
                results = _run_search()
                st = service.status()

        if not results:
            return format_empty_search_message(st)

        return _format_hits(results, st)

    @tool
    def indexer_status() -> str:
        """Return the current state of the workspace semantic search indexer.

        Shows whether the indexer is running, how many files have been indexed
        vs discovered, chunk counts, embedder state, and any errors. Use this to
        understand why search results may be incomplete, whether the indexer has
        stalled, or if a reset is needed.
        """
        service.start()
        st = service.status()
        return _format_status_detail(st)

    @tool
    def reset_search_index() -> str:
        """Reset and rebuild the semantic search index from scratch.

        Clears all indexed data and starts a fresh scan of the workspace.
        Useful when: the index contains stale or incorrect data, search scope
        has changed, or the index is in an unrecoverable error state.
        The rebuild runs in the background; indexer_status reports progress.
        """
        service.reset_index()
        st = service.status()
        return f"Index cleared and rebuild started.\n{_format_status_detail(st)}"

    return [semantic_search, indexer_status, reset_search_index]

from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from arion_agent.semantic_search.incremental import IndexerStatus
    from arion_agent.semantic_search.service import SearchService


def format_empty_search_message(st: IndexerStatus) -> str:
    """Explain why semantic_search returned no hits (startup vs partial vs truly empty)."""
    if st.last_error:
        return (
            f"No results — indexer error: {st.last_error}. "
            f"Indexed {st.indexed_files}/{st.total_files} files ({st.chunk_count} chunks)."
        )

    if not st.thread_alive and st.indexed_files < st.total_files:
        return (
            "No results — background indexer is not running. "
            f"Indexed {st.indexed_files}/{st.total_files} files ({st.chunk_count} chunks). "
            "Send another message to restart indexing."
        )

    if st.indexed_files == 0 and st.chunk_count == 0:
        if st.embedding or (st.running and st.pending_files > 0 and not st.embedder_ready):
            detail = "loading embedding models" if not st.embedder_ready else "processing first batch"
            return (
                "No results yet — index startup in progress "
                f"({detail}; {st.pending_files}/{st.total_files} files queued). "
                "First run on a new workspace can take several minutes before the first file "
                "is searchable; retry shortly."
            )
        if st.running and st.pending_files > 0:
            return (
                "No results yet — index is starting "
                f"({st.pending_files}/{st.total_files} files queued)."
            )
        return f"No results. Indexed {st.indexed_files}/{st.total_files} files."

    if not st.initial_sync_done:
        extra = " (indexing batch in progress)" if st.embedding else ""
        return (
            f"No results in the indexed portion "
            f"({st.indexed_files}/{st.total_files} files, {st.chunk_count} chunks{extra}). "
            "Try a broader query or wait for more files to finish indexing."
        )

    return "No results."


def _format_hits(results, st: IndexerStatus | None) -> str:
    lines = [f"({len(results)} hits)"]
    if st is not None and not st.initial_sync_done and st.indexed_files < st.total_files:
        lines.append(
            f"\n[index partial: {st.indexed_files}/{st.total_files} files indexed; "
            "more results may appear as indexing continues]"
        )
    for i, hit in enumerate(results, start=1):
        lines.append(
            f"\n--- [{i}] score={hit.score:.3f} ---\n"
            f"{hit.path}:{hit.start_line}-{hit.end_line} [{hit.kind}]\n"
            f"{hit.snippet}"
        )
    return "".join(lines)


def create_search_tools(service: SearchService, *, min_score: float, default_num_results: int) -> list:
    @tool
    def semantic_search(
        query: Annotated[str, "Natural-language search query"],
        target_directories: Annotated[
            list[str] | None,
            "Optional path prefixes to limit search (workspace-relative)",
        ] = None,
        num_results: Annotated[int, "Max results to return (1-25)"] = default_num_results,
    ) -> str:
        """Semantic search over workspace files by meaning, not exact text.

        Returns path, line range, and snippet for each hit. Indexing runs in the
        background; results improve as more files are indexed.
        """
        capped = max(1, min(num_results, 25))
        service.start()

        def _run_search():
            return service.search(
                query,
                target_directories=target_directories or None,
                num_results=capped,
                min_score=min_score,
            )

        results = _run_search()
        st = service.status()
        if not results and not st.thread_alive and st.indexed_files < st.total_files:
            service.start()
            results = _run_search()
            st = service.status()

        if not results:
            return format_empty_search_message(st)

        return _format_hits(results, st)

    return [semantic_search]

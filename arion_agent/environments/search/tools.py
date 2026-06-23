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


def format_empty_search_message(
    st: IndexerStatus,
    *,
    filter_matched: int | None = None,
    filter_total: int | None = None,
    path_glob: str | None = None,
    target_directories: list[str] | None = None,
) -> str:
    """Explain why semantic_search returned no hits (startup vs partial vs truly empty).

    When filter_matched and filter_total are provided, they tell the agent how many
    indexed files match the current path_glob/target_directories filters — zero
    filter matches usually means the agent used a wrong filter syntax (e.g. bare
    filename instead of **/filename).
    """
    status = _index_status_line(st)
    if st.last_error:
        return f"{status}\nTry fixing the error and retrying."

    # ── Filter diagnostic (zero matches due to filters, not query) ──
    if (
        filter_matched is not None
        and filter_total is not None
        and filter_total > 0
        and filter_matched == 0
        and (path_glob or target_directories)
    ):
        hint_parts: list[str] = []
        # Check for bare filename mistake
        if path_glob and "*" not in path_glob and "/" not in path_glob:
            hint_parts.append(
                f'path_glob "{path_glob}" is a bare filename — use "**/{path_glob}" '
                "to match files in subdirectories"
            )
        elif path_glob and "/" in path_glob and not path_glob.startswith("**"):
            hint_parts.append(
                f'path_glob lacks **/ prefix — try "**/{path_glob}" to match anywhere'
            )
        elif target_directories and not path_glob:
            hint_parts.append(
                f"no indexed files under {target_directories} — check directory path"
            )
        else:
            hint_parts.append("no indexed files match these filters")
        hint = "\n   Hint: " + "; ".join(hint_parts)

        return (
            f"No results with current filters.\n"
            f"   {filter_matched} of {filter_total} indexed files matched "
            f"{_describe_filters(target_directories, path_glob)}{hint}\n"
            f"   {status}"
        )

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

    # ── Filter matched files exist but query got no semantic hits ──
    if (
        filter_matched is not None
        and filter_matched > 0
        and (path_glob or target_directories)
    ):
        return (
            f"No results.\n"
            f"   {filter_matched} files matched filters but no semantic hits — "
            f"try a broader query or adjust filters\n"
            f"   {status}"
        )

    return f"No results.\n{status}"


def _describe_filters(
    target_directories: list[str] | None,
    path_glob: str | None,
) -> str:
    parts: list[str] = []
    if path_glob:
        parts.append(f'path_glob="{path_glob}"')
    if target_directories:
        parts.append(f"target_directories={target_directories}")
    return f"({', '.join(parts)})" if parts else "(no filters)"


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


def _format_stale_warning(info: dict | None) -> str | None:
    """Build a preamble about stale index if the backup is in use."""
    if not info:
        return None
    stale = info.get("stale_count", 0)
    missing = info.get("missing_count", 0)
    total_old = info.get("total_old", 0)

    parts = [f"⚠️ Using stale search index ({total_old} files) — being rebuilt."]

    if stale > 0:
        if stale <= 10:
            files = info.get("stale_files", [])
            parts.append(f"   {stale} file(s) have changed:")
            for f in files:
                parts.append(f"     • {f}")
        else:
            parts.append(f"   {stale} files have changed content.")

    if missing > 0:
        if missing <= 10:
            files = info.get("missing_files", [])
            parts.append(f"   {missing} file(s) deleted from workspace:")
            for f in files:
                parts.append(f"     • {f}")
        else:
            parts.append(f"   {missing} files deleted from workspace.")

    return "\n".join(parts)


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


def _backup_status(service) -> str:
    """Return a line about the backup index if one exists."""
    backup_path = service.index_dir.parent / (service.index_dir.name + ".backup")
    if backup_path.exists():
        stale = service.stale_info
        if stale:
            count = stale.get("total_old", 0)
            return f"ℹ️ Fallback index active ({count} files) — new index building in background"
        return "ℹ️ Fallback index active — new index building in background"
    return ""


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
        """Semantic search over workspace files by meaning, not exact keywords.

        Use to locate code, documentation, or configuration related to a concept.
        Begin with a 3-8 word query using project-specific terms, then narrow
        with path_glob or target_directories.

        path_glob requires **/ prefix to match subdirectories.
        "**/*.py" matches all Python files; "run.py" only matches at root.
        For CODE: include function/class names ("build_report function").
        For DOCS: describe the concept ("architecture design pipeline").
        Scores above 0.7 are relevant; above 1.0 are precise matches.
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

        # Check if results came from a stale backup index
        stale_info = service.stale_info
        stale_note = _format_stale_warning(stale_info)

        if not results:
            filter_matched, filter_total = service.count_filter_match_paths(
                target_directories=target_directories or None,
                path_glob=path_glob,
            )
            msg = format_empty_search_message(
                st,
                filter_matched=filter_matched,
                filter_total=filter_total,
                path_glob=path_glob,
                target_directories=target_directories or None,
            )
            if stale_note:
                msg = stale_note + "\n" + msg
            return msg

        msg = _format_hits(results, st)
        if stale_note:
            msg = stale_note + "\n" + msg
        return msg

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
        backup_line = _backup_status(service)
        msg = _format_status_detail(st)
        if backup_line:
            msg = backup_line + "\n" + msg
        return msg

    @tool
    def reset_search_index() -> str:
        """Reset and rebuild the semantic search index from scratch.

        The old index is preserved as a fallback until the rebuild completes,
        so searches remain uninterrupted. Once the new index is ready, the
        fallback is automatically removed.
        Useful when: the index contains stale or incorrect data, search scope
        has changed, or the index is in an unrecoverable error state.
        The rebuild runs in the background; indexer_status reports progress.
        """
        service.reset_index()
        st = service.status()
        backup_line = _backup_status(service)
        msg = f"Index rebuild started.\n{_format_status_detail(st)}"
        if backup_line:
            msg = backup_line + "\n" + msg
        return msg

    return [semantic_search, indexer_status, reset_search_index]

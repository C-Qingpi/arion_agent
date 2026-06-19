from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from arion_agent.semantic_search.service import SearchService


def create_search_tools(service: SearchService, *, min_score: float, default_num_results: int) -> list:
    @tool
    def semantic_search(
        query: Annotated[str, "Natural-language search query"],
        target_directories: Annotated[
            list[str],
            "Optional path prefixes to limit search (workspace-relative)",
        ] = None,
        num_results: Annotated[int, "Max results to return (1-25)"] = default_num_results,
    ) -> str:
        """Semantic search over workspace files by meaning, not exact text.

        Returns path, line range, and snippet for each hit. Indexing runs in the
        background; results improve as more files are indexed.
        """
        capped = max(1, min(num_results, 25))
        results = service.search(
            query,
            target_directories=target_directories or None,
            num_results=capped,
            min_score=min_score,
        )
        if not results:
            st = service.status()
            if st.indexed_files == 0:
                return (
                    "No results (index still building). "
                    f"Indexed {st.indexed_files}/{st.total_files} files so far."
                )
            return "No results."

        lines = [f"({len(results)} hits)"]
        for i, hit in enumerate(results, start=1):
            lines.append(
                f"\n--- [{i}] score={hit.score:.3f} ---\n"
                f"{hit.path}:{hit.start_line}-{hit.end_line} [{hit.kind}]\n"
                f"{hit.snippet}"
            )
        return "".join(lines)

    return [semantic_search]

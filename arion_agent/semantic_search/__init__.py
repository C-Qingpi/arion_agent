"""Local generalized semantic search for agent workspaces."""

from arion_agent.semantic_search.config import resolve_index_dir
from arion_agent.semantic_search.incremental import IncrementalIndexer, IndexerStatus
from arion_agent.semantic_search.indexer import IndexStats, index_workspace, plan_sync, scan_manifest
from arion_agent.semantic_search.retriever import SearchResult, hybrid_search
from arion_agent.semantic_search.service import SearchService, SearchServiceConfig

__all__ = [
    "SearchResult",
    "IndexStats",
    "IndexerStatus",
    "SearchService",
    "SearchServiceConfig",
    "IncrementalIndexer",
    "hybrid_search",
    "index_workspace",
    "plan_sync",
    "scan_manifest",
    "resolve_index_dir",
]

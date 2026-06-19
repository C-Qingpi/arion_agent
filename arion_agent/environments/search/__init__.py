"""Search environment: optional workspace semantic search."""

import logging as _logging

_logger = _logging.getLogger(__name__)

from arion_agent.environments.search.config import SearchConfig  # noqa: E402

try:
    from arion_agent.environments.search.middleware import SearchEnvironment
    from arion_agent.semantic_search.service import SearchService

    _SEARCH_AVAILABLE = True
except ImportError:
    _SEARCH_AVAILABLE = False
    _logger.debug(
        "Search environment unavailable: semantic search dependencies not installed. "
        "Install with: pip install arion-agent[search]"
    )

    SearchEnvironment = None  # type: ignore[assignment,misc]
    SearchService = None  # type: ignore[assignment,misc]


def is_search_available() -> bool:
    return _SEARCH_AVAILABLE


__all__ = [
    "SearchConfig",
    "SearchEnvironment",
    "SearchService",
    "is_search_available",
]

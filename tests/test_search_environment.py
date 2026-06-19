"""Tests for optional SearchEnvironment middleware."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401


def test_search_environment_lifecycle():
    from arion_agent.environments.search.middleware import SearchEnvironment

    mock_service = MagicMock()
    env = SearchEnvironment("/tmp/workspace", service=mock_service, system_prompt=False)

    assert len(env.tools) == 1
    assert env.tools[0].name == "semantic_search"

    env.before_agent({})
    mock_service.start.assert_called_once()

    env.after_agent({})
    mock_service.stop.assert_called_once()


def test_search_tool_formats_empty_results():
    from arion_agent.environments.search.tools import create_search_tools

    mock_service = MagicMock()
    mock_service.search.return_value = []
    mock_service.status.return_value = MagicMock(indexed_files=0, total_files=12)

    tool = create_search_tools(mock_service, min_score=0.32, default_num_results=5)[0]
    out = tool.invoke({"query": "checkpoint sqlite"})
    assert "index still building" in out


def test_search_import_graceful():
    from arion_agent.environments import search as search_env

    assert hasattr(search_env, "is_search_available")
    assert hasattr(search_env, "SearchEnvironment")

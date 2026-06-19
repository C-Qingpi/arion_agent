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
    mock_service.stop.assert_not_called()


def test_search_tool_formats_empty_results():
    from arion_agent.environments.search.tools import create_search_tools, format_empty_search_message

    mock_service = MagicMock()
    mock_service.search.return_value = []
    mock_service.status.return_value = MagicMock(
        indexed_files=0,
        total_files=148,
        chunk_count=0,
        embedding=True,
        embedder_ready=False,
        pending_files=148,
        running=True,
        thread_alive=True,
        last_error=None,
        initial_sync_done=False,
    )

    tool = create_search_tools(mock_service, min_score=0.32, default_num_results=5)[0]
    out = tool.invoke({"query": "checkpoint sqlite"})
    assert "startup in progress" in out
    assert "148" in out

    st = mock_service.status.return_value
    st.embedder_ready = True
    st.embedding = True
    assert "first batch" in format_empty_search_message(st)

    st.indexed_files = 12
    st.chunk_count = 40
    st.initial_sync_done = False
    st.embedding = False
    out = tool.invoke({"query": "other"})
    assert "indexed portion" in out


def test_search_import_graceful():
    from arion_agent.environments import search as search_env

    assert hasattr(search_env, "is_search_available")
    assert hasattr(search_env, "SearchEnvironment")

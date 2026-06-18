"""Phase 17a.4 - Unit tests for Docker-safe SQLite VFS and container detection.

These tests use mocks to verify code paths without requiring Docker.
They validate:
  - is_container() detection logic
  - _setup_checkpointer() VFS and pragma selection
  - create_checkpointer() docker_safe parameter
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---- is_container() tests ----


class TestIsContainer:
    """Test arion_agent.util.runtime.is_container()."""

    def setup_method(self):
        import arion_agent.util.runtime as mod
        mod._is_container = None

    def teardown_method(self):
        import arion_agent.util.runtime as mod
        mod._is_container = None

    @patch("os.path.exists")
    @patch("sys.platform", "linux")
    def test_returns_true_when_dockerenv_exists(self, mock_exists):
        mock_exists.side_effect = lambda p: p == "/.dockerenv"
        from arion_agent.util.runtime import is_container
        assert is_container() is True

    @patch("os.path.exists")
    @patch("sys.platform", "linux")
    def test_returns_true_when_containerenv_exists(self, mock_exists):
        mock_exists.side_effect = lambda p: p == "/run/.containerenv"
        from arion_agent.util.runtime import is_container
        assert is_container() is True

    @patch("os.path.exists", return_value=False)
    @patch("sys.platform", "linux")
    def test_returns_false_on_bare_linux(self, mock_exists):
        from arion_agent.util.runtime import is_container
        assert is_container() is False

    @patch("os.path.exists", return_value=True)
    @patch("sys.platform", "win32")
    def test_returns_false_on_windows(self, mock_exists):
        from arion_agent.util.runtime import is_container
        assert is_container() is False

    @patch("os.path.exists", return_value=True)
    @patch("sys.platform", "darwin")
    def test_returns_false_on_macos(self, mock_exists):
        from arion_agent.util.runtime import is_container
        assert is_container() is False

    @patch("os.path.exists")
    @patch("sys.platform", "linux")
    def test_result_is_cached(self, mock_exists):
        mock_exists.side_effect = lambda p: p == "/.dockerenv"
        from arion_agent.util.runtime import is_container
        first = is_container()
        mock_exists.side_effect = lambda p: False
        second = is_container()
        assert first is True
        assert second is True


# ---- _setup_checkpointer() tests ----


class TestSetupCheckpointer:
    """Test arion_agent.graph._setup_checkpointer() VFS selection."""

    @patch("arion_agent.util.runtime.is_container", return_value=True)
    def test_uses_unix_none_vfs_in_container(self, mock_container, tmp_path):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        with (
            patch("aiosqlite.connect", new_callable=AsyncMock, return_value=mock_conn) as mock_connect,
            patch("langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver", return_value=mock_saver),
        ):
            from arion_agent.graph import _setup_checkpointer
            result = _setup_checkpointer(tmp_path, checkpointer=True)

            call_args = mock_connect.call_args
            uri_arg = call_args[0][0]
            assert "vfs=unix-none" in uri_arg
            assert call_args[1].get("uri") is True

    @patch("arion_agent.util.runtime.is_container", return_value=True)
    def test_sets_pragmas_in_container(self, mock_container, tmp_path):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        with (
            patch("aiosqlite.connect", new_callable=AsyncMock, return_value=mock_conn),
            patch("langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver", return_value=mock_saver),
        ):
            from arion_agent.graph import _setup_checkpointer
            _setup_checkpointer(tmp_path, checkpointer=True)

            execute_calls = [str(c) for c in mock_conn.execute.call_args_list]
            assert any("journal_mode=DELETE" in c for c in execute_calls)
            assert any("mmap_size=0" in c for c in execute_calls)

    @patch("arion_agent.util.runtime.is_container", return_value=False)
    def test_uses_plain_path_on_native(self, mock_container, tmp_path):
        mock_conn = AsyncMock()
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        with (
            patch("aiosqlite.connect", new_callable=AsyncMock, return_value=mock_conn) as mock_connect,
            patch("langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver", return_value=mock_saver),
        ):
            from arion_agent.graph import _setup_checkpointer
            _setup_checkpointer(tmp_path, checkpointer=True)

            call_args = mock_connect.call_args
            uri_arg = call_args[0][0]
            assert "vfs=" not in uri_arg
            assert "uri" not in call_args[1] or call_args[1].get("uri") is not True

    def test_returns_none_when_false(self, tmp_path):
        from arion_agent.graph import _setup_checkpointer
        assert _setup_checkpointer(tmp_path, checkpointer=False) is None

    def test_returns_none_when_none(self, tmp_path):
        from arion_agent.graph import _setup_checkpointer
        assert _setup_checkpointer(tmp_path, checkpointer=None) is None

    def test_returns_instance_when_given(self, tmp_path):
        from arion_agent.graph import _setup_checkpointer
        sentinel = MagicMock()
        assert _setup_checkpointer(tmp_path, checkpointer=sentinel) is sentinel


# ---- create_checkpointer() tests ----


class TestCreateCheckpointer:
    """Test arion_agent.session.create_checkpointer() docker_safe parameter."""

    @pytest.mark.asyncio
    async def test_docker_safe_true_uses_unix_none(self, tmp_path):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        db_path = tmp_path / "test.db"

        with (
            patch("aiosqlite.connect", new_callable=AsyncMock, return_value=mock_conn) as mock_connect,
            patch("arion_agent.session.AsyncSqliteSaver", return_value=mock_saver),
        ):
            from arion_agent.session import create_checkpointer
            await create_checkpointer(db_path, docker_safe=True)

            call_args = mock_connect.call_args
            uri_arg = call_args[0][0]
            assert "vfs=unix-none" in uri_arg
            assert call_args[1].get("uri") is True

    @pytest.mark.asyncio
    async def test_docker_safe_false_uses_plain_path(self, tmp_path):
        mock_conn = AsyncMock()
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        db_path = tmp_path / "test.db"

        with (
            patch("aiosqlite.connect", new_callable=AsyncMock, return_value=mock_conn) as mock_connect,
            patch("arion_agent.session.AsyncSqliteSaver", return_value=mock_saver),
        ):
            from arion_agent.session import create_checkpointer
            await create_checkpointer(db_path, docker_safe=False)

            call_args = mock_connect.call_args
            uri_arg = call_args[0][0]
            assert "vfs=" not in uri_arg

    @pytest.mark.asyncio
    async def test_docker_safe_none_auto_detects(self, tmp_path):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_saver = MagicMock()
        mock_saver.setup = AsyncMock()

        db_path = tmp_path / "test.db"

        with (
            patch("aiosqlite.connect", new_callable=AsyncMock, return_value=mock_conn) as mock_connect,
            patch("arion_agent.session.AsyncSqliteSaver", return_value=mock_saver),
            patch("arion_agent.util.runtime.is_container", return_value=True),
        ):
            from arion_agent.session import create_checkpointer
            await create_checkpointer(db_path)

            call_args = mock_connect.call_args
            uri_arg = call_args[0][0]
            assert "vfs=unix-none" in uri_arg

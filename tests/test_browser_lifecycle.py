"""Phase 17b.5-6 - Unit tests for browser lifecycle methods and tab adoption.

Tests use mocks (no real browser). Validates:
  - status() read-only inspection
  - reconnect() teardown and relaunch
  - _get_page() delegation to reconnect()
  - _adopt_existing_pages() tab recovery
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arion_agent.environments.browser.config import BrowserConfig
from arion_agent.environments.browser.session import BrowserSession

PLAYWRIGHT_AVAILABLE = True
try:
    from playwright.async_api import async_playwright
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def _make_session(**config_kwargs) -> BrowserSession:
    return BrowserSession(BrowserConfig(**config_kwargs))


# ---- status() tests ----


class TestStatus:

    def test_not_connected_by_default(self):
        session = _make_session()
        info = session.status()
        assert info["connected"] is False
        assert info["mode"] == "not_connected"
        assert info["tabs"] == []
        assert info["headless"] is False

    def test_connected_local(self):
        session = _make_session()
        session._launched = True
        session._context = MagicMock()
        info = session.status()
        assert info["connected"] is True
        assert info["mode"] == "local"
        assert info["endpoint"] == "local"

    def test_connected_cdp(self):
        session = _make_session(cdp_endpoint="http://localhost:9222")
        session._launched = True
        session._context = MagicMock()
        info = session.status()
        assert info["connected"] is True
        assert info["mode"] == "cdp"
        assert info["endpoint"] == "http://localhost:9222"

    def test_connected_ws(self):
        session = _make_session(ws_endpoint="ws://localhost:3000")
        session._launched = True
        session._context = MagicMock()
        info = session.status()
        assert info["connected"] is True
        assert info["mode"] == "ws"

    def test_shows_tabs(self):
        session = _make_session()
        session._launched = True
        session._context = MagicMock()
        mock_page = MagicMock()
        mock_page.url = "https://example.com"
        session._tabs["default"] = mock_page
        info = session.status()
        assert len(info["tabs"]) == 1
        assert info["tabs"][0]["name"] == "default"
        assert info["tabs"][0]["url"] == "https://example.com"

    def test_headless_flag(self):
        session = _make_session(headless=True)
        info = session.status()
        assert info["headless"] is True


# ---- reconnect() tests ----


class TestReconnect:

    @pytest.mark.asyncio
    async def test_reconnect_success(self):
        session = _make_session()
        session._launched = True
        session._context = MagicMock()
        session._browser = MagicMock()
        session._tabs["old"] = MagicMock()

        mock_pw = AsyncMock()
        mock_context = MagicMock()
        mock_context.pages = []
        mock_browser = MagicMock()
        mock_browser.contexts = [mock_context]

        with patch("arion_agent.environments.browser.session.BrowserSession._ensure_launched") as mock_launch:
            async def fake_launch():
                session._launched = True
                session._context = mock_context
            mock_launch.side_effect = fake_launch

            result = await session.reconnect()

        assert "reconnected" in result.lower()
        assert session._launched is True
        assert "old" not in session._tabs

    @pytest.mark.asyncio
    async def test_reconnect_failure_returns_error_string(self):
        session = _make_session()

        with patch("arion_agent.environments.browser.session.BrowserSession._ensure_launched") as mock_launch:
            mock_launch.side_effect = ConnectionError("CDP endpoint unreachable")
            result = await session.reconnect()

        assert "failed" in result.lower()
        assert "unreachable" in result.lower()
        assert session._launched is False

    @pytest.mark.asyncio
    async def test_reconnect_clears_old_state(self):
        session = _make_session()
        session._tabs["tab1"] = MagicMock()
        session._console_logs["tab1"] = ["log1"]
        session._launched = True
        session._context = MagicMock()
        session._browser = MagicMock()
        mock_pw = AsyncMock()
        session._playwright = mock_pw

        with patch("arion_agent.environments.browser.session.BrowserSession._ensure_launched") as mock_launch:
            async def fake_launch():
                session._launched = True
                session._context = MagicMock()
                session._context.pages = []
            mock_launch.side_effect = fake_launch

            await session.reconnect()

        mock_pw.stop.assert_awaited_once()


# ---- _get_page() resurrection via reconnect() ----


class TestGetPageResurrection:

    @pytest.mark.asyncio
    async def test_get_page_delegates_to_reconnect_on_context_failure(self):
        session = _make_session()
        session._launched = True

        mock_context_dead = MagicMock()
        mock_context_dead.new_page = AsyncMock(side_effect=Exception("context closed"))
        session._context = mock_context_dead

        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.on = MagicMock()

        call_count = 0
        async def fake_reconnect():
            nonlocal call_count
            call_count += 1
            session._launched = True
            mock_context_alive = MagicMock()
            mock_context_alive.new_page = AsyncMock(return_value=mock_page)
            mock_context_alive.pages = []
            session._context = mock_context_alive
            return "Browser reconnected (local)."

        with patch.object(session, "reconnect", side_effect=fake_reconnect):
            page = await session._get_page("test")

        assert call_count == 1
        assert page is mock_page


# ---- _adopt_existing_pages() tests ----


class TestAdoptExistingPages:

    def test_adopts_pages_from_context(self):
        session = _make_session()
        mock_page_1 = MagicMock()
        mock_page_1.url = "https://example.com"
        mock_page_1.on = MagicMock()
        mock_page_2 = MagicMock()
        mock_page_2.url = "https://google.com"
        mock_page_2.on = MagicMock()

        mock_context = MagicMock()
        mock_context.pages = [mock_page_1, mock_page_2]
        session._context = mock_context

        session._adopt_existing_pages()

        assert len(session._tabs) == 2
        assert "recovered-0" in session._tabs
        assert "recovered-1" in session._tabs
        assert session._tabs["recovered-0"] is mock_page_1
        assert mock_page_1.on.call_count == 2

    def test_no_pages_means_no_adoption(self):
        session = _make_session()
        mock_context = MagicMock()
        mock_context.pages = []
        session._context = mock_context

        session._adopt_existing_pages()
        assert len(session._tabs) == 0

    def test_no_context_is_safe(self):
        session = _make_session()
        session._context = None
        session._adopt_existing_pages()
        assert len(session._tabs) == 0

    @pytest.mark.asyncio
    async def test_reconnect_calls_adopt(self):
        session = _make_session(cdp_endpoint="http://localhost:9222")

        mock_page = MagicMock()
        mock_page.url = "https://existing.com"
        mock_page.on = MagicMock()

        with patch.object(session, "_ensure_launched") as mock_launch:
            async def fake_launch():
                session._launched = True
                mock_ctx = MagicMock()
                mock_ctx.pages = [mock_page]
                session._context = mock_ctx
            mock_launch.side_effect = fake_launch

            result = await session.reconnect()

        assert "Adopted 1 existing tab" in result
        assert "recovered-0" in session._tabs


# ---- Integration tests with real headless browser ----


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
class TestRealBrowserLifecycle:

    @pytest.mark.asyncio
    async def test_status_before_and_after_launch(self):
        session = BrowserSession(BrowserConfig(headless=True, stealth=False, humanize=False))

        info_before = session.status()
        assert info_before["connected"] is False

        await session._ensure_launched()
        info_after = session.status()
        assert info_after["connected"] is True
        assert info_after["mode"] == "local"

        await session.close()
        info_closed = session.status()
        assert info_closed["connected"] is False

    @pytest.mark.asyncio
    async def test_reconnect_after_close(self):
        session = BrowserSession(BrowserConfig(headless=True, stealth=False, humanize=False))

        page = await session._get_page("test")
        await page.goto("about:blank")
        assert session.status()["connected"] is True

        await session.close()
        assert session.status()["connected"] is False

        result = await session.reconnect()
        assert "reconnected" in result.lower()
        assert session.status()["connected"] is True

        page2 = await session._get_page("test2")
        title = await page2.title()
        assert isinstance(title, str)

        await session.close()

    @pytest.mark.asyncio
    async def test_resurrection_via_get_page(self):
        session = BrowserSession(BrowserConfig(headless=True, stealth=False, humanize=False))

        page = await session._get_page("tab1")
        await page.goto("about:blank")

        await session.close()

        page2 = await session._get_page("tab2")
        title = await page2.title()
        assert isinstance(title, str)
        assert session.status()["connected"] is True

        await session.close()

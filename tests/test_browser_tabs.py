"""Unit tests for multi-tab management, save_page, and tool tab routing.

Tests use mocks (no real browser). Validates:
  - create_tab() creates named tabs and optionally navigates
  - save_page() captures text and HTML to disk
  - Tab management tools are present and functional
  - Existing tools route to correct tab via optional tab parameter
  - browser_save_page resolves relative paths against workspace_dir
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arion_agent.environments.browser.config import BrowserConfig
from arion_agent.environments.browser.session import BrowserSession


def _make_session(**config_kwargs) -> BrowserSession:
    return BrowserSession(BrowserConfig(**config_kwargs))


def _mock_launched_session() -> BrowserSession:
    """Return a session that appears launched with a mock context."""
    session = _make_session()
    session._launched = True
    session._context = MagicMock()
    return session


# ---- create_tab() ----


class TestCreateTab:

    @pytest.mark.asyncio
    async def test_create_tab_without_url(self):
        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Blank")
        mock_page.on = MagicMock()
        session._context.new_page = AsyncMock(return_value=mock_page)

        result = await session.create_tab("research")

        assert "research" in result
        assert "research" in session._tabs

    @pytest.mark.asyncio
    async def test_create_tab_with_url_navigates(self):
        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.on = MagicMock()
        mock_page.goto = AsyncMock()
        session._context.new_page = AsyncMock(return_value=mock_page)

        result = await session.create_tab("docs", "https://example.com")

        assert "docs" in session._tabs
        mock_page.goto.assert_awaited_once()
        assert "Navigated" in result or "example" in result.lower()

    @pytest.mark.asyncio
    async def test_create_tab_reuses_existing(self):
        session = _mock_launched_session()
        existing_page = AsyncMock()
        existing_page.title = AsyncMock(return_value="Already open")
        session._tabs["reuse"] = existing_page
        session._context.new_page = AsyncMock()

        result = await session.create_tab("reuse")

        assert "reuse" in result
        session._context.new_page.assert_not_awaited()


# ---- save_page() ----


class TestSavePage:

    @pytest.mark.asyncio
    async def test_save_text_format(self):
        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.url = "https://example.com"
        mock_page.on = MagicMock()

        mock_locator = AsyncMock()
        mock_locator.first = AsyncMock()
        mock_locator.first.text_content = AsyncMock(return_value="Hello World content")
        mock_page.locator = MagicMock(return_value=mock_locator)

        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as td:
            save_path = Path(td) / "output.txt"
            result = await session.save_page(save_path, format="text")

            assert save_path.exists()
            assert "Hello World content" in save_path.read_text(encoding="utf-8")
            assert "Saved text" in result
            assert "Test Page" in result

    @pytest.mark.asyncio
    async def test_save_html_format(self):
        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="HTML Page")
        mock_page.url = "https://example.com/page"
        mock_page.on = MagicMock()
        mock_page.content = AsyncMock(return_value="<html><body><h1>Title</h1></body></html>")
        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as td:
            save_path = Path(td) / "page.html"
            result = await session.save_page(save_path, format="html")

            assert save_path.exists()
            content = save_path.read_text(encoding="utf-8")
            assert "<h1>Title</h1>" in content
            assert "Saved html" in result

    @pytest.mark.asyncio
    async def test_save_html_with_selector(self):
        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Article")
        mock_page.url = "https://example.com/article"
        mock_page.on = MagicMock()

        mock_locator = AsyncMock()
        mock_locator.first = AsyncMock()
        mock_locator.first.inner_html = AsyncMock(return_value="<p>Article body</p>")
        mock_page.locator = MagicMock(return_value=mock_locator)

        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as td:
            save_path = Path(td) / "article.html"
            result = await session.save_page(
                save_path, format="html", selector=".article-body",
            )

            content = save_path.read_text(encoding="utf-8")
            assert "<p>Article body</p>" in content
            mock_page.locator.assert_called_with(".article-body")

    @pytest.mark.asyncio
    async def test_save_with_wait_for(self):
        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Dynamic")
        mock_page.url = "https://spa.example.com"
        mock_page.on = MagicMock()
        mock_page.wait_for_selector = AsyncMock()

        mock_locator = AsyncMock()
        mock_locator.first = AsyncMock()
        mock_locator.first.text_content = AsyncMock(return_value="Loaded content")
        mock_page.locator = MagicMock(return_value=mock_locator)

        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as td:
            save_path = Path(td) / "dynamic.txt"
            result = await session.save_page(
                save_path, wait_for=".loaded", format="text",
            )

            mock_page.wait_for_selector.assert_awaited_once_with(
                ".loaded", timeout=session._config.timeout_ms,
            )
            assert save_path.exists()

    @pytest.mark.asyncio
    async def test_save_wait_for_failure(self):
        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Timeout")
        mock_page.on = MagicMock()
        mock_page.wait_for_selector = AsyncMock(
            side_effect=TimeoutError("Selector not found"),
        )
        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as td:
            save_path = Path(td) / "fail.txt"
            result = await session.save_page(
                save_path, wait_for=".missing", format="text",
            )

            assert "Wait failed" in result
            assert not save_path.exists()

    @pytest.mark.asyncio
    async def test_save_creates_parent_dirs(self):
        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Nested")
        mock_page.url = "https://example.com"
        mock_page.on = MagicMock()

        mock_locator = AsyncMock()
        mock_locator.first = AsyncMock()
        mock_locator.first.text_content = AsyncMock(return_value="nested file")
        mock_page.locator = MagicMock(return_value=mock_locator)

        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as td:
            save_path = Path(td) / "deep" / "nested" / "output.txt"
            await session.save_page(save_path, format="text")

            assert save_path.exists()

    @pytest.mark.asyncio
    async def test_save_to_specific_tab(self):
        session = _mock_launched_session()

        mock_page_a = AsyncMock()
        mock_page_a.title = AsyncMock(return_value="Tab A")
        mock_page_a.url = "https://a.example.com"
        mock_page_a.on = MagicMock()

        mock_locator_a = AsyncMock()
        mock_locator_a.first = AsyncMock()
        mock_locator_a.first.text_content = AsyncMock(return_value="Content from A")
        mock_page_a.locator = MagicMock(return_value=mock_locator_a)

        session._tabs["tab-a"] = mock_page_a

        with tempfile.TemporaryDirectory() as td:
            save_path = Path(td) / "from_a.txt"
            result = await session.save_page(save_path, tab="tab-a", format="text")

            assert "Content from A" in save_path.read_text(encoding="utf-8")
            assert "Tab A" in result


# ---- list_tabs() ----


class TestListTabs:

    def test_no_tabs(self):
        session = _make_session()
        assert "No open tabs" in session.list_tabs()

    def test_with_tabs(self):
        session = _make_session()
        page1 = MagicMock()
        page1.url = "https://one.com"
        page2 = MagicMock()
        page2.url = "https://two.com"
        session._tabs["first"] = page1
        session._tabs["second"] = page2

        result = session.list_tabs()
        assert "first" in result
        assert "second" in result
        assert "https://one.com" in result
        assert "2/8" in result


# ---- Tool creation and tab routing ----


class TestToolCreation:

    def test_all_tools_present(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _make_session()
        tools = create_browser_tools(session)
        tool_names = {t.name for t in tools}

        expected = {
            "browser_action", "browser_snapshot", "browser_screenshot",
            "browser_wait_for_human", "browser_console", "browser_eval_js",
            "browser_tab_new", "browser_tab_list", "browser_tab_close",
            "browser_save_page",
            "http_request",
            "browser_status", "browser_reconnect", "browser_close",
        }
        assert expected == tool_names

    def test_tool_count(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _make_session()
        tools = create_browser_tools(session)
        assert len(tools) == 14


class TestToolTabRouting:

    @pytest.mark.asyncio
    async def test_browser_action_uses_default_tab(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Default")
        mock_page.on = MagicMock()
        mock_page.goto = AsyncMock()
        session._context.new_page = AsyncMock(return_value=mock_page)

        tools = create_browser_tools(session, tab_name="agent1")
        action_tool = [t for t in tools if t.name == "browser_action"][0]

        await action_tool.ainvoke({
            "action": "navigate",
            "target": "https://example.com",
        })

        assert "agent1" in session._tabs

    @pytest.mark.asyncio
    async def test_browser_action_uses_override_tab(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Override")
        mock_page.on = MagicMock()
        mock_page.goto = AsyncMock()
        session._context.new_page = AsyncMock(return_value=mock_page)

        tools = create_browser_tools(session, tab_name="agent1")
        action_tool = [t for t in tools if t.name == "browser_action"][0]

        await action_tool.ainvoke({
            "action": "navigate",
            "target": "https://example.com",
            "tab": "custom-tab",
        })

        assert "custom-tab" in session._tabs

    @pytest.mark.asyncio
    async def test_browser_snapshot_tab_override(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Snap")
        mock_page.url = "https://example.com"
        mock_page.on = MagicMock()
        mock_page.evaluate = AsyncMock(return_value="<h1>Hello</h1>")
        session._context.new_page = AsyncMock(return_value=mock_page)

        tools = create_browser_tools(session, tab_name="main")
        snap_tool = [t for t in tools if t.name == "browser_snapshot"][0]

        await snap_tool.ainvoke({"tab": "other"})

        assert "other" in session._tabs

    @pytest.mark.asyncio
    async def test_tab_new_tool(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="New Tab")
        mock_page.on = MagicMock()
        session._context.new_page = AsyncMock(return_value=mock_page)

        tools = create_browser_tools(session, tab_name="main")
        new_tab = [t for t in tools if t.name == "browser_tab_new"][0]

        result = await new_tab.ainvoke({"name": "research"})

        assert "research" in session._tabs
        assert "research" in result

    @pytest.mark.asyncio
    async def test_tab_list_tool(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _make_session()
        page = MagicMock()
        page.url = "https://example.com"
        session._tabs["tab1"] = page

        tools = create_browser_tools(session, tab_name="main")
        list_tool = [t for t in tools if t.name == "browser_tab_list"][0]

        result = await list_tool.ainvoke({})

        assert "tab1" in result
        assert "https://example.com" in result

    @pytest.mark.asyncio
    async def test_tab_close_tool(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _make_session()
        mock_page = AsyncMock()
        session._tabs["temp"] = mock_page

        tools = create_browser_tools(session, tab_name="main")
        close_tool = [t for t in tools if t.name == "browser_tab_close"][0]

        result = await close_tool.ainvoke({"name": "temp"})

        assert "temp" not in session._tabs
        assert "closed" in result.lower()


class TestSavePageTool:

    @pytest.mark.asyncio
    async def test_save_page_relative_path(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Save Test")
        mock_page.url = "https://example.com"
        mock_page.on = MagicMock()

        mock_locator = AsyncMock()
        mock_locator.first = AsyncMock()
        mock_locator.first.text_content = AsyncMock(return_value="Saved text")
        mock_page.locator = MagicMock(return_value=mock_locator)

        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as ws:
            tools = create_browser_tools(
                session, tab_name="main", workspace_dir=Path(ws),
            )
            save_tool = [t for t in tools if t.name == "browser_save_page"][0]

            result = await save_tool.ainvoke({
                "path": "downloads/article.txt",
                "format": "text",
            })

            saved = Path(ws) / "downloads" / "article.txt"
            assert saved.exists()
            assert "Saved text" in saved.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_save_page_absolute_path(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Abs Test")
        mock_page.url = "https://example.com"
        mock_page.on = MagicMock()

        mock_locator = AsyncMock()
        mock_locator.first = AsyncMock()
        mock_locator.first.text_content = AsyncMock(return_value="Abs content")
        mock_page.locator = MagicMock(return_value=mock_locator)

        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as target:
            tools = create_browser_tools(
                session, tab_name="main", workspace_dir=Path(ws),
            )
            save_tool = [t for t in tools if t.name == "browser_save_page"][0]
            abs_path = str(Path(target) / "output.txt")

            result = await save_tool.ainvoke({
                "path": abs_path,
                "format": "text",
            })

            assert Path(abs_path).exists()
            assert "Abs content" in Path(abs_path).read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_save_page_no_workspace_uses_path_as_is(self):
        from arion_agent.environments.browser.tools import create_browser_tools

        session = _mock_launched_session()
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="No WS")
        mock_page.url = "https://example.com"
        mock_page.on = MagicMock()

        mock_locator = AsyncMock()
        mock_locator.first = AsyncMock()
        mock_locator.first.text_content = AsyncMock(return_value="Raw path")
        mock_page.locator = MagicMock(return_value=mock_locator)

        session._context.new_page = AsyncMock(return_value=mock_page)

        with tempfile.TemporaryDirectory() as td:
            tools = create_browser_tools(session, tab_name="main")
            save_tool = [t for t in tools if t.name == "browser_save_page"][0]
            target = str(Path(td) / "out.txt")

            await save_tool.ainvoke({"path": target, "format": "text"})

            assert Path(target).exists()

"""Browser environment middleware: optional, provides web interaction tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.environments.browser.config import BrowserConfig
from arion_agent.environments.browser.session import BrowserSession
from arion_agent.environments.browser.tools import create_browser_tools
from arion_agent.middleware.base import ArionMiddleware

BROWSER_SYSTEM_PROMPT = """\
<browser_environment>
You have access to a persistent browser session for web automation.

Use browser_snapshot to see the page as simplified HTML. It shows
interactive elements, page structure, and key attributes while stripping
scripts, styles, SVGs, and hidden content. Use the attributes from the
snapshot to build CSS selectors for browser_action (e.g. click
[data-item-key="value"], fill input[name="query"]).

Use browser_screenshot only when visual context is needed (layout
debugging, image verification, CAPTCHA). Do not use browser_eval_js
to read page content -- use browser_snapshot instead.

For login pages or password prompts, ask the operator to handle
authentication manually in the visible browser window.

Use http_request ONLY for known API endpoints (REST APIs, webhooks).
For websites and web pages, use browser_action instead.

If a browser operation fails with a connection error, use browser_status
to check the connection state, then browser_reconnect to restore it.
Use browser_close when the browser is no longer needed.

Tab management: you can open multiple tabs with browser_tab_new, list
them with browser_tab_list, and close them with browser_tab_close.
All page tools (browser_action, browser_snapshot, browser_screenshot,
browser_eval_js, browser_console, browser_save_page) require a tab
parameter. Use "default" for the main tab.
Use multiple tabs for parallel page loading or comparing content.

Use browser_save_page to capture rendered page content to a file. Set
format='text' for readable extraction, format='html' for full raw HTML.
Use wait_for to wait for dynamic content before capturing.

Use browser_download to start a file download (non-blocking). Pass a
URL or CSS selector as source. Downloads run in the background; use
browser_download_list to check progress. The browser's suggested
filename is used when the path has no extension.
</browser_environment>"""


class BrowserEnvironment(ArionMiddleware):
    """Optional middleware providing persistent browser and HTTP tools.

    The browser session is workspace-level (one Chrome process shared across
    all agents). Each agent gets its own named tab. Login cookies are shared.

    For multi-agent setups, pass the same BrowserSession to all agents:
        session = BrowserSession(BrowserConfig(channel="chrome", user_data_dir=...))
        agent_a = create_arion_agent(..., middleware=[BrowserEnvironment(session=session, agent_id="a")])
        agent_b = create_arion_agent(..., middleware=[BrowserEnvironment(session=session, agent_id="b")])
    """

    def __init__(
        self,
        config: BrowserConfig | None = None,
        *,
        session: BrowserSession | None = None,
        workspace_dir: str | None = None,
        agent_id: str = "default",
        tool_factory: Any | None = None,
        system_prompt: str | None | bool = None,
    ) -> None:
        self._config = config or BrowserConfig()
        self._session = session or BrowserSession(self._config)
        self._agent_id = agent_id
        if system_prompt is False:
            self._system_prompt: str | None = None
        elif isinstance(system_prompt, str):
            self._system_prompt = system_prompt
        else:
            self._system_prompt = BROWSER_SYSTEM_PROMPT
        from pathlib import Path
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None
        screenshot_dir = self._workspace_dir / ".arion" / "screenshots" if self._workspace_dir else None
        if self._workspace_dir and not self._session._storage_dir:
            self._session.set_storage_dir(self._workspace_dir / ".arion" / "browser")
        if tool_factory is not None:
            self._tools = tool_factory(self._session, screenshot_dir=screenshot_dir, tab_name=agent_id)
        else:
            self._tools = create_browser_tools(
                self._session, screenshot_dir=screenshot_dir, tab_name=agent_id,
                workspace_dir=self._workspace_dir,
            )

        if self._workspace_dir:
            from arion_agent.environments.browser.skills import seed_browser_skills
            seed_browser_skills(self._workspace_dir)

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    @property
    def session(self) -> BrowserSession:
        return self._session

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        if self._system_prompt:
            parts.append(self._system_prompt)
        return parts

    def after_agent(self, state: dict[str, Any]) -> None:
        """Close browser on agent completion if still open."""
        import asyncio
        if self._session.is_open:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except Exception:
                pass

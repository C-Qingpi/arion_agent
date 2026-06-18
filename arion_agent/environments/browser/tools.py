"""Browser environment tools for web interaction and tab management.

Core tools provide browser primitives (navigate, click, snapshot, etc.).
Tab management tools enable multi-tab workflows (parallel page loading,
content extraction from multiple sources). The save tool captures rendered
page content to disk.

Advanced browser workflows (multi-step navigation, form filling, scraping)
are handled via skills, not tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import aiohttp
from langchain_core.tools import tool

from arion_agent.environments.browser.session import BrowserSession


def create_browser_tools(
    session: BrowserSession,
    screenshot_dir: Path | None = None,
    tab_name: str = "default",
    workspace_dir: Path | None = None,
) -> list:
    """Create browser tools bound to a persistent session.

    Tools use ``tab_name`` as the fallback when the agent passes an empty
    tab value.  All page-level tools require an explicit ``tab`` argument
    so the agent is always aware of which tab it is operating on.
    """

    def _tab(override: str) -> str:
        return override if override else tab_name

    def _with_tabs(result: str, tab: str) -> str:
        return f"{result}\n{session.tab_context(_tab(tab))}"

    def _resolve_save_path(path: str) -> Path:
        if workspace_dir:
            from arion_agent.environments._sandbox.paths import resolve_path
            return resolve_path(path, workspace_dir)
        return Path(path)

    # ---- Core interaction tools ----

    @tool
    async def browser_action(
        action: Annotated[str, "Action: navigate, click, fill, type, select, scroll, wait"],
        tab: Annotated[str, "Tab name to operate on (e.g. 'default')"],
        target: Annotated[str, "URL (navigate), CSS selector (click/fill/type/select/wait), or empty (scroll)"] = "",
        value: Annotated[str, "Value to fill/type, option to select, direction to scroll (up/down), or seconds to wait"] = "",
    ) -> str:
        """Interact with the browser. Actions: navigate, click, fill, type, select, scroll, wait.

        Examples:
          action=navigate, tab=default, target=https://example.com
          action=click, tab=default, target=button[type=submit]
          action=fill, tab=default, target=#email, value=user@example.com
          action=type, tab=default, target=#search, value=query text
          action=scroll, tab=default, value=down
          action=wait, tab=default, target=.results
          action=navigate, tab=research, target=https://other.com
        """
        result = await session.action(action, target, value, tab=_tab(tab))
        return _with_tabs(result, tab)

    @tool
    async def browser_snapshot(
        tab: Annotated[str, "Tab name to snapshot (e.g. 'default')"],
    ) -> str:
        """Get a simplified HTML snapshot of the current page.

        Returns URL, title, and condensed HTML showing interactive elements
        (links, buttons, inputs), page structure (headings, nav, sections),
        and key attributes (id, href, role, data-*, aria-*). Hidden elements,
        scripts, styles, and SVGs are excluded.
        Use for: finding elements to interact with, reading page structure,
        form discovery. Attributes in the output can be used as CSS selectors
        in browser_action (e.g. [data-item-key="value"], a[href="/path"]).

        Before snapshotting: if you do not know which tab has the page, or
        if a previous snapshot showed about:blank, call browser_tab_list()
        first to see open tabs and their URLs, then pass the correct tab name.
        """
        result = await session.snapshot(tab=_tab(tab))
        return _with_tabs(result, tab)

    @tool
    async def browser_screenshot(
        tab: Annotated[str, "Tab name to capture (e.g. 'default')"],
    ) -> str:
        """Capture a visual screenshot of the current page.

        Returns the image for visual inspection. The screenshot is also
        saved to disk. Use only when you need visual context (layout
        debugging, image content, CAPTCHA detection). For text/structure
        tasks, prefer browser_snapshot.
        """
        result = await session.screenshot(tab=_tab(tab), save_dir=screenshot_dir)
        return _with_tabs(result, tab)

    @tool
    async def browser_wait_for_human(
        message: Annotated[str, "Message to display to the operator explaining what to do"],
        tab: Annotated[str, "Tab name (e.g. 'default')"],
        wait_for_selector: Annotated[str, "CSS selector to wait for after human acts (optional)"] = "",
        timeout: Annotated[int | float, "Max seconds to wait (default 180)"] = 180,
    ) -> str:
        """Pause and ask the human operator to act in the visible browser.

        Use for: login pages, CAPTCHAs, 2FA, or any action requiring
        human credentials or verification. The browser must be in visible
        mode (headless=False).
        """
        result = await session.wait_for_human(message, wait_for_selector, tab=_tab(tab), timeout=timeout)
        return _with_tabs(result, tab)

    @tool
    async def browser_console(
        tab: Annotated[str, "Tab name (e.g. 'default')"],
        clear: Annotated[bool, "Clear the log after reading"] = False,
    ) -> str:
        """Read browser console output (JS errors, warnings, console.log).

        Use for debugging frontend issues, checking for JS errors, or
        reading console.log output from web applications.
        """
        result = await session.console(clear, tab=_tab(tab))
        return _with_tabs(result, tab)

    @tool
    async def browser_eval_js(
        expression: Annotated[str, "JavaScript expression to evaluate in page context"],
        tab: Annotated[str, "Tab name (e.g. 'default')"],
    ) -> str:
        """Evaluate a JavaScript expression in the browser page context.

        Use for: reading DOM properties, checking JS state, running
        diagnostic queries. Returns the result as a string.
        """
        result = await session.evaluate_js(expression, tab=_tab(tab))
        return _with_tabs(result, tab)

    # ---- Tab management tools ----

    @tool
    async def browser_tab_new(
        name: Annotated[str, "Name for the new tab (used to reference it later)"],
        url: Annotated[str, "URL to navigate to (optional)"] = "",
    ) -> str:
        """Create a new named browser tab, optionally navigating to a URL.

        Use to open additional tabs for parallel page loading or comparing
        content across sites. Each tab is independent and retains its page
        state. Reference the tab by name in other browser tools.
        """
        result = await session.create_tab(name, url)
        return _with_tabs(result, name)

    @tool
    async def browser_tab_list() -> str:
        """List all open browser tabs with their names and current URLs.

        Call this first when you need to inspect the page: see which tabs
        exist and pick the right tab name for browser_snapshot or
        browser_action. Use again before closing tabs or switching context.
        """
        return session.list_tabs()

    @tool
    async def browser_tab_close(
        name: Annotated[str, "Name of the tab to close"],
    ) -> str:
        """Close a browser tab by name.

        Frees browser resources. The default agent tab can also be closed
        and will be recreated on next use.
        """
        result = await session.close_tab(name)
        return f"{result}\n{session.tab_context('')}"

    # ---- Page capture tools ----

    @tool
    async def browser_save_page(
        path: Annotated[str, "File path to save to (relative to workspace or absolute)"],
        tab: Annotated[str, "Tab name (e.g. 'default')"],
        selector: Annotated[str, "CSS selector to scope content (default: full page)"] = "",
        wait_for: Annotated[str, "CSS selector to wait for before capturing (for dynamic content)"] = "",
        format: Annotated[str, "'text' for plain text content, 'html' for rendered HTML"] = "text",
    ) -> str:
        """Save the current page content to a file.

        Captures rendered page content (after JavaScript execution) and
        writes it to disk. Use 'text' format for readable content extraction,
        'html' format for preserving structure. The wait_for parameter lets
        you wait for dynamic content to load before capturing.

        Examples:
          path=article.txt, selector=.article-body, wait_for=.article-body
          path=page.html, format=html
          path=results.txt, tab=research, selector=#results
        """
        resolved = _resolve_save_path(path)
        result = await session.save_page(
            resolved, tab=_tab(tab), selector=selector,
            wait_for=wait_for, format=format,
        )
        return _with_tabs(result, tab)

    @tool
    async def browser_download(
        source: Annotated[str, "URL to download, or CSS selector of a download link/button to click"],
        tab: Annotated[str, "Tab name (e.g. 'default')"],
        path: Annotated[str, "Workspace path to save to. If no file extension, the browser's suggested filename is used"] = "downloads",
    ) -> str:
        """Start a file download (non-blocking).

        Fires the download and returns immediately with a download ID.
        *source* can be a direct URL (starts with http) or a CSS selector
        to click. If *path* has no extension, the browser's suggested
        filename is appended. Use browser_download_list to check progress.
        """
        resolved = _resolve_save_path(path)
        result = await session.download(source, resolved, tab=_tab(tab))
        return _with_tabs(result, tab)

    @tool
    async def browser_download_list() -> str:
        """List all downloads and their status.

        Shows download ID, filename, status (starting/downloading/complete/
        failed), save path, and size. Use after browser_download to check
        if files have finished downloading.
        """
        return session.download_list()

    # ---- HTTP tool ----

    @tool
    async def http_request(
        method: Annotated[str, "HTTP method: GET, POST, PUT, DELETE, PATCH"] = "GET",
        url: Annotated[str, "Full URL to request"] = "",
        headers: Annotated[str, "Headers as key:value pairs, one per line"] = "",
        body: Annotated[str, "Request body (for POST/PUT/PATCH)"] = "",
    ) -> str:
        """Send a direct HTTP request. Use ONLY for known API endpoints
        (REST APIs, webhooks, JSON services). For websites and web pages,
        use browser_action instead.

        Args:
            method: HTTP method.
            url: Full URL.
            headers: One header per line, format 'Key: Value'.
            body: Request body for POST/PUT/PATCH.
        """
        if not url:
            return "Error: url is required."

        parsed_headers: dict[str, str] = {}
        if headers:
            for line in headers.strip().split("\n"):
                if ":" in line:
                    k, _, v = line.partition(":")
                    parsed_headers[k.strip()] = v.strip()

        try:
            async with aiohttp.ClientSession() as client:
                async with client.request(
                    method.upper(),
                    url,
                    headers=parsed_headers or None,
                    data=body.encode("utf-8") if body else None,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    status = resp.status
                    resp_headers = dict(resp.headers)
                    text = await resp.text()
                    if len(text) > 10000:
                        text = text[:10000] + f"\n[...truncated, {len(text)} chars total]"
                    return f"HTTP {status}\nHeaders: {resp_headers}\n\n{text}"
        except Exception as exc:
            return f"HTTP request failed: {exc}"

    # ---- Connection management tools ----

    @tool
    async def browser_status() -> str:
        """Check browser connection state without triggering a connection.

        Returns whether the browser is connected, the connection mode
        (local/cdp/ws), open tabs, and whether it is headless.
        Use this to diagnose browser issues before attempting reconnection.
        """
        info = session.status()
        lines = [
            f"connected: {info['connected']}",
            f"mode: {info['mode']}",
            f"endpoint: {info['endpoint']}",
            f"headless: {info['headless']}",
        ]
        if info["tabs"]:
            lines.append(f"tabs ({len(info['tabs'])}):")
            for t in info["tabs"]:
                lines.append(f"  {t['name']}: {t['url']}")
        else:
            lines.append("tabs: (none)")
        return "\n".join(lines)

    @tool
    async def browser_reconnect() -> str:
        """Reconnect or relaunch the browser after a failure.

        Use this after a browser operation fails or after browser_status
        shows disconnected. In local mode, this relaunches the browser
        process (cookies persist if user_data_dir is configured). In CDP
        mode, this reconnects to the remote browser and recovers open
        tabs if the browser is still running. If the remote browser was
        restarted by a supervisor, connects to the new instance.
        """
        return await session.reconnect()

    @tool
    async def browser_close() -> str:
        """Disconnect from the browser and release resources.

        In local mode, this closes the browser process (state is saved
        if user_data_dir is set). In CDP mode, this disconnects the
        agent from the remote browser without killing it -- the browser
        keeps running for the user. Use browser_reconnect to reconnect.
        """
        return await session.close()

    return [
        browser_action, browser_snapshot, browser_screenshot,
        browser_wait_for_human, browser_console, browser_eval_js,
        browser_tab_new, browser_tab_list, browser_tab_close,
        browser_save_page, browser_download, browser_download_list,
        http_request,
        browser_status, browser_reconnect, browser_close,
    ]

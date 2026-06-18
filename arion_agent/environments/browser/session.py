"""Persistent browser session using Playwright.

Workspace-level: one browser process shared across all agents.
Each agent owns a named tab (page) within the shared browser context.
Login cookies are shared because all tabs use the same context.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from arion_agent.environments.browser.config import BrowserConfig

logger = logging.getLogger(__name__)

MAX_TABS = 8
SNAPSHOT_MAX_CHARS = 50000

_SIMPLIFY_DOM_JS = r"""(() => {
  const SKIP_TAGS = new Set([
    'SCRIPT','STYLE','NOSCRIPT','SVG','LINK','META','HEAD',
    'IFRAME','CANVAS','VIDEO','AUDIO','SOURCE','TRACK','TEMPLATE','SLOT'
  ]);
  const INTERACTIVE_TAGS = new Set([
    'A','BUTTON','INPUT','SELECT','OPTION','TEXTAREA','LABEL',
    'FORM','DETAILS','SUMMARY'
  ]);
  const STRUCTURAL_TAGS = new Set([
    'NAV','HEADER','FOOTER','MAIN','SECTION','ARTICLE','ASIDE',
    'H1','H2','H3','H4','H5','H6','TABLE','THEAD','TBODY','TFOOT',
    'TR','TH','TD','UL','OL','LI','DL','DT','DD',
    'FIGURE','FIGCAPTION','DIALOG','P','BLOCKQUOTE','PRE','CODE','IMG'
  ]);
  const VOID_TAGS = new Set(['IMG','INPUT','BR','HR']);
  const KEEP_ATTRS = [
    'id','name','type','href','src','role',
    'aria-label','aria-expanded','aria-selected','aria-checked',
    'aria-haspopup','aria-controls','placeholder','value',
    'title','alt','for','action','method',
    'disabled','checked','selected','readonly','target'
  ];
  const MAX_TEXT = %%MAX_TEXT%%;
  const MAX_OUT = %%MAX_CHARS%%;
  let output = '';

  function isVisible(el) {
    if (el.getAttribute('aria-hidden') === 'true' || el.hidden) return false;
    try {
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    } catch (e) {}
    return true;
  }

  function buildAttrs(el) {
    let r = '';
    for (const name of KEEP_ATTRS) {
      const v = el.getAttribute(name);
      if (v != null && v !== '') {
        const s = (v.length > 80 ? v.slice(0, 80) + '...' : v).replace(/"/g, '&quot;');
        r += ' ' + name + '="' + s + '"';
      }
    }
    for (const at of el.attributes) {
      if (at.name.startsWith('data-') && !at.name.startsWith('data-v-') &&
          !KEEP_ATTRS.includes(at.name) && at.value.length < 60) {
        r += ' ' + at.name + '="' + at.value.replace(/"/g, '&quot;') + '"';
      }
    }
    const cls = el.className;
    if (typeof cls === 'string' && cls.trim()) {
      const arr = cls.trim().split(/\s+/).filter(c => c.length < 40).slice(0, 3);
      if (arr.length) r += ' class="' + arr.join(' ') + '"';
    }
    return r;
  }

  function shouldKeep(el) {
    if (INTERACTIVE_TAGS.has(el.tagName) || STRUCTURAL_TAGS.has(el.tagName)) return true;
    if (el.getAttribute('role') || el.getAttribute('aria-label') || el.id) return true;
    for (const at of el.attributes) {
      if (at.name.startsWith('data-') && !at.name.startsWith('data-v-')) return true;
    }
    if (el.onclick) return true;
    try { if (getComputedStyle(el).cursor === 'pointer') return true; } catch (e) {}
    return false;
  }

  function hasChildEls(el) {
    for (const c of el.childNodes) { if (c.nodeType === 1) return true; }
    return false;
  }

  function clip(t) { return t.length > MAX_TEXT ? t.slice(0, MAX_TEXT) + '...' : t; }

  function walk(node, depth, indent) {
    if (output.length >= MAX_OUT) return;

    if (node.nodeType === 3) {
      const t = node.textContent.replace(/\s+/g, ' ').trim();
      if (t) output += indent + clip(t) + '\n';
      return;
    }
    if (node.nodeType !== 1) return;

    const tag = node.tagName;
    if (SKIP_TAGS.has(tag) || !isVisible(node) || depth > 15) return;

    if (shouldKeep(node)) {
      const tl = tag.toLowerCase(), at = buildAttrs(node);
      if (VOID_TAGS.has(tag)) {
        output += indent + '<' + tl + at + ' />\n';
        return;
      }
      if (!hasChildEls(node)) {
        const t = node.textContent.replace(/\s+/g, ' ').trim();
        if (t) {
          output += indent + '<' + tl + at + '>' + clip(t) + '</' + tl + '>\n';
        } else if (INTERACTIVE_TAGS.has(tag)) {
          output += indent + '<' + tl + at + ' />\n';
        }
      } else {
        output += indent + '<' + tl + at + '>\n';
        for (const c of node.childNodes) walk(c, depth + 1, indent + '  ');
      }
    } else {
      for (const c of node.childNodes) walk(c, depth, indent);
    }

    if (node.shadowRoot) {
      for (const c of node.shadowRoot.childNodes) walk(c, depth, indent);
    }
  }

  if (!document.body) return '(no body element)';
  walk(document.body, 0, '');
  if (output.length >= MAX_OUT) output += '[...truncated]\n';
  return output;
})()"""

_STEALTH_UA_TEMPLATE = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/{version} Safari/537.36"
)


def _current_stable_chrome_version() -> str:
    """Return Chrome major version that stays plausible over time.

    Derives from current date: Chrome ships a major release roughly every
    4 weeks starting from v131 in Nov 2024.
    """
    from datetime import date
    baseline_version = 131
    baseline_date = date(2024, 11, 12)
    weeks = (date.today() - baseline_date).days / 7
    major = baseline_version + int(weeks / 4)
    return f"{major}.0.0.0"


_STEALTH_USER_AGENT = _STEALTH_UA_TEMPLATE.format(
    version=_current_stable_chrome_version()
)


def _resolve_cdp_endpoint(endpoint: str) -> str:
    """Resolve hostname in CDP endpoint to IP for Chrome's Host-header check.

    Chrome's DevTools HTTP server rejects requests whose Host header is not
    an IP address or ``localhost``.  When connecting from Docker via
    ``host.docker.internal``, Chrome returns 500.  Resolving to IP fixes this.
    """
    import socket
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(endpoint)
    hostname = parsed.hostname
    if not hostname or hostname in ("localhost", "127.0.0.1", "::1"):
        return endpoint
    try:
        ip = socket.gethostbyname(hostname)
        if ip == hostname:
            return endpoint
        resolved = parsed._replace(netloc=f"{ip}:{parsed.port}" if parsed.port else ip)
        resolved_url = urlunparse(resolved)
        logger.info("Resolved CDP endpoint %s -> %s", endpoint, resolved_url)
        return resolved_url
    except socket.gaierror:
        logger.warning("Could not resolve %s, using as-is", hostname)
        return endpoint


class BrowserSession:
    """Workspace-level browser with tab management.

    One browser process, one context (shared cookies), multiple named tabs.
    Agents request tabs by name. Lost tabs are recreated on next use.
    """

    def __init__(self, config: BrowserConfig) -> None:
        self._config = config
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._tabs: dict[str, Any] = {}
        self._console_logs: dict[str, list[str]] = {}
        self._mouse_pos: dict[str, tuple[float, float]] = {}
        self._storage_dir: Path | None = None
        self._launched = False
        self._downloads: dict[str, dict[str, Any]] = {}
        self._download_counter = 0

    def set_storage_dir(self, path: Path) -> None:
        """Set directory for persisting browser state (cookies, localStorage)."""
        self._storage_dir = path

    # ---- Launch ----

    async def _ensure_launched(self) -> None:
        if self._launched and self._context is not None:
            return

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        if self._config.cdp_endpoint:
            await self._connect_cdp()
        elif self._config.ws_endpoint:
            await self._connect_ws()
        else:
            await self._launch_local()

        self._launched = True

    async def _connect_cdp(self) -> None:
        """Connect to a browser via Chrome DevTools Protocol.

        Reuses the browser's existing context if one exists (preserving
        cookies, tabs, and login state). After connection, call
        _adopt_existing_pages() to recover open tabs into _tabs.

        For resilient deployments, the host-side process that runs the
        browser should auto-relaunch on crash/exit. The agent's
        reconnect() will then reconnect to the newly launched browser
        transparently, with login state preserved via user_data_dir.
        """
        endpoint = _resolve_cdp_endpoint(self._config.cdp_endpoint)
        self._browser = await self._playwright.chromium.connect_over_cdp(
            endpoint,
        )
        if self._browser.contexts:
            self._context = self._browser.contexts[0]
        else:
            self._context = await self._browser.new_context(viewport={
                "width": self._config.viewport_width,
                "height": self._config.viewport_height,
            })
        logger.info("Connected to remote browser via CDP: %s", self._config.cdp_endpoint)

    async def _connect_ws(self) -> None:
        """Connect to a Playwright browser server via WebSocket."""
        launcher = getattr(self._playwright, self._config.browser_type)
        self._browser = await launcher.connect(self._config.ws_endpoint)
        self._context = await self._browser.new_context(viewport={
            "width": self._config.viewport_width,
            "height": self._config.viewport_height,
        })
        logger.info("Connected to remote browser via WebSocket: %s", self._config.ws_endpoint)

    async def _launch_local(self) -> None:
        """Launch a local browser process.

        Auto-detects Linux/root/container environments and adds required
        flags (--no-sandbox, --disable-dev-shm-usage) for Playwright's
        bundled Chromium. No need for system Chrome.
        """
        launcher = getattr(self._playwright, self._config.browser_type)

        viewport = {
            "width": self._config.viewport_width,
            "height": self._config.viewport_height,
        }

        common_kwargs: dict[str, Any] = {"headless": self._config.headless}
        if self._config.channel:
            common_kwargs["channel"] = self._config.channel
        if self._config.proxy:
            common_kwargs["proxy"] = self._config.proxy

        args = list(self._config.extra_args or [])
        ignore_defaults = list(self._config.ignore_default_args or [])

        is_chromium = self._config.browser_type == "chromium"

        if is_chromium:
            import os
            import sys
            from arion_agent.util.runtime import is_container
            is_linux = sys.platform.startswith("linux")
            is_root = is_linux and os.getuid() == 0

            if is_root or is_container():
                if "--no-sandbox" not in args:
                    args.append("--no-sandbox")
            if is_linux:
                if "--disable-dev-shm-usage" not in args:
                    args.append("--disable-dev-shm-usage")
                if self._config.headless and "--disable-gpu" not in args:
                    args.append("--disable-gpu")

        if self._config.stealth and is_chromium:
            args.append("--disable-blink-features=AutomationControlled")
            if "--enable-automation" not in ignore_defaults:
                ignore_defaults.append("--enable-automation")

        if args:
            common_kwargs["args"] = args
        if ignore_defaults:
            common_kwargs["ignore_default_args"] = ignore_defaults

        stealth_context: dict[str, Any] = {}
        if self._config.stealth and is_chromium:
            stealth_context["user_agent"] = _STEALTH_USER_AGENT
            stealth_context["locale"] = "en-US"

        if self._config.user_data_dir:
            self._context = await launcher.launch_persistent_context(
                self._config.user_data_dir, viewport=viewport,
                **stealth_context, **common_kwargs,
            )
            self._browser = None
        else:
            self._browser = await launcher.launch(**common_kwargs)
            context_kwargs: dict[str, Any] = {"viewport": viewport, **stealth_context}
            storage_path = self._storage_state_path()
            if storage_path and storage_path.exists():
                context_kwargs["storage_state"] = str(storage_path)
                logger.debug("Restored browser state from %s", storage_path)
            self._context = await self._browser.new_context(**context_kwargs)

    # ---- Lifecycle management ----

    def status(self) -> dict[str, Any]:
        """Read-only inspection of browser connection state.

        Does NOT trigger a connection attempt. Safe to call at any time.
        """
        if self._config.cdp_endpoint:
            mode = "cdp"
            endpoint = self._config.cdp_endpoint
        elif self._config.ws_endpoint:
            mode = "ws"
            endpoint = self._config.ws_endpoint
        else:
            mode = "local"
            endpoint = "local"

        if not self._launched or self._context is None:
            mode = "not_connected"

        tabs = []
        for name, page in self._tabs.items():
            try:
                url = page.url
            except Exception:
                url = "(closed)"
            tabs.append({"name": name, "url": url})

        return {
            "connected": self._launched and self._context is not None,
            "mode": mode,
            "endpoint": endpoint,
            "tabs": tabs,
            "headless": self._config.headless,
        }

    async def reconnect(self) -> str:
        """Tear down stale Playwright state and reconnect or relaunch.

        Recovery behavior depends on the connection mode:

        Local mode: launches a new browser process. If user_data_dir is
        set, cookies and login state persist. Tabs start fresh.

        CDP mode: reconnects to the browser at the configured endpoint.
        If the browser is still running (connection blip, container restart),
        _adopt_existing_pages() recovers open tabs -- full revival.
        If the browser was killed and a supervisor relaunched it, connects
        to the new process. Login state persists via user_data_dir on the
        host side. Tabs start fresh.
        If the CDP endpoint is unreachable (browser down, no supervisor),
        returns an error string. The agent should inform the user.

        Returns a human-readable result string (success or failure detail).
        """
        self._tabs.clear()
        self._console_logs.clear()
        self._mouse_pos.clear()

        if not self.is_remote:
            try:
                if self._browser:
                    await self._browser.close()
                elif self._context:
                    await self._context.close()
            except Exception:
                pass

        self._context = None
        self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._playwright = None
        self._launched = False

        try:
            await self._ensure_launched()
            self._adopt_existing_pages()
            mode = "CDP" if self._config.cdp_endpoint else (
                "WebSocket" if self._config.ws_endpoint else "local"
            )
            tab_count = len(self._tabs)
            msg = f"Browser reconnected ({mode})."
            if tab_count > 0:
                msg += f" Adopted {tab_count} existing tab(s)."
            logger.info(msg)
            return msg
        except Exception as exc:
            logger.error("Browser reconnection failed: %s", exc)
            return f"Browser reconnection failed: {exc}"

    def _adopt_existing_pages(self) -> None:
        """Adopt existing pages from the browser context into _tabs.

        Called after reconnect or resurrection. In CDP mode, if the remote
        browser kept running while the agent's connection dropped, the
        context still has open pages with their URLs, DOM state, and
        cookies. This method recovers them into _tabs so the agent can
        continue where it left off rather than starting from blank tabs.

        In local mode or after the browser was killed and relaunched,
        context.pages is typically empty (no-op).
        """
        if self._context is None:
            return
        try:
            existing_pages = self._context.pages
        except Exception:
            return
        if not existing_pages:
            return

        for i, page in enumerate(existing_pages):
            tab_name = f"recovered-{i}"
            self._tabs[tab_name] = page
            self._console_logs[tab_name] = []
            page.on("console", lambda msg, t=tab_name: self._console_logs.get(t, []).append(
                f"[{msg.type}] {msg.text}"
            ))
            page.on("pageerror", lambda exc, t=tab_name: self._console_logs.get(t, []).append(
                f"[error] {exc}"
            ))

        urls = []
        for name, page in self._tabs.items():
            try:
                urls.append(f"{name}: {page.url}")
            except Exception:
                urls.append(f"{name}: (unknown)")
        logger.info("Adopted %d existing tab(s): %s", len(self._tabs), ", ".join(urls))

    # ---- Tab management ----

    async def _get_page(self, tab: str = "default") -> Any:
        """Get or create a named tab. Recreates if the tab was closed.

        If the browser was closed externally (CDP disconnect, user close),
        delegates to reconnect() and then creates the requested tab.
        """
        await self._ensure_launched()

        page = self._tabs.get(tab)
        if page is not None:
            try:
                await page.title()
                return page
            except Exception:
                self._tabs.pop(tab, None)
                self._console_logs.pop(tab, None)
                logger.debug("Tab '%s' was lost, recreating", tab)

        if len(self._tabs) >= MAX_TABS:
            oldest = next(iter(self._tabs))
            await self.close_tab(oldest)

        try:
            page = await self._context.new_page()
        except Exception:
            logger.warning("Browser context lost, attempting resurrection")
            result = await self.reconnect()
            if not self._launched or self._context is None:
                raise RuntimeError(f"Browser resurrection failed: {result}")
            page = await self._context.new_page()

        self._tabs[tab] = page
        self._console_logs[tab] = []

        page.on("console", lambda msg, t=tab: self._console_logs.get(t, []).append(
            f"[{msg.type}] {msg.text}"
        ))
        page.on("pageerror", lambda exc, t=tab: self._console_logs.get(t, []).append(
            f"[error] {exc}"
        ))

        if self._config.stealth:
            try:
                from playwright_stealth import stealth_async
                await stealth_async(page)
            except ImportError:
                pass

        return page

    async def close_tab(self, tab: str) -> str:
        page = self._tabs.pop(tab, None)
        self._console_logs.pop(tab, None)
        self._mouse_pos.pop(tab, None)
        if page is None:
            return f"Tab '{tab}' not found."
        try:
            await page.close()
        except Exception:
            pass
        return f"Tab '{tab}' closed."

    def list_tabs(self) -> str:
        if not self._tabs:
            return "No open tabs."
        parts = []
        for name, page in self._tabs.items():
            try:
                url = page.url
            except Exception:
                url = "(closed)"
            parts.append(f"  {name}: {url}")
        return f"Tabs ({len(self._tabs)}/{MAX_TABS}):\n" + "\n".join(parts)

    def tab_context(self, current_tab: str) -> str:
        """Compact one-line summary of all open tabs for tool result footers."""
        if not self._tabs:
            return "[tabs] (none)"
        parts = []
        for name, page in self._tabs.items():
            try:
                url = page.url
            except Exception:
                url = "(closed)"
            label = f"{name}*" if name == current_tab else name
            parts.append(f"{label}: {url}")
        return "[tabs] " + " | ".join(parts)

    # ---- Operations (all take a tab name) ----

    async def navigate(self, url: str, *, tab: str = "default") -> str:
        page = await self._get_page(tab)
        try:
            if self._config.humanize:
                import random
                await asyncio.sleep(random.uniform(0.3, 1.0) * self._config.humanize_speed)
            current_url = page.url
            kwargs: dict = {"timeout": self._config.timeout_ms}
            if current_url and current_url not in ("", "about:blank"):
                kwargs["referer"] = current_url
            await page.goto(url, **kwargs)
            title = await page.title()
            return f"Navigated to {url} (title: {title})"
        except Exception as exc:
            return f"Navigation failed: {exc}"

    async def snapshot(
        self,
        *,
        tab: str = "default",
        max_chars: int | None = None,
        max_text_len: int = 80,
    ) -> str:
        """Return a simplified HTML snapshot of the current page.

        Walks the rendered DOM and keeps interactive elements (links, buttons,
        inputs), semantic structure (nav, headings, sections, tables), and
        elements with meaningful attributes (id, role, aria-*, data-*).
        Strips scripts, styles, SVGs, and hidden elements.

        The output is compact enough for LLM consumption while preserving
        the attributes needed to construct CSS selectors for browser_action.
        """
        page = await self._get_page(tab)
        try:
            url = page.url
            title = await page.title()
            budget = max_chars if max_chars is not None else SNAPSHOT_MAX_CHARS
            js = (_SIMPLIFY_DOM_JS
                  .replace("%%MAX_CHARS%%", str(budget))
                  .replace("%%MAX_TEXT%%", str(max_text_len)))
            html = await page.evaluate(js)
            if not html or not html.strip():
                html = "(empty page)"
            return f"[page] url: {url}\n[page] title: {title}\n{html}"
        except Exception as exc:
            return f"Snapshot failed: {exc}"

    async def screenshot(self, *, tab: str = "default", save_dir: Path | None = None) -> str:
        page = await self._get_page(tab)
        buf = await page.screenshot(
            type="jpeg",
            quality=self._config.screenshot_quality,
            full_page=False,
        )

        from datetime import UTC, datetime
        from arion_agent.util.persistence import ensure_directory
        from arion_agent.util.multimodal import IMAGE_BLOCK_SENTINEL

        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        if save_dir is None:
            import tempfile
            save_dir = Path(tempfile.gettempdir()) / "arion_screenshots"
        ensure_directory(save_dir)
        saved_path = save_dir / f"screenshot-{ts}.jpg"
        saved_path.write_bytes(buf)
        file_uri = saved_path.resolve().as_uri()
        return f"{IMAGE_BLOCK_SENTINEL}:image/jpeg:{file_uri}"

    async def action(self, action_name: str, target: str = "", value: str = "", *, tab: str = "default") -> str:
        page = await self._get_page(tab)

        delay = 0
        if self._config.humanize:
            import random
            delay = int(80 * self._config.humanize_speed + random.gauss(0, 30))
            delay = max(20, delay)

        try:
            if action_name == "navigate":
                return await self.navigate(target or value, tab=tab)

            elif action_name == "click":
                if self._config.humanize:
                    import random
                    from arion_agent.environments.browser.humanize import human_curve
                    speed = self._config.humanize_speed
                    try:
                        box = await page.locator(target).bounding_box(timeout=self._config.timeout_ms)
                        if box:
                            cx = box["x"] + box["width"] / 2 + random.gauss(0, 2)
                            cy = box["y"] + box["height"] / 2 + random.gauss(0, 2)
                            origin = self._mouse_pos.get(tab, (
                                self._config.viewport_width / 2,
                                self._config.viewport_height / 2,
                            ))
                            waypoints = human_curve(origin, (cx, cy))
                            for wx, wy in waypoints:
                                await page.mouse.move(wx, wy)
                                await asyncio.sleep(random.uniform(0.005, 0.02) * speed)
                            self._mouse_pos[tab] = (cx, cy)
                            await asyncio.sleep(random.uniform(0.1, 0.4) * speed)
                            await page.mouse.click(cx, cy)
                        else:
                            await asyncio.sleep(random.uniform(0.1, 0.3) * speed)
                            await page.click(target, timeout=self._config.timeout_ms)
                    except Exception:
                        await asyncio.sleep(random.uniform(0.1, 0.3) * speed)
                        await page.click(target, timeout=self._config.timeout_ms)
                    await asyncio.sleep(random.uniform(0.05, 0.2) * speed)
                else:
                    await page.click(target, timeout=self._config.timeout_ms)
                return f"Clicked: {target}"

            elif action_name == "fill":
                await page.fill(target, value, timeout=self._config.timeout_ms)
                return f"Filled '{target}' with '{value[:50]}...'" if len(value) > 50 else f"Filled '{target}' with '{value}'"

            elif action_name == "type":
                await page.type(target, value, delay=delay, timeout=self._config.timeout_ms)
                return f"Typed into '{target}'"

            elif action_name == "select":
                await page.select_option(target, value, timeout=self._config.timeout_ms)
                return f"Selected '{value}' in '{target}'"

            elif action_name == "scroll":
                direction = value.lower() if value else "down"
                if self._config.humanize:
                    import random
                    speed = self._config.humanize_speed
                    vw = self._config.viewport_width
                    vh = self._config.viewport_height

                    if tab not in self._mouse_pos:
                        mx = vw * random.uniform(0.3, 0.7)
                        my = vh * random.uniform(0.3, 0.6)
                        await page.mouse.move(mx, my)
                        self._mouse_pos[tab] = (mx, my)

                    steps = random.randint(2, 4)
                    for _ in range(steps):
                        delta = random.randint(80, 200) * (1 if direction != "up" else -1)
                        await page.mouse.wheel(0, delta)
                        mx, my = self._mouse_pos[tab]
                        mx += random.gauss(0, 2)
                        my += random.gauss(0, 1.5)
                        mx = max(10, min(vw - 10, mx))
                        my = max(10, min(vh - 10, my))
                        await page.mouse.move(mx, my)
                        self._mouse_pos[tab] = (mx, my)
                        await asyncio.sleep(random.uniform(0.03, 0.12) * speed)
                else:
                    delta = 300 if direction != "up" else -300
                    await page.mouse.wheel(0, delta)
                return f"Scrolled {direction}"

            elif action_name == "wait":
                if target:
                    await page.wait_for_selector(target, timeout=self._config.timeout_ms)
                    return f"Element appeared: {target}"
                else:
                    secs = float(value) if value else 2.0
                    await asyncio.sleep(secs)
                    return f"Waited {secs}s"

            else:
                return f"Unknown action: {action_name}. Use: navigate, click, fill, type, select, scroll, wait."

        except Exception as exc:
            return f"Action '{action_name}' failed: {exc}"

    async def wait_for_human(self, message: str, wait_for_selector: str = "", *, tab: str = "default", timeout: float = 180) -> str:
        page = await self._get_page(tab)
        print(f"\n  [BROWSER - HUMAN ACTION REQUIRED] {message}", flush=True)

        if wait_for_selector:
            print(f"  Waiting for element: {wait_for_selector} (timeout {timeout}s)", flush=True)
            try:
                await page.wait_for_selector(wait_for_selector, timeout=timeout * 1000)
                return f"Human completed action. Element '{wait_for_selector}' appeared."
            except Exception:
                return "Timeout waiting for human action."
        else:
            print(f"  Press Enter when done (timeout {timeout}s)...", flush=True)
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, input),
                    timeout=timeout,
                )
                return "Human signaled completion."
            except asyncio.TimeoutError:
                return "Timeout waiting for human action."

    async def console(self, clear: bool = False, *, tab: str = "default") -> str:
        await self._get_page(tab)
        logs = self._console_logs.get(tab, [])
        if not logs:
            return "(no console output)"
        output = "\n".join(logs[-100:])
        total = len(logs)
        if clear:
            logs.clear()
        if total > 100:
            output = f"[showing last 100 of {total} entries]\n" + output
        return output

    async def evaluate_js(self, expression: str, *, tab: str = "default") -> str:
        page = await self._get_page(tab)
        try:
            result = await page.evaluate(expression)
            return str(result)
        except Exception as exc:
            return f"JS evaluation failed: {exc}"

    async def create_tab(self, name: str, url: str = "") -> str:
        """Create a named tab, optionally navigating to a URL.

        If a tab with this name already exists, returns it. If the tab limit
        is reached, the oldest tab is auto-closed.
        """
        await self._get_page(name)
        if url:
            return await self.navigate(url, tab=name)
        return f"Tab '{name}' created."

    async def save_page(
        self,
        save_path: Path | str,
        *,
        tab: str = "default",
        selector: str = "",
        wait_for: str = "",
        format: str = "text",
    ) -> str:
        """Save current page content to a file.

        Args:
            save_path: Where to save the file.
            tab: Tab to capture from.
            selector: CSS selector to scope content (default: full page body).
            wait_for: CSS selector to wait for before capturing.
            format: 'text' for plain text, 'html' for rendered HTML.
        """
        page = await self._get_page(tab)

        if wait_for:
            try:
                await page.wait_for_selector(wait_for, timeout=self._config.timeout_ms)
            except Exception as exc:
                return f"Wait failed for '{wait_for}': {exc}"

        try:
            if format == "html":
                if selector:
                    content = await page.locator(selector).first.inner_html()
                else:
                    content = await page.content()
            else:
                target = selector or "body"
                content = await page.locator(target).first.text_content() or ""

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(content, encoding="utf-8")

            size = save_path.stat().st_size
            title = await page.title()
            url = page.url
            return f"Saved {format} ({size:,} bytes) to {save_path}\nSource: {url} ({title})"
        except Exception as exc:
            return f"Save failed: {exc}"

    async def download(
        self,
        source: str,
        save_path: Path,
        *,
        tab: str = "default",
    ) -> str:
        """Fire a non-blocking download and return immediately.

        *source* is a URL (starts with http) or a CSS selector to click.
        The download runs in a background task. Use download_list() to
        check progress.  If *save_path* has no suffix, the browser's
        suggested filename is appended automatically.
        """
        page = await self._get_page(tab)
        self._download_counter += 1
        dl_id = f"dl_{self._download_counter}"
        is_url = source.startswith("http://") or source.startswith("https://")

        info: dict[str, Any] = {
            "id": dl_id,
            "source": source[:120],
            "filename": None,
            "status": "starting",
            "path": str(save_path),
            "error": None,
            "tab": tab,
        }
        self._downloads[dl_id] = info

        asyncio.create_task(
            self._download_task(dl_id, page, source, save_path, is_url)
        )
        return f"Download {dl_id} started. Use browser_download_list to check progress."

    async def _download_task(
        self,
        dl_id: str,
        page: Any,
        source: str,
        save_path: Path,
        is_url: bool,
    ) -> None:
        info = self._downloads[dl_id]
        try:
            async with page.expect_download(timeout=120_000) as dl_info:
                if is_url:
                    try:
                        await page.goto(source, timeout=self._config.timeout_ms)
                    except Exception:
                        pass
                else:
                    await page.click(source, timeout=self._config.timeout_ms)
            dl = await dl_info.value
            suggested = dl.suggested_filename
            info["filename"] = suggested
            info["status"] = "downloading"

            if not save_path.suffix:
                save_path = save_path / suggested
            save_path.parent.mkdir(parents=True, exist_ok=True)
            await dl.save_as(str(save_path))

            size = save_path.stat().st_size
            info["status"] = "complete"
            info["path"] = str(save_path)
            info["size"] = size
            logger.info("Download %s complete: %s (%s bytes)", dl_id, save_path, size)
        except Exception as exc:
            info["status"] = "failed"
            info["error"] = str(exc)
            logger.warning("Download %s failed: %s", dl_id, exc)

    def download_list(self) -> str:
        """Return status of all tracked downloads."""
        if not self._downloads:
            return "No downloads."
        lines = []
        for dl_id, info in self._downloads.items():
            status = info["status"]
            name = info["filename"] or "(pending)"
            if status == "complete":
                lines.append(f"  {dl_id} [complete]: {name} -> {info['path']} ({info.get('size', '?'):,} bytes)")
            elif status == "failed":
                lines.append(f"  {dl_id} [failed]: {name} - {info['error']}")
            else:
                lines.append(f"  {dl_id} [{status}]: {name}")
        return f"Downloads ({len(self._downloads)}):\n" + "\n".join(lines)

    async def read(self, selector: str = "body", *, tab: str = "default") -> str:
        page = await self._get_page(tab)
        try:
            text = await page.text_content(selector, timeout=self._config.timeout_ms)
            if text and len(text) > 5000:
                text = text[:5000] + f"\n[...truncated, {len(text)} chars total]"
            return text or "(no text content)"
        except Exception as exc:
            return f"Read failed: {exc}"

    # ---- Lifecycle ----

    def _storage_state_path(self) -> Path | None:
        if self._storage_dir is None:
            return None
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        return self._storage_dir / "browser_state.json"

    @property
    def is_remote(self) -> bool:
        return self._config.cdp_endpoint is not None or self._config.ws_endpoint is not None

    async def close(self) -> str:
        is_remote = self.is_remote

        if not is_remote and self._context and not self._config.user_data_dir:
            storage_path = self._storage_state_path()
            if storage_path:
                try:
                    await self._context.storage_state(path=str(storage_path))
                    logger.debug("Saved browser state to %s", storage_path)
                except Exception:
                    logger.debug("Failed to save browser state", exc_info=True)

        for tab_name in list(self._tabs):
            await self.close_tab(tab_name)
        self._console_logs.clear()

        if is_remote:
            pass
        elif self._browser:
            await self._browser.close()
        elif self._context:
            await self._context.close()

        if self._playwright:
            await self._playwright.stop()
        self._context = None
        self._browser = None
        self._playwright = None
        self._launched = False
        return "Browser disconnected." if is_remote else "Browser closed."

    @property
    def is_open(self) -> bool:
        return self._launched and self._context is not None

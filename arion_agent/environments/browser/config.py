"""Browser environment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BrowserConfig:
    """Configuration for the browser environment.

    Three launch modes:

    1. Local (default): launches a local browser process.
       BrowserConfig()
       BrowserConfig(channel="chrome", user_data_dir="/path/to/profile")

    2. CDP: connects to a remote Chrome via DevTools Protocol.
       BrowserConfig(cdp_endpoint="http://host:9222")

    3. WebSocket: connects to a Playwright browser server.
       BrowserConfig(ws_endpoint="ws://host:3000/playwright")

    Modes 2 and 3 decouple the agent from the browser. Cookies and
    login state live on the browser host side.

    Resilience model:
    - Local mode: reconnect() launches a new browser process if the old
      one died. Set user_data_dir to persist cookies across relaunches.
    - CDP mode: reconnect() reconnects to the same browser if it is still
      running (reviving open tabs), or to a newly launched browser if a
      supervisor relaunched it. The agent cannot launch the remote browser
      itself -- a host-side supervisor should auto-relaunch on crash/exit
      to keep the CDP endpoint available.
    - In both modes, browser_status checks connection state without side
      effects, and browser_close disconnects the agent without killing
      the remote browser process (CDP) or saves state before closing (local).
    """

    cdp_endpoint: str | None = None
    ws_endpoint: str | None = None

    headless: bool = False
    browser_type: str = "chromium"
    channel: str | None = None
    viewport_width: int = 1280
    viewport_height: int = 720
    user_data_dir: str | None = None
    extra_args: list[str] | None = None
    ignore_default_args: list[str] | None = None
    timeout_ms: int = 30000
    screenshot_quality: int = 50
    stealth: bool = True
    humanize: bool = True
    humanize_speed: float = 1.0
    proxy: dict[str, str] | None = None

"""Browser environment: optional persistent browser + HTTP tools.

Degrades gracefully if playwright is not installed. Importing this module
never crashes — the config class is always available. BrowserEnvironment
and BrowserSession raise ImportError only when instantiated without
playwright, not at import time.
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

from arion_agent.environments.browser.config import BrowserConfig  # noqa: E402

try:
    from arion_agent.environments.browser.middleware import BrowserEnvironment
    from arion_agent.environments.browser.session import BrowserSession
    from arion_agent.environments.browser.skills import get_browser_skill_names

    _BROWSER_AVAILABLE = True
except ImportError:
    _BROWSER_AVAILABLE = False
    _logger.debug(
        "Browser environment unavailable: playwright not installed. "
        "Install with: pip install arion-agent[browser] && playwright install chromium. "
        "On Linux, also run: playwright install-deps (or playwright install --with-deps chromium)"
    )

    BrowserEnvironment = None  # type: ignore[assignment,misc]
    BrowserSession = None  # type: ignore[assignment,misc]

    def get_browser_skill_names() -> dict[str, list[str]]:  # type: ignore[misc]
        return {"important": [], "generic": []}


def is_browser_available() -> bool:
    """Check whether playwright is installed and browser tools can be used."""
    return _BROWSER_AVAILABLE


__all__ = [
    "BrowserConfig",
    "BrowserEnvironment",
    "BrowserSession",
    "get_browser_skill_names",
    "is_browser_available",
]

"""Cross-cutting timezone utility for consistent time handling across the agent."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


class AgentClock:
    """Provides timezone-aware time operations for the entire agent.

    Created once per agent and shared across middleware. All heartbeat
    scheduling, identity timestamps, and signal timestamps should use
    this clock for consistency.
    """

    def __init__(self, tz: str = "UTC") -> None:
        self._tz_name = tz
        self._tz = ZoneInfo(tz)

    @property
    def timezone_name(self) -> str:
        return self._tz_name

    @property
    def tz(self) -> ZoneInfo:
        return self._tz

    def now(self) -> datetime:
        """Current time in the agent's configured timezone."""
        return datetime.now(self._tz)

    def now_utc(self) -> datetime:
        """Current time in UTC."""
        return datetime.now(timezone.utc)

    def format_iso(self, dt: datetime | None = None) -> str:
        """Format datetime as ISO 8601 with timezone offset."""
        d = dt or self.now()
        return d.isoformat()

    def format_human(self, dt: datetime | None = None) -> str:
        """Format datetime in human-readable form (e.g., 'Monday, February 24, 2026 2:30 PM EST')."""
        d = dt or self.now()
        return d.strftime("%A, %B %d, %Y %I:%M %p %Z")

    def parse(self, iso_str: str) -> datetime:
        """Parse an ISO 8601 string into a timezone-aware datetime in the agent's tz.

        If the string has no timezone info, assumes the agent's timezone.
        """
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._tz)
        return dt.astimezone(self._tz)

    def format_duration(self, seconds: float) -> str:
        """Format a duration in seconds into human-readable form."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        minutes = seconds / 60
        if minutes < 60:
            return f"{int(minutes)} minutes"
        hours = minutes / 60
        if hours < 24:
            h = int(hours)
            m = int(minutes % 60)
            return f"{h} hours {m} minutes" if m else f"{h} hours"
        days = int(hours / 24)
        h = int(hours % 24)
        return f"{days} days {h} hours" if h else f"{days} days"

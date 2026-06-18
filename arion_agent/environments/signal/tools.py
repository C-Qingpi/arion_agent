"""Signal environment tools: signal_send and signal_check."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from arion_agent.environments.signal.store import SignalStore


def create_signal_tools(
    agent_id: str,
    store: "SignalStore",
) -> list[BaseTool]:
    """Create signal_send and signal_check tools bound to a SignalStore."""

    @tool
    async def signal_send(channel: str, signal_type: str, content: str) -> str:
        """Post a structured signal to a channel.

        Signals are append-only messages for coordination, status updates,
        or communication with external systems and other agents.

        Args:
            channel: Channel name to post to (e.g. "default", "status").
            signal_type: Label for this signal (e.g. "info", "request",
                "approval", "error", "stop"). Not enforced by system.
            content: The message payload.

        Returns:
            Confirmation with the signal id.
        """
        signal = store.send(channel, agent_id, signal_type, content)
        return f"Signal posted: [{signal['id']}] {signal['timestamp']} | {channel} | {signal_type}"

    @tool
    async def signal_check(channel: str = "default", last_n: int = 10) -> str:
        """Read recent signals from a channel.

        Args:
            channel: Channel name to read from. Defaults to "default".
            last_n: Number of recent signals to show. Defaults to 10.

        Returns:
            Formatted list of recent signals, or a message if none found.
        """
        signals = store.check(channel, last_n)
        if not signals:
            return f"No signals in channel '{channel}'."

        lines = []
        for s in signals:
            lines.append(
                f"[{s['id']}] {s['timestamp']} | {s['sender']} | {s['type']}"
            )
            lines.append(s["content"])
            lines.append("")
        return "\n".join(lines).rstrip()

    return [signal_send, signal_check]

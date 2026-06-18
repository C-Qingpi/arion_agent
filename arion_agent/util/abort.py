"""Abort polling for long-running tools (wait, etc.)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


async def interruptible_sleep(
    seconds: float,
    abort_check: Callable[[], bool] | None,
    *,
    poll_interval: float = 0.25,
) -> None:
    """Sleep in short chunks, raising AgentAborted when abort_check returns True."""
    if seconds <= 0:
        return

    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        if abort_check is not None and abort_check():
            from arion_agent.graph import AgentAborted

            raise AgentAborted("Aborted during wait")
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(poll_interval, remaining))

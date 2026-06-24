"""Optional checkpoint cleanup for long-lived agents.

The checkpoint chain grows as the agent runs. This module provides opt-in
utilities for pruning old checkpoints. Not imported by default; developers
who need it import explicitly.

Usage:
    from arion_agent.util.checkpoint_cleanup import prune_checkpoints
    await prune_checkpoints(graph, thread_id, keep_last=100)

Or configure via create_arion_agent:
    create_arion_agent(
        ...,
        checkpoint_cleanup=CheckpointCleanupConfig(keep_last=100),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointCleanupConfig:
    """Configuration for optional checkpoint pruning.

    keep_last: Number of most recent checkpoints to retain per thread.
        Set to 0 or None to disable cleanup.
    run_on_heartbeat: If True, prune runs automatically on heartbeat ticks.
    """
    keep_last: int = 450
    run_on_heartbeat: bool = False


async def prune_checkpoints(
    graph: Any,
    thread_id: str,
    *,
    keep_last: int = 450,
) -> int:
    """Remove old checkpoints for a thread, keeping the most recent N.

    Returns the number of checkpoints removed, or 0 if pruning is not
    supported by the checkpointer.
    """
    if keep_last <= 0:
        return 0

    try:
        config = {"configurable": {"thread_id": thread_id}}
        history = list(graph.get_state_history(config))
    except Exception:
        logger.debug("Failed to read checkpoint history for pruning", exc_info=True)
        return 0

    if len(history) <= keep_last:
        return 0

    to_remove = history[keep_last:]
    removed = 0

    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        logger.debug("No checkpointer found on graph, skipping prune")
        return 0

    for snapshot in to_remove:
        checkpoint_config = snapshot.config
        try:
            if hasattr(checkpointer, "adelete"):
                await checkpointer.adelete(checkpoint_config)
                removed += 1
            elif hasattr(checkpointer, "delete"):
                checkpointer.delete(checkpoint_config)
                removed += 1
        except Exception:
            logger.debug("Failed to delete checkpoint %s", checkpoint_config, exc_info=True)

    if removed > 0:
        logger.info(
            "Pruned %d checkpoints for thread '%s' (kept %d)",
            removed, thread_id, keep_last,
        )

    return removed

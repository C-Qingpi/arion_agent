"""Checkpoint-based pagination for conversation history.

Uses checkpoint compression boundaries as natural pagination anchors.
Each pre-compression checkpoint contains the full history window for that era.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AnyMessage

logger = logging.getLogger(__name__)


@dataclass
class CheckpointPage:
    """One page of conversation history, corresponding to one compression window."""
    checkpoint_id: str
    messages: list[AnyMessage]
    message_count: int
    has_summary: bool


@dataclass
class CheckpointHistory:
    """Paginated conversation history from checkpoints."""
    pages: list[CheckpointPage] = field(default_factory=list)
    has_older: bool = False
    next_before_checkpoint_id: str | None = None


async def get_checkpoint_history(
    graph: Any,
    thread_id: str,
    *,
    num_pages: int = 3,
) -> CheckpointHistory:
    """Return paginated conversation history from checkpoint snapshots.

    Each page corresponds to one compression window. Adjacent checkpoints
    within the same window are collapsed; only compression boundaries
    create new pages.

    Args:
        graph: A compiled langgraph StateGraph (must have checkpointer).
        thread_id: Thread identifier.
        num_pages: Number of pages (compression windows) to return.

    Returns:
        CheckpointHistory with pages, has_older flag, and cursor.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        logger.warning("No checkpointer on graph, returning empty history")
        return CheckpointHistory()

    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshots = list(graph.get_state_history(config))
    except Exception:
        logger.debug("Failed to read checkpoint history", exc_info=True)
        return CheckpointHistory()

    if not snapshots:
        return CheckpointHistory()

    # snapshots are most-recent-first
    pages: list[CheckpointPage] = []
    prev_msg_count = 0
    seen = 0

    for snapshot in snapshots:
        vals = (
            snapshot.values
            if hasattr(snapshot, "values")
            else snapshot.checkpoint.get("channel_values", {})
        )
        msgs: list[AnyMessage] = list(vals.get("messages", []))
        summary = vals.get("summary", "")
        cid = snapshot.config.get("configurable", {}).get("checkpoint_id", "")

        seen += 1
        msg_count = len(msgs)

        # Compression boundary: going backward in time, message count
        # jumps UP at the pre-compression checkpoint.
        is_boundary = (
            prev_msg_count > 0
            and msg_count > prev_msg_count * 1.3
            and msg_count > 20
        )

        if is_boundary or seen == 1:
            pages.append(CheckpointPage(
                checkpoint_id=cid,
                messages=msgs,
                message_count=msg_count,
                has_summary=bool(summary),
            ))
            if len(pages) >= num_pages:
                break

        prev_msg_count = msg_count

    has_older = seen < len(snapshots)
    next_before_checkpoint_id = None

    if has_older and pages:
        oldest_cid = pages[-1].checkpoint_id
        found = False
        for snapshot in snapshots:
            cid = snapshot.config.get("configurable", {}).get("checkpoint_id", "")
            if cid == oldest_cid:
                found = True
                continue
            if found:
                vals = (
                    snapshot.values
                    if hasattr(snapshot, "values")
                    else snapshot.checkpoint.get("channel_values", {})
                )
                if vals.get("summary", ""):
                    next_before_checkpoint_id = cid
                    break

    return CheckpointHistory(
        pages=pages,
        has_older=has_older,
        next_before_checkpoint_id=next_before_checkpoint_id,
    )

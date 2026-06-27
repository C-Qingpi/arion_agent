"""Agentic core tools: tools that affect the agent's reasoning loop and lifecycle.

These are tools core to agent intelligence, not external environment interaction.
  - maintenance_tool: echo/test (diagnostic)
  - lookup_user_prompts: search past user messages from conversation history
  - update_plan: structured work planning with programmatic enforcement
  - get_running_status: session metrics awareness
  - The task tool (subagent spawning) is contributed by SubagentMiddleware
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from arion_agent.util.persistence import (
    file_exists,
    glob_files,
    load_jsonl,
    workspace_relative_path,
)

if TYPE_CHECKING:
    from arion_agent.environments.agentic_core.plan_registry import PlanRegistry
    from arion_agent.util.timezone import AgentClock

logger = logging.getLogger(__name__)


@tool
async def maintenance_tool(
    message: str,
    delay_seconds: float = 0,
    mode: str = "echo",
) -> str:
    """Echo, error, or hang for testing. Modes: echo, error, hang, partial.

    Args:
        message: The message to echo back.
        delay_seconds: Seconds to wait before responding. Defaults to 0.
        mode: Behavior mode.
            echo (default) - echo message after delay.
            error - raise a RuntimeError with message as the error text.
            hang - sleep forever (will be killed by timeout).
            partial - print partial output then raise an error.

    Returns:
        The echoed message with timing info (echo/partial modes).
    """
    if mode == "error":
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        raise RuntimeError(f"Simulated tool failure: {message}")

    if mode == "hang":
        await asyncio.sleep(999999)
        return "unreachable"

    if mode == "partial":
        partial = f"[maintenance_tool] partial output before crash: {message}"
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        raise RuntimeError(f"Crashed after partial work. Partial: {partial}")

    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)
    return f"[maintenance_tool] echo after {delay_seconds}s: {message}"


def _create_lookup_user_prompts(identity_dir: Path) -> "BaseTool":
    """Build a lookup_user_prompts tool bound to the agent's identity directory."""

    @tool
    async def lookup_user_prompts(
        thread_id: str,
        page: int = 1,
        count: int = 20,
        since_days: int = 2,
        regex: str = ".*",
    ) -> str:
        """Search past user prompts from conversation history archives.

        Reads JSONL transcript files written by the compaction system.
        Returns user prompts matching the filters.

        Args:
            thread_id: Thread/subagent identifier (uses "default" for main thread).
            page: Page number for pagination (1-indexed, default 1).
            count: Results per page (default 20).
            since_days: Only consider transcripts from the last N days (default 2).
            regex: Filter prompts by regex on content (default ".*" = all).

        Returns:
            Paginated matching user prompts with timestamps and event numbers.
            If no history exists, explains that compaction may not be enabled.
        """
        history_dir = identity_dir / "conversation_history" / thread_id
        if not file_exists(history_dir):
            return (
                "No conversation history directory found for this thread. "
                "Compaction may not be enabled, or no compression events "
                "have occurred yet."
            )

        try:
            pattern_re = re.compile(regex)
        except re.error as exc:
            return f"Error: invalid regex pattern -- {exc}"

        cutoff = datetime.now(UTC) - timedelta(days=since_days)

        jsonl_files = glob_files(history_dir, "*.jsonl")
        if not jsonl_files:
            return (
                "No JSONL transcript files found in conversation history "
                "for this thread. No compression events since JSONL support "
                "was added, or compaction is not enabled."
            )

        matched: list[dict] = []
        for fpath in jsonl_files:
            try:
                records = load_jsonl(fpath)
            except (OSError, json.JSONDecodeError):
                logger.debug("Failed to read JSONL transcript: %s", fpath)
                continue

            for record in records:
                ts_utc = record.get("ts_utc", "")
                if not ts_utc:
                    continue
                try:
                    rec_ts = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                if rec_ts < cutoff:
                    continue

                event = record.get("event", "?")
                participants = record.get("participants", [])
                for p in participants:
                    if p.get("role") != "human":
                        continue
                    content = p.get("content", "").strip()
                    if not content:
                        continue
                    if pattern_re.search(content):
                        short = content[:500] + ("..." if len(content) > 500 else "")
                        matched.append({
                            "ts": ts_utc,
                            "event": event,
                            "file": fpath.name,
                            "content": short,
                        })

        if not matched:
            return (
                f"No user prompts matched (since_days={since_days}, "
                f"regex={regex!r}). "
                f"Searched {len(jsonl_files)} transcript files."
            )

        total = len(matched)
        start = (page - 1) * count
        end = start + count
        page_items = matched[start:end]

        lines = [
            f"User prompts matching since_days={since_days}, regex={regex!r}",
            f"Total matches: {total} | Page {page}/{max(1, (total - 1) // count + 1)} | Showing {len(page_items)}",
            "",
        ]
        for item in page_items:
            lines.append(f"[Event {item['event']}] [{item['ts']}] ({item['file']})")
            lines.append(f"  {item['content']}")
            lines.append("")
        return "\n".join(lines)

    return lookup_user_prompts


def _create_update_plan(
    plan_registry: "PlanRegistry",
    workspace_dir: Path,
    plan_config: "PlanConfig",
) -> "BaseTool":
    """Build an update_plan tool closure bound to a PlanRegistry."""
    from arion_agent.environments.agentic_core.config import PlanConfig  # noqa: F811

    @tool(description=plan_config.effective_tool_description())
    async def update_plan(plan: str) -> str:
        """Update the structured work plan.

        Args:
            plan: JSON object with plan sections (deliverables, methodology,
                context, items, confirmation). A bare JSON array is also
                accepted as items-only shorthand.
        """
        try:
            parsed = json.loads(plan)
        except (ValueError, TypeError) as exc:
            return f"Error: invalid JSON -- {exc}"

        if isinstance(parsed, dict):
            validated = plan_registry.set_plan(parsed)
        elif isinstance(parsed, list):
            validated = plan_registry.set_items(parsed)
        else:
            return "Error: plan must be a JSON object or array."

        persist_path = plan_registry.get_persist_path()
        if persist_path is not None:
            from arion_agent.util.persistence import write_file as persistence_write

            persistence_write(persist_path, plan_registry.to_json())
            rel = workspace_relative_path(persist_path, workspace_dir)
            logger.debug("update_plan: wrote %d items to %s", len(validated), rel)

        return f"Plan updated ({len(validated)} items). {plan_registry.pending_summary()}"

    return update_plan


def _create_get_running_status(stats: "AgentStats", clock: "AgentClock | None" = None) -> "BaseTool":
    """Build a get_running_status tool closure bound to an AgentStats instance."""

    @tool
    async def get_running_status() -> str:
        """Get agent session status: turns, tokens, elapsed time.

        Returns current session metrics for workload awareness.
        """
        if clock is not None:
            now = clock.format_iso()
            tz_label = clock.timezone_name
        else:
            now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            tz_label = "UTC"
        lines = [
            "Session Status",
            f"Model calls: {stats.model_calls}",
            f"Tool calls: {stats.tool_calls}",
            f"Messages in context: {stats.total_messages}",
            f"Tokens (estimated): input={stats.input_tokens_estimated}, output={stats.output_tokens_estimated}",
            f"Elapsed: {stats.elapsed_seconds:.1f}s",
            f"Current time: {now} ({tz_label})",
        ]
        return "\n".join(lines)

    return get_running_status


def create_agentic_core_tools(
    agent_id: str | None = None,
    workspace_dir: Path | None = None,
    stats: "AgentStats | None" = None,
    plan_config: "PlanConfig | None" = None,
    plan_registry: "PlanRegistry | None" = None,
    enable_status: bool = False,
    clock: "AgentClock | None" = None,
    identity_dir: Path | None = None,
) -> list:
    """Create all agentic core tools.

    Args:
        agent_id: Agent identifier (needed for update_plan).
        workspace_dir: Workspace root (needed for update_plan).
        stats: AgentStats instance (needed for get_running_status).
        plan_config: PlanConfig for planning tool. None = planning disabled
            (when called without parameters for backward compatibility).
        plan_registry: PlanRegistry instance for structured plan storage.
        enable_status: Whether to include get_running_status tool.
        clock: AgentClock for timezone-aware timestamps. None = UTC fallback.
        identity_dir: Agent identity directory (needed for lookup_user_prompts).
    """
    tools = [maintenance_tool]

    if identity_dir is not None:
        tools.append(_create_lookup_user_prompts(identity_dir))

    if (
        plan_config is not None
        and plan_registry is not None
        and workspace_dir is not None
    ):
        tools.append(_create_update_plan(plan_registry, workspace_dir, plan_config))

    if enable_status and stats is not None:
        tools.append(_create_get_running_status(stats, clock=clock))

    return tools

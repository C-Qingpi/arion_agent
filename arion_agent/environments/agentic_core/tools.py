"""Agentic core tools: tools that affect the agent's reasoning loop and lifecycle.

These are tools core to agent intelligence, not external environment interaction.
  - maintenance_tool: echo/test (diagnostic)
  - update_plan: structured work planning with programmatic enforcement
  - get_running_status: session metrics awareness
  - The task tool (subagent spawning) is contributed by SubagentMiddleware
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from arion_agent.util.persistence import workspace_relative_path

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
    """
    tools = [maintenance_tool]

    if (
        plan_config is not None
        and plan_registry is not None
        and workspace_dir is not None
    ):
        tools.append(_create_update_plan(plan_registry, workspace_dir, plan_config))

    if enable_status and stats is not None:
        tools.append(_create_get_running_status(stats, clock=clock))

    return tools

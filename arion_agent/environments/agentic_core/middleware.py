"""Agentic core middleware: contributes agent reasoning/lifecycle tools.

Provides:
  - maintenance_tool (always)
  - lookup_user_prompts (always, reads conversation history archives)
  - update_plan (default on, configurable via PlanConfig)
  - get_running_status (default off, opt-in via enable_status)

The plan is stored as structured JSON items in a PlanRegistry with
per-thread isolation. Plan enforcement (nudging on premature stop)
is handled at the graph level via the plan_guard node; this middleware
creates and exposes the PlanRegistry and resets the nudge counter per
user turn.

Planning guidance for the system message is handled by
wrap_system_message. Tool guidance lives in the tool descriptions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.environments.agentic_core.config import PlanConfig
from arion_agent.environments.agentic_core.plan_registry import PlanRegistry
from arion_agent.environments.agentic_core.tools import create_agentic_core_tools
from arion_agent.middleware.base import ArionMiddleware
from arion_agent.util.stats import AgentStats

logger = logging.getLogger(__name__)


class AgenticCoreEnvironment(ArionMiddleware):
    """Middleware providing agent's core reasoning and lifecycle tools.

    Creates a PlanRegistry with per-thread plan storage when planning
    is enabled. The registry lazily loads thread-specific plan files
    from disk on first access. Resets the nudge counter at the start
    of each user turn via before_agent.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        workspace_dir: Path | None = None,
        identity_dir: Path | None = None,
        stats: AgentStats | None = None,
        plan_config: PlanConfig | None = None,
        enable_status: bool = False,
        clock: Any | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._workspace_dir = workspace_dir
        self._identity_dir = identity_dir
        self._stats = stats
        self._plan_config = plan_config
        self._enable_status = enable_status
        self._plan_registry: PlanRegistry | None = None

        if plan_config is not None and agent_id and workspace_dir:
            plans_dir = workspace_dir / ".arion" / "agents" / agent_id / "plans"
            self._plan_registry = PlanRegistry(
                max_nudges=plan_config.max_nudges,
                plans_dir=plans_dir,
            )

        self._tools = create_agentic_core_tools(
            agent_id=agent_id,
            workspace_dir=workspace_dir,
            stats=stats,
            plan_config=plan_config,
            plan_registry=self._plan_registry,
            enable_status=enable_status,
            clock=clock,
            identity_dir=identity_dir,
        )

    @property
    def plan_registry(self) -> PlanRegistry | None:
        """The PlanRegistry for this agent, or None if planning is disabled."""
        return self._plan_registry

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Reset nudge counter at the start of each user turn."""
        if self._plan_registry is not None:
            self._plan_registry.reset_nudge_count()
        return None

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        """Contribute <planning> section to the system message."""
        cfg = self._plan_config
        if cfg is None or not self._workspace_dir or not self._agent_id:
            return parts

        from arion_agent.util.persistence import workspace_relative_path

        reg = self._plan_registry
        if reg is not None and reg.get_persist_path() is not None:
            plan_path = workspace_relative_path(
                reg.get_persist_path(), self._workspace_dir,
            )
        else:
            plan_path = workspace_relative_path(
                self._workspace_dir / ".arion" / "agents" / self._agent_id / "plans",
                self._workspace_dir,
            )

        prompt = cfg.effective_system_instructions().format(plan_path=plan_path)
        parts.append(prompt)
        return parts

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

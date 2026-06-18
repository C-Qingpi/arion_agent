"""Subagenting configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from arion_agent.identity.config import SoulConfig


@dataclass
class SubAgentSpec:
    """Defines a subagent class (like an RPG class: researcher, reviewer, etc.).

    Each spec is a template. Spawning creates an instance with its own agent_id.
    """

    name: str
    description: str
    soul: Any | None = None
    model: Any | None = None
    tools: list[Any] | None = None
    workspace_dir: str | None = None
    tier: str = "important"
    fertile: bool = False
    inherit_identity: bool = False
    subagents: list[SubAgentSpec] | None = None
    summarization: Any | None = None
    max_turns: int = 50


@dataclass(frozen=True)
class SubagentEvent:
    """Record of a subagent lifecycle event."""

    phase: str
    parent_agent_id: str
    child_agent_id: str
    child_thread_id: str
    subagent_class: str
    error: str | None = None


@runtime_checkable
class SubagentCallback(Protocol):
    """Protocol for subagent lifecycle callbacks."""

    def __call__(self, event: SubagentEvent) -> None: ...

"""Subagenting: spawn and manage child agents.

Two schemes:
  SubagentMiddleware        - task tool (fire-and-forget, default)
  ManagedSubagentMiddleware - terminal-like (spawn/send/read/dismiss)
"""

from arion_agent.subagenting.config import (
    SubAgentSpec,
    SubagentCallback,
    SubagentEvent,
)
from arion_agent.subagenting.managed import ManagedSubagentMiddleware
from arion_agent.subagenting.middleware import SubagentMiddleware
from arion_agent.subagenting.templates import (
    SELF_CLONE,
    SELF_INFERTILE_CLONE,
    TASK_SUBAGENT,
)

__all__ = [
    "ManagedSubagentMiddleware",
    "SELF_CLONE",
    "SELF_INFERTILE_CLONE",
    "SubAgentSpec",
    "SubagentCallback",
    "SubagentEvent",
    "SubagentMiddleware",
    "TASK_SUBAGENT",
]

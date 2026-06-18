"""AgentContext: shared construction-time state for middleware assemblers.

Eliminates repeated closure variables across middleware assembly blocks
in create_arion_agent. Each assembler receives this context instead of
5-6 individual parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arion_agent.util.timezone import AgentClock


class AgentContext:
    """Shared state during agent construction.

    Created once at the start of create_arion_agent and passed to all
    middleware assembler functions. Mutable: heartbeat can override the
    clock if it specifies a non-UTC timezone.
    """

    __slots__ = (
        "agent_id",
        "identity_dir",
        "workspace_dir",
        "clock",
        "stats",
        "default_model_spec",
        "extra_model_kwargs",
    )

    def __init__(
        self,
        agent_id: str,
        identity_dir: Path,
        workspace_dir: Path,
        clock: AgentClock,
        stats: Any,
        default_model_spec: Any,
        extra_model_kwargs: dict[str, Any],
    ) -> None:
        self.agent_id = agent_id
        self.identity_dir = identity_dir
        self.workspace_dir = workspace_dir
        self.clock = clock
        self.stats = stats
        self.default_model_spec = default_model_spec
        self.extra_model_kwargs = extra_model_kwargs

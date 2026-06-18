"""Heartbeat effectors: what happens when a trigger fires.

SyntheticPromptEffector is the primary effector — it constructs a
HumanMessage and invokes the agent. The three-part structure (prepend,
body, append) is a countermeasure against LLM mid-interruption attention
drift, not redundancy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arion_agent.environments.heartbeat.config import BaseEffector, TriggerContext

logger = logging.getLogger(__name__)


def _resolve_thread_id(
    thread_id: str | None,
    context: TriggerContext,
    prefix: str = "heartbeat",
) -> str:
    """Resolve thread_id: None=auto, string=template or literal."""
    if thread_id is None:
        trigger_name = context.get("trigger_name", "unknown")
        date_str = context.get("timestamp", "")[:10]
        return f"{prefix}-{trigger_name}-{date_str}"
    return context.format_template(thread_id)


class SyntheticPromptEffector(BaseEffector):
    """Constructs a HumanMessage and invokes the agent.

    prepend: context frame (trigger metadata). Sets situational awareness.
    body: core instruction.
    append: task-continuity anchor. Counteracts LLM tendency to overwrite
        prior work context with new instruction.
    thread_id: Controls which conversation thread the heartbeat uses.
        None (default) — auto-generate per trigger per day, isolating
            heartbeat conversations from user threads.
        A string — used as-is or as a template with {variables}.
            Use a fixed string like "main" to inject into the agent's
            main conversation. Use a template like "heartbeat-{trigger_name}"
            for one persistent thread per trigger without date rotation.
    """

    def __init__(
        self,
        body: str = "",
        prepend: str = "",
        append: str = "",
        thread_id: str | None = None,
    ) -> None:
        self.prepend = prepend
        self.body = body
        self.append = append
        self.thread_id = thread_id

    async def execute(self, context: TriggerContext, agent: Any = None) -> None:
        if agent is None:
            logger.warning("SyntheticPromptEffector: no agent provided, skipping")
            return

        parts = []
        if self.prepend:
            parts.append(context.format_template(self.prepend))
        if self.body:
            parts.append(context.format_template(self.body))
        if self.append:
            parts.append(context.format_template(self.append))

        prompt = "\n".join(parts)
        if not prompt.strip():
            logger.warning("SyntheticPromptEffector: empty prompt after template, skipping")
            return

        thread_id = _resolve_thread_id(self.thread_id, context, prefix="heartbeat")

        logger.info("Heartbeat firing synthetic prompt for '%s' (thread: %s)", context.get("trigger_name"), thread_id)

        await agent.ainvoke(
            {"messages": [("user", prompt)]},
            config={"configurable": {"thread_id": thread_id}},
        )


class SpawnAgentEffector(BaseEffector):
    """Triggers subagent spawning via the agent's task tool."""

    def __init__(
        self,
        spec_name: str,
        prompt_template: str = "",
        thread_id: str | None = None,
    ) -> None:
        self.spec_name = spec_name
        self.prompt_template = prompt_template
        self.thread_id = thread_id

    async def execute(self, context: TriggerContext, agent: Any = None) -> None:
        prompt = context.format_template(self.prompt_template) if self.prompt_template else ""
        synthetic = (
            f"[Heartbeat: spawning subagent '{self.spec_name}' at {{timestamp}}]\n"
            f"Use the task tool to spawn '{self.spec_name}' with this instruction: {prompt}"
        )
        synthetic = context.format_template(synthetic)
        if agent is None:
            logger.warning("SpawnAgentEffector: no agent provided, skipping")
            return

        tid = _resolve_thread_id(self.thread_id, context, prefix="heartbeat-spawn")

        await agent.ainvoke(
            {"messages": [("user", synthetic)]},
            config={"configurable": {"thread_id": tid}},
        )


class FileOperationEffector(BaseEffector):
    """Performs a file operation without invoking the agent."""

    def __init__(
        self,
        operation: str = "append",
        target_path: str = "",
        content: str = "",
        workspace_dir: Path | None = None,
    ) -> None:
        self.operation = operation
        self.target_path = target_path
        self.content = content
        self.workspace_dir = workspace_dir

    async def execute(self, context: TriggerContext, agent: Any = None) -> None:
        resolved_path = context.format_template(self.target_path)
        resolved_content = context.format_template(self.content)

        if self.workspace_dir:
            path = self.workspace_dir / resolved_path
        else:
            path = Path(resolved_path)

        from arion_agent.util.persistence import ensure_directory, append_file, write_file as persistence_write, touch

        ensure_directory(path.parent)

        if self.operation == "append":
            append_file(path, resolved_content)
        elif self.operation == "write":
            persistence_write(path, resolved_content)
        elif self.operation == "touch":
            touch(path)
        else:
            logger.warning("FileOperationEffector: unknown operation '%s'", self.operation)


class CallbackEffector(BaseEffector):
    """Developer-provided async callable for maximum flexibility."""

    def __init__(self, callback: Any) -> None:
        self.callback = callback

    async def execute(self, context: TriggerContext, agent: Any = None) -> None:
        await self.callback(context)


class CompositeEffector(BaseEffector):
    """Chains multiple effectors sequentially."""

    def __init__(self, effectors: list[BaseEffector]) -> None:
        self.effectors = effectors

    async def execute(self, context: TriggerContext, agent: Any = None) -> None:
        for eff in self.effectors:
            await eff.execute(context, agent=agent)

"""Subagent middleware: contributes the task tool and injects roster into system prompt."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool

from arion_agent.middleware.base import ArionMiddleware
from arion_agent.subagenting.config import SubAgentSpec, SubagentCallback, SubagentEvent
from arion_agent.subagenting.prompts import (
    DEFAULT_SUBAGENT_INSTRUCTIONS,
    TASK_TOOL_DESCRIPTION,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def _resolve_clone_identity(
    parent_workspace: str,
    parent_agent_id: str,
    parent_soul: Any,
    parent_deep_memory: Any,
    parent_shallow_memory: Any,
    parent_skills: Any,
) -> dict[str, Any]:
    """Read parent's evolved identity files and build configs for a clone.

    For self-clone and self-infertile-clone, the child should start with
    the parent's current on-disk identity (SOUL.md, DEEPMEMORY.md, etc.),
    not the original template the parent was seeded with. This function
    reads evolved files and constructs fresh config objects preserving
    the parent's instruction text.
    """
    from arion_agent.identity.config import MemoryConfig, ShallowMemoryConfig, SoulConfig
    from arion_agent.util.persistence import file_exists, is_directory, read_file_text

    parent_identity_dir = Path(parent_workspace) / ".arion" / "agents" / parent_agent_id
    result: dict[str, Any] = {}

    # --- Soul ---
    if isinstance(parent_soul, SoulConfig):
        instructions, template = parent_soul.instructions, parent_soul.initial_template
    elif isinstance(parent_soul, str):
        instructions, template = "", parent_soul
    else:
        instructions, template = "", ""

    soul_path = parent_identity_dir / "SOUL.md"
    evolved = read_file_text(soul_path, max_chars=10_000) if file_exists(soul_path) else template
    if evolved:
        result["soul"] = SoulConfig(initial_template=evolved, instructions=instructions)

    # --- Deep Memory ---
    if isinstance(parent_deep_memory, MemoryConfig):
        dm_instr, dm_tmpl = parent_deep_memory.instructions, parent_deep_memory.initial_template
    elif isinstance(parent_deep_memory, str):
        dm_instr, dm_tmpl = "", parent_deep_memory
    else:
        dm_instr, dm_tmpl = "", ""

    dm_path = parent_identity_dir / "DEEPMEMORY.md"
    evolved_dm = read_file_text(dm_path, max_chars=5_000) if file_exists(dm_path) else dm_tmpl
    if evolved_dm:
        result["deep_memory"] = MemoryConfig(initial_template=evolved_dm, instructions=dm_instr)

    # --- Shallow Memory ---
    if isinstance(parent_shallow_memory, ShallowMemoryConfig):
        sm_guidance = parent_shallow_memory.guidance
        sm_folders = list(parent_shallow_memory.initial_folders)
        sm_instr = parent_shallow_memory.instructions
    else:
        sm_guidance, sm_folders, sm_instr = "", [], ""

    sm_path = parent_identity_dir / "SHALLOW_MEMORY.md"
    evolved_sm = read_file_text(sm_path) if file_exists(sm_path) else sm_guidance
    if evolved_sm:
        result["shallow_memory"] = ShallowMemoryConfig(
            guidance=evolved_sm, initial_folders=sm_folders, instructions=sm_instr,
        )

    # --- Skills ---
    if parent_skills is not None:
        from arion_agent.skills.config import scan_skills_directory
        from arion_agent.skills.middleware import SkillMiddleware

        important_dir = parent_identity_dir / "skills" / "important"
        generic_dir = parent_identity_dir / "skills" / "generic"
        sources: list[str] = []
        important_names: list[str] = []

        if is_directory(important_dir):
            sources.append(str(important_dir))
            for meta in scan_skills_directory(important_dir):
                important_names.append(meta.name)
        if is_directory(generic_dir):
            sources.append(str(generic_dir))

        skill_instr = getattr(parent_skills, "_instructions", "")
        if sources:
            result["skills"] = SkillMiddleware(
                important_skills=important_names,
                skill_sources=sources,
                instructions=skill_instr,
            )

    return result


class SubagentMiddleware(ArionMiddleware):
    """Provides subagent spawning capability via a task tool.

    Subagents are full ArionAgent instances created via recursive
    create_arion_agent, each with its own agent_id and thread_id.
    """

    def __init__(
        self,
        specs: list[SubAgentSpec],
        *,
        parent_agent_id: str = "unknown",
        parent_model: Any = None,
        parent_workspace: str = ".",
        parent_tools: list[BaseTool] | None = None,
        parent_soul: Any = None,
        parent_deep_memory: Any = None,
        parent_shallow_memory: Any = None,
        parent_skills: Any = None,
        parent_mounts: Any | None = None,
        instructions: str = DEFAULT_SUBAGENT_INSTRUCTIONS,
        on_subagent: SubagentCallback | list[SubagentCallback] | None = None,
        max_recursion_depth: int | None = None,
        checkpointer: Any = None,
    ) -> None:
        self._specs = {s.name: s for s in specs}
        self._parent_agent_id = parent_agent_id
        self._parent_model = parent_model
        self._parent_workspace = parent_workspace
        self._parent_tools = parent_tools
        self._parent_soul = parent_soul
        self._parent_deep_memory = parent_deep_memory
        self._parent_shallow_memory = parent_shallow_memory
        self._parent_skills = parent_skills
        self._parent_mounts = parent_mounts
        self._instructions = instructions
        self._max_recursion_depth = max_recursion_depth
        self._checkpointer = checkpointer

        if on_subagent is None:
            self._on_subagent: list[SubagentCallback] = []
        elif isinstance(on_subagent, list):
            self._on_subagent = on_subagent
        else:
            self._on_subagent = [on_subagent]

        self._important = [s for s in specs if s.tier == "important"]
        self._generic = [s for s in specs if s.tier == "generic"]

        self._task_tool = self._build_task_tool()

    def _fire_callbacks(self, event: SubagentEvent) -> None:
        for cb in self._on_subagent:
            try:
                cb(event)
            except Exception:
                logger.exception("Subagent callback failed")

    def _build_task_tool(self) -> BaseTool:
        class_descriptions = []
        for s in self._specs.values():
            class_descriptions.append(f"  {s.name}: {s.description}")
        classes_text = "\n".join(class_descriptions) or "  (none defined)"

        description = TASK_TOOL_DESCRIPTION.format(available_classes=classes_text)

        middleware_ref = self

        async def _task_impl(
            subagent_class: str,
            task: str,
            context: str = "",
        ) -> str:
            return await middleware_ref._spawn_and_run(subagent_class, task, context)

        return StructuredTool.from_function(
            coroutine=_task_impl,
            name="task",
            description=description,
        )

    async def _spawn_and_run(
        self,
        subagent_class: str,
        task: str,
        context: str,
        thread_id: str | None = None,
    ) -> str:
        spec = self._specs.get(subagent_class)
        if spec is None:
            available = ", ".join(self._specs.keys())
            return f"Unknown subagent class: {subagent_class}. Available: {available}"

        child_agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        child_thread_id = thread_id or child_agent_id

        self._fire_callbacks(SubagentEvent(
            phase="spawn",
            parent_agent_id=self._parent_agent_id,
            child_agent_id=child_agent_id,
            child_thread_id=child_thread_id,
            subagent_class=subagent_class,
        ))

        child_workspace = spec.workspace_dir or self._parent_workspace
        child_model = spec.model or self._parent_model
        child_soul = spec.soul or self._parent_soul
        child_tools = spec.tools
        child_subagents = spec.subagents if spec.fertile else None

        child_depth = None
        if self._max_recursion_depth is not None:
            child_depth = self._max_recursion_depth - 1
            if child_depth <= 0:
                child_subagents = None

        clone_kwargs: dict[str, Any] = {}
        if spec.inherit_identity:
            clone_kwargs = _resolve_clone_identity(
                self._parent_workspace,
                self._parent_agent_id,
                self._parent_soul,
                self._parent_deep_memory,
                self._parent_shallow_memory,
                self._parent_skills,
            )
            child_soul = clone_kwargs.pop("soul", child_soul)

        from arion_agent.graph import create_arion_agent

        try:
            child_mounts = self._parent_mounts if child_workspace == self._parent_workspace else None

            child = create_arion_agent(
                model=child_model,
                workspace_dir=child_workspace,
                agent_id=child_agent_id,
                soul=child_soul,
                tools=child_tools,
                subagents=child_subagents,
                mounts=child_mounts,
                summarization=spec.summarization,
                checkpointer=self._checkpointer if child_workspace == self._parent_workspace else True,
                recursion_limit=spec.max_turns,
                max_recursion_depth=child_depth,
                _parent_agent_id=self._parent_agent_id,
                _parent_thread_id=child_thread_id,
                **clone_kwargs,
            )

            full_task = task
            if context:
                full_task = f"{task}\n\nContext:\n{context}"

            result = await child.ainvoke(
                {"messages": [("user", full_task)]},
                config={"configurable": {"thread_id": child_thread_id}},
            )

            ai_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "ai"]
            if ai_msgs:
                final = ai_msgs[-1].content
                if isinstance(final, list):
                    final = "\n".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in final
                    )
            else:
                final = "(Subagent produced no response)"

            self._fire_callbacks(SubagentEvent(
                phase="complete",
                parent_agent_id=self._parent_agent_id,
                child_agent_id=child_agent_id,
                child_thread_id=child_thread_id,
                subagent_class=subagent_class,
            ))

            return final

        except Exception as exc:
            error_msg = f"Subagent {subagent_class} ({child_agent_id}) failed: {exc}"
            logger.error(error_msg)
            self._fire_callbacks(SubagentEvent(
                phase="error",
                parent_agent_id=self._parent_agent_id,
                child_agent_id=child_agent_id,
                child_thread_id=child_thread_id,
                subagent_class=subagent_class,
                error=str(exc),
            ))
            return error_msg

    @property
    def tools(self) -> list[BaseTool]:
        return [self._task_tool]

    @property
    def tool_timeout_overrides(self) -> dict[str, int | None]:
        return {"task": None}

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        """Contribute subagent guidance sections to the system message."""
        parts.append(f"<subagent_guidance>\n{self._instructions}\n</subagent_guidance>")

        if self._important:
            lines = ["<important_subagents>"]
            for s in self._important:
                lines.append(f'<subagent name="{s.name}">')
                lines.append(s.description)
                lines.append("</subagent>")
            lines.append("</important_subagents>")
            parts.append("\n".join(lines))

        if self._generic:
            parts.append(
                "<generic_subagents>\n"
                "Additional subagent classes are available. "
                "Use the task tool with any of these classes: "
                + ", ".join(s.name for s in self._generic)
                + "\n</generic_subagents>"
            )

        return parts

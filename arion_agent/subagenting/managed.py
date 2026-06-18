"""Managed subagent scheme: spawn/send/read/dismiss (terminal-like).

Alternative to the default task tool (subagent-as-tool). Provides
long-lived subagents that the parent can interact with across turns.

Tools contributed:
  subagent_spawn   - create a new subagent from a class
  subagent_send    - send a message to a running subagent
  subagent_read    - read the latest response from a subagent
  subagent_dismiss - end a subagent session
  subagent_list    - list active subagents
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from arion_agent.middleware.base import ArionMiddleware
from arion_agent.subagenting.config import SubAgentSpec, SubagentCallback, SubagentEvent
from arion_agent.subagenting.prompts import DEFAULT_SUBAGENT_INSTRUCTIONS

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

MAX_MANAGED_SUBAGENTS = 4


class _ManagedChild:
    """Tracks a running managed subagent."""

    def __init__(self, agent_id: str, thread_id: str, subagent_class: str, agent: Any):
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.subagent_class = subagent_class
        self.agent = agent
        self.messages: list[Any] = []
        self.last_response: str = "(no response yet)"


class ManagedSubagentMiddleware(ArionMiddleware):
    """Terminal-like subagent management: spawn, send, read, dismiss.

    Unlike the task tool (fire-and-forget), managed subagents persist
    across parent turns. The parent can send follow-up messages, read
    responses, and dismiss children when done.
    """

    def __init__(
        self,
        specs: list[SubAgentSpec],
        *,
        parent_agent_id: str = "unknown",
        parent_model: Any = None,
        parent_workspace: str = ".",
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
        self._parent_soul = parent_soul
        self._parent_deep_memory = parent_deep_memory
        self._parent_shallow_memory = parent_shallow_memory
        self._parent_skills = parent_skills
        self._parent_mounts = parent_mounts
        self._instructions = instructions
        self._max_recursion_depth = max_recursion_depth
        self._checkpointer = checkpointer
        self._children: dict[str, _ManagedChild] = {}

        if on_subagent is None:
            self._on_subagent: list[SubagentCallback] = []
        elif isinstance(on_subagent, list):
            self._on_subagent = on_subagent
        else:
            self._on_subagent = [on_subagent]

        self._tools_list = self._build_tools()

    def _fire_callbacks(self, event: SubagentEvent) -> None:
        for cb in self._on_subagent:
            try:
                cb(event)
            except Exception:
                logger.exception("Subagent callback failed")

    def _build_tools(self) -> list[BaseTool]:
        mw = self

        class_list = ", ".join(self._specs.keys()) or "(none)"

        async def subagent_spawn(
            subagent_class: str,
            name: str,
            initial_task: str,
        ) -> str:
            """Spawn a new managed subagent. It persists until dismissed.

            Args:
                subagent_class: Class from the roster to spawn.
                name: A short name for this instance (used to reference it later).
                initial_task: First task/message to send to the subagent.

            Returns:
                The subagent's initial response.
            """
            if name in mw._children:
                return f"Subagent '{name}' already exists. Use subagent_send to message it."
            if len(mw._children) >= MAX_MANAGED_SUBAGENTS:
                active = ", ".join(mw._children.keys())
                return f"Max {MAX_MANAGED_SUBAGENTS} managed subagents. Active: {active}. Dismiss one first."

            spec = mw._specs.get(subagent_class)
            if spec is None:
                return f"Unknown subagent class: {subagent_class}. Available: {class_list}"

            child_agent_id = f"agent-{uuid.uuid4().hex[:8]}"
            child_thread_id = child_agent_id

            mw._fire_callbacks(SubagentEvent(
                phase="spawn",
                parent_agent_id=mw._parent_agent_id,
                child_agent_id=child_agent_id,
                child_thread_id=child_thread_id,
                subagent_class=subagent_class,
            ))

            from arion_agent.graph import create_arion_agent
            from arion_agent.subagenting.middleware import _resolve_clone_identity

            child_workspace = spec.workspace_dir or mw._parent_workspace
            child_model = spec.model or mw._parent_model
            child_soul = spec.soul or mw._parent_soul
            child_subagents = spec.subagents if spec.fertile else None

            child_depth = None
            if mw._max_recursion_depth is not None:
                child_depth = mw._max_recursion_depth - 1
                if child_depth <= 0:
                    child_subagents = None

            clone_kwargs: dict[str, Any] = {}
            if spec.inherit_identity:
                clone_kwargs = _resolve_clone_identity(
                    mw._parent_workspace,
                    mw._parent_agent_id,
                    mw._parent_soul,
                    mw._parent_deep_memory,
                    mw._parent_shallow_memory,
                    mw._parent_skills,
                )
                child_soul = clone_kwargs.pop("soul", child_soul)

            try:
                child_mounts = mw._parent_mounts if child_workspace == mw._parent_workspace else None

                agent = create_arion_agent(
                    model=child_model,
                    workspace_dir=child_workspace,
                    agent_id=child_agent_id,
                    soul=child_soul,
                    subagents=child_subagents,
                    mounts=child_mounts,
                    summarization=spec.summarization,
                    checkpointer=mw._checkpointer if child_workspace == mw._parent_workspace else True,
                    recursion_limit=spec.max_turns,
                    max_recursion_depth=child_depth,
                    _parent_agent_id=mw._parent_agent_id,
                    _parent_thread_id=child_thread_id,
                    **clone_kwargs,
                )

                result = await agent.ainvoke(
                    {"messages": [("user", initial_task)]},
                    config={"configurable": {"thread_id": child_thread_id}},
                )

                child = _ManagedChild(child_agent_id, child_thread_id, subagent_class, agent)
                child.messages = result["messages"]
                ai_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "ai"]
                child.last_response = _extract_text(ai_msgs[-1]) if ai_msgs else "(no response)"
                mw._children[name] = child

                return f"[{name}] spawned ({subagent_class}, id={child_agent_id}).\nResponse: {child.last_response}"

            except Exception as exc:
                mw._fire_callbacks(SubagentEvent(
                    phase="error",
                    parent_agent_id=mw._parent_agent_id,
                    child_agent_id=child_agent_id,
                    child_thread_id=child_thread_id,
                    subagent_class=subagent_class,
                    error=str(exc),
                ))
                return f"Failed to spawn {subagent_class}: {exc}"

        async def subagent_send(name: str, message: str) -> str:
            """Send a follow-up message to a running managed subagent.

            Args:
                name: Name of the subagent (from subagent_spawn).
                message: Message to send.

            Returns:
                The subagent's response.
            """
            child = mw._children.get(name)
            if child is None:
                active = ", ".join(mw._children.keys()) or "(none)"
                return f"No subagent named '{name}'. Active: {active}"

            try:
                result = await child.agent.ainvoke(
                    {"messages": child.messages + [("user", message)]},
                    config={"configurable": {"thread_id": child.thread_id}},
                )
                child.messages = result["messages"]
                ai_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "ai"]
                child.last_response = _extract_text(ai_msgs[-1]) if ai_msgs else "(no response)"
                return f"[{name}] response: {child.last_response}"
            except Exception as exc:
                return f"[{name}] error: {exc}"

        async def subagent_read(name: str) -> str:
            """Read the latest response from a managed subagent.

            Args:
                name: Name of the subagent.

            Returns:
                The most recent response.
            """
            child = mw._children.get(name)
            if child is None:
                active = ", ".join(mw._children.keys()) or "(none)"
                return f"No subagent named '{name}'. Active: {active}"
            return f"[{name}] last response: {child.last_response}"

        async def subagent_dismiss(name: str) -> str:
            """End a managed subagent session and release its resources.

            Args:
                name: Name of the subagent to dismiss.

            Returns:
                Confirmation with the subagent's final response.
            """
            child = mw._children.pop(name, None)
            if child is None:
                active = ", ".join(mw._children.keys()) or "(none)"
                return f"No subagent named '{name}'. Active: {active}"

            mw._fire_callbacks(SubagentEvent(
                phase="complete",
                parent_agent_id=mw._parent_agent_id,
                child_agent_id=child.agent_id,
                child_thread_id=child.thread_id,
                subagent_class=child.subagent_class,
            ))
            return f"[{name}] dismissed. Final response: {child.last_response}"

        async def subagent_list() -> str:
            """List all active managed subagents.

            Returns:
                Names, classes, and IDs of active subagents.
            """
            if not mw._children:
                return "No active subagents."
            lines = []
            for name, child in mw._children.items():
                lines.append(f"  {name}: class={child.subagent_class}, id={child.agent_id}")
            return "Active subagents:\n" + "\n".join(lines)

        return [
            StructuredTool.from_function(coroutine=subagent_spawn, name="subagent_spawn",
                description=f"Spawn a managed subagent. Available classes: {class_list}"),
            StructuredTool.from_function(coroutine=subagent_send, name="subagent_send",
                description="Send a follow-up message to a running subagent."),
            StructuredTool.from_function(coroutine=subagent_read, name="subagent_read",
                description="Read the latest response from a subagent."),
            StructuredTool.from_function(coroutine=subagent_dismiss, name="subagent_dismiss",
                description="End a subagent session and release resources."),
            StructuredTool.from_function(coroutine=subagent_list, name="subagent_list",
                description="List active managed subagents."),
        ]

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools_list

    @property
    def tool_timeout_overrides(self) -> dict[str, int | None]:
        return {"subagent_spawn": None, "subagent_send": None}

    def wrap_model_call(
        self,
        messages: list[Any],
        tools: list[BaseTool],
        **kwargs: Any,
    ) -> tuple[list[Any], list[BaseTool], dict[str, Any]]:
        prompt = (
            "<subagent_guidance>\n"
            f"{self._instructions}\n"
            "You manage subagents like terminal sessions: spawn, send follow-ups, "
            "read responses, and dismiss when done.\n"
            "</subagent_guidance>"
        )
        has_prompt = any(
            isinstance(m, SystemMessage) and "subagent_guidance" in str(m.content)
            for m in messages
        )
        if not has_prompt:
            messages = messages + [SystemMessage(content=prompt)]
        return messages, tools, kwargs


def _extract_text(msg: Any) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in c
        )
    return str(c)

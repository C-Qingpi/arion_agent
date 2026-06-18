"""Base middleware class for ArionAgent.

Middleware can hook into seven lifecycle points:
  - before_agent:         Patch state before the first model call
  - wrap_system_message:  Contribute sections to the single system message
  - wrap_model_call:      Modify messages/tools before each LLM invocation
  - wrap_model_response:  Transform the LLM response before it is checkpointed
  - wrap_tool_call:       Intercept individual tool executions
  - drain_state_updates:  Return state mutations (e.g. RemoveMessage) after LLM responds
  - after_agent:          Run cleanup after the agent loop completes

System message assembly:
  Each middleware contributes sections via wrap_system_message (identity,
  planning, etc.). The graph assembles all sections into a single
  SystemMessage before the model call. This avoids cross-provider issues
  (e.g. Anthropic rejecting non-consecutive SystemMessages) and keeps
  each middleware responsible for its own domain.

Middleware can also expose tools via a `tools` property.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class ArionMiddleware(ABC):
    """Base class for all ArionAgent middleware.

    Subclass and override only the hooks you need.
    """

    @property
    def tools(self) -> list[BaseTool]:
        """Tools contributed by this middleware. Override to add tools."""
        return []

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Called once before the agent loop starts. Return state patch or None."""
        return None

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        """Contribute sections to the system message.

        Called before each LLM invocation. Append text sections to `parts`.
        The graph joins all parts into one SystemMessage. This keeps
        system message assembly cross-provider safe (single SystemMessage)
        while each middleware owns its own domain's content.

        Args:
            parts: Accumulated sections from earlier middleware. Append to this.
            **kwargs: Shared context (thread_id, etc.) from model_node.

        Returns:
            The parts list with any new sections appended.
        """
        return parts

    def wrap_model_call(
        self,
        messages: list[Any],
        tools: list[BaseTool],
        **kwargs: Any,
    ) -> tuple[list[Any], list[BaseTool], dict[str, Any]]:
        """Transform messages and tools before each LLM call.

        NOTE: System message injection should use wrap_system_message instead.
        This hook is for message/tool transforms (truncation, filtering, etc.)

        Returns (messages, tools, extra_kwargs).
        """
        return messages, tools, kwargs

    def wrap_model_response(self, response: Any, **kwargs: Any) -> Any:
        """Transform the LLM response before it is appended to state (checkpoint).

        Use this to normalize provider-specific content blocks (e.g. strip
        signature blocks, move thinking into additional_kwargs) so the
        checkpoint stays provider-agnostic for all consumers.
        """
        return response

    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
    ) -> Any:
        """Post-process a tool result. Return the (possibly modified) result."""
        return tool_result

    def drain_state_updates(self) -> list[Any]:
        """Return pending state mutations (e.g. RemoveMessage) and clear the buffer.

        Called by model_node after the LLM responds.
        PatchToolCallsMiddleware overrides this to return cleanup messages.
        """
        return []

    def after_agent(self, state: dict[str, Any]) -> None:
        """Called after the agent loop finishes."""

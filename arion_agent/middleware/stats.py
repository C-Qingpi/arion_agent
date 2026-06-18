"""Stats middleware: tracks token usage, call counts, and session logging.

Always-on, zero-overhead middleware that accumulates session statistics.
Optionally writes structured JSONL session logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.middleware.base import ArionMiddleware
from arion_agent.util.stats import AgentStats, SessionLogger
from arion_agent.util.tokens import estimate_message_tokens


class StatsMiddleware(ArionMiddleware):
    """Accumulates session statistics and optional JSONL logging."""

    def __init__(
        self,
        stats: AgentStats,
        *,
        session_logger: SessionLogger | None = None,
    ) -> None:
        self._stats = stats
        self._logger = session_logger

    def wrap_model_call(
        self,
        messages: list[Any],
        tools: list[BaseTool],
        **kwargs: Any,
    ) -> tuple[list[Any], list[BaseTool], dict[str, Any]]:
        self._stats.model_calls += 1
        self._stats.total_messages = len(messages)

        input_est = 0
        for m in messages:
            input_est += estimate_message_tokens(m.content)
        self._stats.input_tokens_estimated += input_est

        if self._logger:
            self._logger.log(
                "model_call",
                call_number=self._stats.model_calls,
                message_count=len(messages),
                tool_count=len(tools),
                input_tokens_estimated=input_est,
            )

        return messages, tools, kwargs

    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
    ) -> Any:
        self._stats.tool_calls += 1

        if self._logger:
            self._logger.log("tool_call", tool=tool_name)

        return tool_result

    def after_agent(self, state: dict[str, Any]) -> None:
        """Extract actual token usage from the last AI message if available."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "ai":
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    actual_in = getattr(usage, "input_tokens", 0)
                    actual_out = getattr(usage, "output_tokens", 0)
                    if actual_in:
                        self._stats.input_tokens_actual += actual_in
                    if actual_out:
                        self._stats.output_tokens_actual += actual_out
                        self._stats.output_tokens_estimated += actual_out
                break

"""Global tool executor with timeout, error handling, and output size guardrail.

Every tool call goes through the executor which wraps it with:
  - Configurable timeout (per-tool override or global default)
  - Output size guardrail: truncate oversized results with head+tail preview
  - Error capture: on failure, the LLM receives a structured error with
    the exception type, message, and any partial output produced before failure.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

from langchain_core.messages import ToolMessage

DEFAULT_TOOL_TIMEOUT_SECONDS = 120
DEFAULT_MAX_OUTPUT_CHARS = 100_000


class ToolExecutionError(Exception):
    """Raised when a tool execution fails or times out."""

    def __init__(self, tool_name: str, tool_call_id: str, message: str, partial_output: str = ""):
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.partial_output = partial_output
        super().__init__(message)


def _format_error_result(
    tool_name: str,
    error_type: str,
    error_msg: str,
    partial_output: str = "",
) -> str:
    parts = [
        f"TOOL ERROR ({tool_name})",
        f"Type: {error_type}",
        f"Message: {error_msg}",
    ]
    if partial_output:
        parts.append(f"Partial output before failure:\n{partial_output}")
    return "\n".join(parts)


def _truncate_output(content: str, max_chars: int) -> str:
    """Truncate oversized tool output with head + tail preview.

    Keeps first 80% of budget as head, last 10% as tail, with a truncation
    notice in the middle. The agent sees the beginning (headers/context) and
    end (final result/error) of the output.
    """
    if len(content) <= max_chars:
        return content

    head_budget = int(max_chars * 0.80)
    tail_budget = int(max_chars * 0.10)

    head = content[:head_budget]
    tail = content[-tail_budget:] if tail_budget > 0 else ""
    omitted = len(content) - head_budget - tail_budget
    omitted_lines = content[head_budget : len(content) - tail_budget].count("\n") if tail_budget > 0 else content[head_budget:].count("\n")

    notice = (
        f"\n\n[...TRUNCATED: {omitted:,} chars omitted (~{omitted_lines:,} lines). "
        f"Output exceeded {max_chars:,} char limit. "
        f"Consider using more targeted commands or processing in smaller chunks.]\n\n"
    )

    return head + notice + tail


class ToolExecutor:
    """Wraps tool invocations with timeout, error handling, and output truncation.

    Usage:
        executor = ToolExecutor(default_timeout=120)
        result_message = await executor.execute(tool, tool_call)
    """

    def __init__(
        self,
        default_timeout: int = DEFAULT_TOOL_TIMEOUT_SECONDS,
        timeout_overrides: dict[str, int | None] | None = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ):
        self.default_timeout = default_timeout
        self.timeout_overrides: dict[str, int | None] = timeout_overrides or {}
        self.max_output_chars = max_output_chars

    def get_timeout(self, tool_name: str) -> int | None:
        """Return timeout for a tool. None means no timeout."""
        if tool_name in self.timeout_overrides:
            return self.timeout_overrides[tool_name]
        return self.default_timeout

    async def execute(
        self,
        tool: Any,
        tool_call: dict[str, Any],
    ) -> ToolMessage:
        """Execute a single tool call with timeout, error handling, and output truncation.

        Args:
            tool: The tool object (must have .ainvoke or .invoke).
            tool_call: Dict with keys 'id', 'name', 'args'.

        Returns:
            A ToolMessage with the result (possibly truncated) or a structured error.
        """
        tool_name = tool_call["name"]
        tool_call_id = tool_call.get("id") or f"missing_{tool_name}_{id(tool_call)}"
        timeout = self.get_timeout(tool_name)

        try:
            coro = (
                tool.ainvoke(tool_call["args"])
                if hasattr(tool, "ainvoke")
                else asyncio.to_thread(tool.invoke, tool_call["args"])
            )
            if timeout is None:
                result = await coro
            else:
                result = await asyncio.wait_for(coro, timeout=timeout)

            if isinstance(result, ToolMessage):
                content = result.content
                if isinstance(content, str):
                    result.content = _truncate_output(content, self.max_output_chars)
                return result

            content = str(result) if not isinstance(result, str) else result
            content = _truncate_output(content, self.max_output_chars)
            return ToolMessage(
                content=content,
                name=tool_name,
                tool_call_id=tool_call_id,
            )

        except asyncio.CancelledError:
            raise

        except asyncio.TimeoutError:
            error_content = _format_error_result(
                tool_name,
                "TimeoutError",
                f"Tool execution exceeded {timeout}s timeout. "
                "Consider breaking the task into smaller steps.",
            )
            return ToolMessage(
                content=error_content,
                name=tool_name,
                tool_call_id=tool_call_id,
            )

        except Exception as exc:
            from arion_agent.graph import AgentAborted

            if isinstance(exc, AgentAborted):
                raise
            tb = traceback.format_exception_only(type(exc), exc)
            error_content = _format_error_result(
                tool_name,
                type(exc).__name__,
                str(exc),
                partial_output="".join(tb),
            )
            return ToolMessage(
                content=error_content,
                name=tool_name,
                tool_call_id=tool_call_id,
            )

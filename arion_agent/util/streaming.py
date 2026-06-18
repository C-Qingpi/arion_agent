"""Visual-only LLM streaming.

Uses provider streaming to emit cumulative UI updates while aggregating chunks
into one complete AIMessage for the graph. Incomplete or failed streams raise
like a normal failed invoke — nothing partial enters checkpoint state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

_STREAMING_MODEL_TYPES = frozenset({"ChatMoonshot", "ChatDeepSeek"})


def supports_visual_stream(model: BaseChatModel) -> bool:
    return type(model).__name__ in _STREAMING_MODEL_TYPES


@dataclass(frozen=True)
class LlmStreamUpdate:
    thread_id: str
    phase: Literal["start", "delta", "end"]
    content: str = ""
    reasoning: str = ""


LlmStreamCallback = Callable[[LlmStreamUpdate], None]


def _chunk_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content else ""


def _snapshot_from_chunk(aggregated: AIMessageChunk) -> tuple[str, str]:
    content = _chunk_text(getattr(aggregated, "content", ""))
    extra = getattr(aggregated, "additional_kwargs", None) or {}
    reasoning = extra.get("reasoning_content", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning) if reasoning else ""
    return content, reasoning


def _chunk_to_message(aggregated: AIMessageChunk) -> AIMessage:
    return AIMessage(
        content=aggregated.content,
        additional_kwargs=dict(getattr(aggregated, "additional_kwargs", {}) or {}),
        tool_calls=list(getattr(aggregated, "tool_calls", None) or []),
        invalid_tool_calls=list(getattr(aggregated, "invalid_tool_calls", None) or []),
        response_metadata=dict(getattr(aggregated, "response_metadata", {}) or {}),
        id=getattr(aggregated, "id", None),
        usage_metadata=getattr(aggregated, "usage_metadata", None),
    )


async def invoke_with_visual_stream(
    bound: Any,
    messages: list[BaseMessage],
    *,
    thread_id: str,
    callback: LlmStreamCallback | None,
) -> AIMessage:
    if callback is not None:
        callback(LlmStreamUpdate(thread_id=thread_id, phase="start"))

    aggregated: AIMessageChunk | None = None
    last_content = ""
    last_reasoning = ""

    try:
        async for chunk in bound.astream(messages):
            if not isinstance(chunk, AIMessageChunk):
                chunk = AIMessageChunk(content=chunk) if chunk else AIMessageChunk(content="")

            aggregated = chunk if aggregated is None else aggregated + chunk
            if callback is None:
                continue

            content, reasoning = _snapshot_from_chunk(aggregated)
            if content != last_content or reasoning != last_reasoning:
                last_content = content
                last_reasoning = reasoning
                callback(
                    LlmStreamUpdate(
                        thread_id=thread_id,
                        phase="delta",
                        content=content,
                        reasoning=reasoning,
                    )
                )

        if aggregated is None:
            raise RuntimeError("LLM stream returned no output")

        return _chunk_to_message(aggregated)
    finally:
        if callback is not None:
            callback(LlmStreamUpdate(thread_id=thread_id, phase="end"))

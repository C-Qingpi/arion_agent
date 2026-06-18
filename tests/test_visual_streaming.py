"""Tests for visual-only LLM streaming."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessageChunk

from arion_agent.providers.deepseek import ChatDeepSeek
from arion_agent.providers.moonshot import ChatMoonshot
from arion_agent.util.streaming import (
    LlmStreamUpdate,
    invoke_with_visual_stream,
    supports_visual_stream,
)


class _FakeBound:
    def __init__(self, chunks: list[AIMessageChunk], fail_after: int | None = None):
        self._chunks = chunks
        self._fail_after = fail_after

    async def astream(self, messages):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError("stream interrupted")
            yield chunk


@pytest.mark.parametrize("model_cls", [ChatDeepSeek, ChatMoonshot])
def test_supports_visual_stream(model_cls):
    model = model_cls(model="test", api_key="test")
    assert supports_visual_stream(model) is True


def test_invoke_with_visual_stream_aggregates_and_emits():
    bound = _FakeBound([
        AIMessageChunk(content="Hel", additional_kwargs={"reasoning_content": "think"}),
        AIMessageChunk(content="lo", additional_kwargs={"reasoning_content": "ing"}),
    ])
    events: list[LlmStreamUpdate] = []

    def callback(ev: LlmStreamUpdate) -> None:
        events.append(ev)

    response = asyncio.run(
        invoke_with_visual_stream(
            bound,
            [],
            thread_id="t1",
            callback=callback,
        )
    )

    assert response.content == "Hello"
    assert response.additional_kwargs["reasoning_content"] == "thinking"
    assert [e.phase for e in events] == ["start", "delta", "delta", "end"]
    assert events[1].content == "Hel"
    assert events[2].content == "Hello"
    assert events[2].reasoning == "thinking"


def test_invoke_with_visual_stream_empty_raises():
    bound = _FakeBound([])
    with pytest.raises(RuntimeError, match="no output"):
        asyncio.run(invoke_with_visual_stream(bound, [], thread_id="t1", callback=None))


def test_invoke_with_visual_stream_failure_clears_via_end():
    bound = _FakeBound(
        [
            AIMessageChunk(content="partial"),
            AIMessageChunk(content="more"),
        ],
        fail_after=1,
    )
    events: list[LlmStreamUpdate] = []

    with pytest.raises(RuntimeError, match="stream interrupted"):
        asyncio.run(
            invoke_with_visual_stream(
                bound,
                [],
                thread_id="t1",
                callback=events.append,
            )
        )

    assert events[-1].phase == "end"

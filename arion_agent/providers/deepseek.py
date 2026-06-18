"""DeepSeek chat model adapter.

Preserves reasoning_content across tool-calling round-trips on DeepSeek's
OpenAI-compatible API (V4 Pro / Flash). Same pattern as ChatMoonshot.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI


def _flatten_content_blocks(blocks: list) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts) if parts else ""


class ChatDeepSeek(BaseChatOpenAI):
    """DeepSeek chat model with reasoning_content round-trip support."""

    @property
    def _llm_type(self) -> str:
        return "ChatDeepSeek"

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)

        choices = getattr(response, "choices", None)
        if not choices:
            return result

        msg = choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning is None and hasattr(msg, "model_extra"):
            reasoning = (msg.model_extra or {}).get("reasoning_content")
        if reasoning is not None:
            result.generations[0].message.additional_kwargs[
                "reasoning_content"
            ] = reasoning

        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            reasoning = choices[0].get("delta", {}).get("reasoning_content")
            if reasoning is not None and isinstance(
                generation_chunk.message, AIMessageChunk
            ):
                generation_chunk.message.additional_kwargs[
                    "reasoning_content"
                ] = reasoning
        return generation_chunk

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        lc_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages", [])

        for i, p_msg in enumerate(payload_messages):
            if p_msg.get("role") != "assistant":
                continue

            lc_msg = lc_messages[i] if i < len(lc_messages) else None
            if isinstance(lc_msg, AIMessage):
                reasoning = lc_msg.additional_kwargs.get("reasoning_content", " ")
            else:
                reasoning = " "
            if not (isinstance(reasoning, str) and reasoning.strip()):
                reasoning = " "
            p_msg["reasoning_content"] = reasoning

            content = p_msg.get("content")
            if isinstance(content, list):
                p_msg["content"] = _flatten_content_blocks(content)

        return payload

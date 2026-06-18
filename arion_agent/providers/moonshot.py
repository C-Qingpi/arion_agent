"""Moonshot (Kimi) chat model adapter.

Extends BaseChatOpenAI to handle Kimi's reasoning_content field.  When
thinking mode is enabled (the default for Kimi K2.5), the API returns
reasoning_content in assistant messages and requires it to be preserved
in subsequent requests during multi-turn tool-calling flows.

ChatOpenAI drops reasoning_content because it only handles standard OpenAI
fields.  This adapter extracts it on response and re-injects it on request,
following the same pattern as langchain-deepseek's ChatDeepSeek but with
the additional round-trip preservation that tool calling demands.

Cross-provider compatibility:
When hot-switching models (e.g. Claude -> Kimi), checkpoint messages from
other providers won't have reasoning_content.  Kimi's thinking mode
requires it on ALL assistant messages, so the adapter defaults to empty
string for foreign messages.  Content in list-block format (e.g.
Anthropic's thinking/text blocks) is flattened to plain text since the
OpenAI-compatible API expects string content on assistant messages.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI


def _flatten_content_blocks(blocks: list) -> str:
    """Convert list-format content blocks to plain text.

    Handles content from providers that use structured blocks (text,
    thinking, tool_use, etc.).  Only text blocks are preserved; thinking
    and other provider-specific blocks are dropped since they're stored
    separately (e.g. in additional_kwargs) or aren't meaningful for replay.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
    return "\n".join(parts) if parts else ""


class ChatMoonshot(BaseChatOpenAI):
    """Moonshot chat model with reasoning_content round-trip support."""

    @property
    def _llm_type(self) -> str:
        return "chat-moonshot"

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

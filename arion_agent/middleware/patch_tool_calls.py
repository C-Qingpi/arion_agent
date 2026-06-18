"""Middleware to patch cross-provider message incompatibilities.

Design principles:
  - Checkpoint stores the rawest provider output (no graph-level normalization).
    Two providers may share argument names (e.g. thought/signature) with different
    semantics; normalizing at graph level would lose or conflate meaning. Auditing
    needs raw output; we optionally tag which ChatBaseClass produced each message.
  - Per-provider conversion happens only when building the next request
    (wrap_model_call). We maintain knowledge of each provider's send/receive shape
    so we can convert raw checkpoint messages into the format the current model
    expects (two-way convertibility per ChatSchema).
  - Display normalization (e.g. unified "reasoning" for the yellow bubble) is
    frontend/reader only, not in the graph.

Concrete handling:

1. Dangling tool_calls: AIMessage has tool_calls but no ToolMessage (e.g.
   interrupted). Fixed by injecting synthetic ToolMessages.

2. Orphaned ToolMessages: no preceding AIMessage with matching tool_call_id.
   Fixed by synthetic AIMessage so the send list is valid; we do not remove
   orphans from checkpoint (avoids "ID doesn't exist" conflict with compression).

3. Per-provider conversion on send: raw AIMessages from checkpoint are converted
   to the target model's expected shape (Kimi: string content + reasoning_content;
   Anthropic/Gemini/OpenAI: list content with thinking/text, no reasoning_content).
   Same-provider passthrough: when target and origin are both Anthropic (or both
   Gemini) and content is already list, the message is returned unchanged so the
   wire payload is byte-identical and Claude (and Gemini) prompt/KV cache can hit.
   Hot-switch: when target is Anthropic but origin is not, list content is built
   as above; when target is Gemini or OpenAI, thinking/signature from other
   providers are stripped. __gemini_function_call_thought_signatures__ is only
   sent when the message originated from Gemini. When building from string
   (e.g. Kimi), no thinking block is sent to Anthropic (signature required);
   reasoning is merged into a single text block.

   Gemini 3+ tool-call collapse: Gemini 3+ with thinking validates thought
   signatures on function_call parts in the "active loop." Signatures become
   stale after compression, hot-switch, or arg truncation. Rather than
   preserving them and relying on constant self-heal retries, ALL tool-call
   exchanges (foreign AND native Gemini) are collapsed into text when the
   target is Gemini. This eliminates "Corrupted thought signature" 400s.
   Self-heal still exists as a fallback for any remaining edge cases.

4. wrap_model_response: does not mutate content or additional_kwargs. Optionally
   adds response_metadata["provider"] = model class name for auditing.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from arion_agent.middleware.base import ArionMiddleware

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_VALID_TOOL_CALL_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


def _patch_dangling_tool_calls(messages: list[Any]) -> list[Any]:
    """Inject synthetic ToolMessages for any tool_call without a response.

    A tool_call is "dangling" when its response ToolMessage is not in the
    contiguous tool-response block that immediately follows the AIMessage.
    A ToolMessage much later in the history (after intervening HumanMessages
    or other AIMessages) does not satisfy the constraint — the API requires
    tool responses to appear consecutively after the assistant message that
    declared them.
    """
    if not messages:
        return messages

    patched: list[Any] = []
    for i, msg in enumerate(messages):
        patched.append(msg)
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            # Collect tool_call_ids from the contiguous ToolMessage block
            # that immediately follows this AIMessage.
            contiguous_ids: set[str] = set()
            for m in messages[i + 1:]:
                if getattr(m, "type", "") == "tool":
                    tcid = getattr(m, "tool_call_id", None)
                    if tcid:
                        contiguous_ids.add(tcid)
                else:
                    break

            for tc in msg.tool_calls:
                if tc["id"] not in contiguous_ids:
                    patched.append(
                        ToolMessage(
                            content=(
                                f"Tool call {tc['name']} (id {tc['id']}) was cancelled "
                                "- the run was interrupted before it could complete."
                            ),
                            name=tc["name"],
                            tool_call_id=tc["id"],
                        )
                    )

    return patched


def _patch_orphaned_tool_messages(messages: list[Any]) -> list[Any]:
    """Inject synthetic AIMessage for ToolMessages without a preceding AIMessage with matching tool_calls.

    Consecutive orphaned ToolMessages are batched under a single synthetic
    AIMessage so the resulting sequence is valid for LLM providers.
    """
    if not messages:
        return messages

    declared: set[str] = set()
    for msg in messages:
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                declared.add(tc["id"])

    if not any(
        getattr(msg, "type", "") == "tool"
        and getattr(msg, "tool_call_id", None)
        and getattr(msg, "tool_call_id", None) not in declared
        for msg in messages
    ):
        return messages

    patched: list[Any] = []
    orphan_batch: list[Any] = []
    seen: set[str] = set(declared)

    def _flush_orphans() -> None:
        if not orphan_batch:
            return
        tool_calls = []
        for om in orphan_batch:
            tool_calls.append({
                "id": getattr(om, "tool_call_id", None) or "unknown",
                "name": getattr(om, "name", None) or "unknown",
                "args": {},
            })
        patched.append(AIMessage(content="", tool_calls=tool_calls))
        patched.extend(orphan_batch)
        logger.warning(
            "Injected synthetic AIMessage for %d orphaned ToolMessage(s): %s",
            len(orphan_batch),
            [getattr(om, "tool_call_id", "?") for om in orphan_batch],
        )
        orphan_batch.clear()

    for msg in messages:
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                seen.add(tc["id"])

        if (
            getattr(msg, "type", "") == "tool"
            and getattr(msg, "tool_call_id", None)
            and msg.tool_call_id not in seen
        ):
            orphan_batch.append(msg)
            seen.add(msg.tool_call_id)
            continue

        _flush_orphans()
        patched.append(msg)

    _flush_orphans()
    return patched


def _find_orphaned_tool_message_ids(messages: list[Any]) -> list[str]:
    """Return checkpoint-removable IDs of ToolMessages without a parent AIMessage.

    Only returns IDs for messages with a non-None id, since RemoveMessage
    requires a valid id for checkpoint eviction.
    """
    declared: set[str] = set()
    for msg in messages:
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                declared.add(tc["id"])

    orphan_ids: list[str] = []
    for msg in messages:
        if (
            getattr(msg, "type", "") == "tool"
            and getattr(msg, "tool_call_id", None)
            and msg.tool_call_id not in declared
            and getattr(msg, "id", None) is not None
        ):
            orphan_ids.append(msg.id)

    return orphan_ids


_PROVIDER_ONLY_KWARGS = frozenset({"reasoning_content"})
# Gemini-only: must not send to other providers; when target is Gemini, only keep on messages that originated from Gemini.
_GEMINI_ONLY_KWARGS = frozenset({"__gemini_function_call_thought_signatures__"})
# Target expects list content with thinking/text blocks.
_MODELS_CONTENT_BLOCKS = frozenset({
    "ChatAnthropic",
    "ChatGoogleGenerativeAI",
    "ChatOpenAI",
})
# Anthropic: preserve thinking+signature for continuity (docs: "pass complete unmodified block back").
_CHAT_ANTHROPIC = "ChatAnthropic"
# Gemini: foreign thought/signature causes "Thought signature is not valid"; strip thinking/signature from non-Gemini messages.
_CHAT_GOOGLE_GENERATIVE_AI = "ChatGoogleGenerativeAI"
# Models that use reasoning_content in additional_kwargs (Kimi).
_MODELS_USE_REASONING_CONTENT = frozenset({"ChatMoonshot", "ChatDeepSeek"})

_THINKING_BLOCK_TYPES = frozenset({"thinking", "thought", "reasoning"})
_THINKING_TEXT_KEYS = ("thinking", "text", "thought", "reasoning")


def _thinking_text_from_block(block: dict[str, Any]) -> str:
    """Extract thinking text from a content block (per-provider keys)."""
    for key in _THINKING_TEXT_KEYS:
        val = block.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _flatten_content_to_string(content: Any) -> str:
    """Flatten list content to plain string (text blocks only). For Kimi send shape."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts) if parts else ""


def _ai_text_for_gemini_tool_collapse(msg: Any) -> str:
    """Build plain-text AI turn when collapsing tool exchanges for Gemini 3+.

    _flatten_content_to_string keeps only ``type: text`` blocks; Gemini list content
    often puts model reasoning in ``thinking`` / ``thought`` blocks. Those must be
    merged into the collapsed string so the next request still carries prior
    reasoning (structured tool_calls are removed, not the cognitive trace).
    """
    reasoning = _extract_reasoning_from_message(msg)
    text = _flatten_content_to_string(getattr(msg, "content", ""))
    parts: list[str] = []
    if isinstance(reasoning, str) and reasoning.strip():
        parts.append(reasoning.strip())
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    return "\n\n".join(parts) if parts else " "


def _extract_reasoning_from_message(msg: Any) -> str:
    """Get reasoning text from raw message: thinking blocks in content or reasoning_content in kwargs."""
    extra = getattr(msg, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "thinking", "reasoning", "thought"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in _THINKING_BLOCK_TYPES:
                text = _thinking_text_from_block(block)
                if text:
                    parts.append(text)
        if parts:
            return "\n\n".join(parts)
    return ""


def _content_blocks_safe_for_non_anthropic(content: list) -> list:
    """Keep only text and tool_use blocks; drop thinking/signature so Gemini/OpenAI do not reject.

    Anthropic uses type 'thinking' with optional 'signature'; Gemini rejects foreign thought
    signature. OpenAI does not use thinking blocks. Use when target is Gemini or OpenAI.
    """
    out: list = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        t = block.get("type")
        if t == "text":
            out.append(block)
        elif t == "tool_use":
            out.append(block)
        elif t == "server_tool_use":
            out.append(block)
    return out


def _content_blocks_safe_for_anthropic(content: list) -> list:
    """Convert foreign thinking blocks into text for Claude.

    Claude requires a valid signature on every thinking block. Thinking blocks
    from other providers (Gemini, etc.) lack this field. Merge their text into
    a text block so reasoning is preserved without triggering a 400.
    """
    out: list = []
    merged_thinking: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        t = block.get("type")
        if t in _THINKING_BLOCK_TYPES and not block.get("signature"):
            text = _thinking_text_from_block(block)
            if text:
                merged_thinking.append(text)
            continue
        out.append(block)
    if merged_thinking:
        reasoning_text = "\n\n".join(merged_thinking)
        out.insert(0, {"type": "text", "text": reasoning_text})
    return out


def _message_origin_provider(msg: Any) -> str | None:
    """Return provider class name from response_metadata (e.g. ChatAnthropic, ChatGoogleGenerativeAI)."""
    meta = getattr(msg, "response_metadata", None) or {}
    return meta.get("provider")


def _foreign_tool_call_ids(messages: list[Any]) -> set[str]:
    """Tool call ids from non-Gemini assistant messages (need signature collapse for Gemini)."""
    foreign: set[str] = set()
    for msg in messages:
        if (
            getattr(msg, "type", "") == "ai"
            and getattr(msg, "tool_calls", None)
            and _message_origin_provider(msg) != _CHAT_GOOGLE_GENERATIVE_AI
        ):
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    foreign.add(tc_id)
    return foreign


def _native_gemini_tool_call_ids(messages: list[Any]) -> set[str]:
    """All tool_call ids from ChatGoogleGenerativeAI-originated assistant messages."""
    out: set[str] = set()
    for msg in messages:
        if (
            getattr(msg, "type", "") == "ai"
            and getattr(msg, "tool_calls", None)
            and _message_origin_provider(msg) == _CHAT_GOOGLE_GENERATIVE_AI
        ):
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    out.add(tc_id)
    return out


def _convert_ai_message_for_provider(msg: Any, target_model: Any | None) -> Any:
    """Convert one raw AIMessage to the target provider's send shape. Checkpoint is not modified."""
    if getattr(msg, "type", "") != "ai":
        return msg
    model_cls = type(target_model).__name__ if target_model is not None else ""
    content = getattr(msg, "content", "")
    extra = dict(getattr(msg, "additional_kwargs", None) or {})

    if model_cls in _MODELS_USE_REASONING_CONTENT:
        # Kimi: string content + reasoning_content in additional_kwargs. API requires reasoning_content
        # on every assistant message when thinking is enabled; use non-empty placeholder if none.
        new_content = _flatten_content_to_string(content)
        reasoning = _extract_reasoning_from_message(msg)
        if not (isinstance(reasoning, str) and reasoning.strip()):
            reasoning = " "
        new_kwargs = {k: v for k, v in extra.items() if k not in _PROVIDER_ONLY_KWARGS}
        new_kwargs.pop("tool_calls", None)
        new_kwargs["reasoning_content"] = reasoning
        return AIMessage(
            content=new_content,
            tool_calls=getattr(msg, "tool_calls", []) or [],
            id=msg.id,
            response_metadata=getattr(msg, "response_metadata", {}),
            additional_kwargs=new_kwargs,
        )

    if model_cls in _MODELS_CONTENT_BLOCKS:
        # Anthropic: preserve thinking+signature when same provider. Gemini/OpenAI: strip foreign thinking/signature.
        origin = _message_origin_provider(msg)
        # Same-provider passthrough: return message unchanged so wire format is byte-identical and Claude KV cache can hit.
        if model_cls == _CHAT_ANTHROPIC and origin == _CHAT_ANTHROPIC and isinstance(content, list):
            return msg
        if model_cls == _CHAT_GOOGLE_GENERATIVE_AI and origin == _CHAT_GOOGLE_GENERATIVE_AI and isinstance(content, list):
            return msg
        strip_gemini_kwargs = (
            _GEMINI_ONLY_KWARGS
            if (model_cls != _CHAT_GOOGLE_GENERATIVE_AI or origin != _CHAT_GOOGLE_GENERATIVE_AI)
            else frozenset()
        )
        new_kwargs = {k: v for k, v in extra.items() if k not in _PROVIDER_ONLY_KWARGS and k not in strip_gemini_kwargs}
        new_kwargs.pop("tool_calls", None)
        if isinstance(content, list):
            if model_cls == _CHAT_ANTHROPIC:
                new_content = _content_blocks_safe_for_anthropic(content)
            elif model_cls == _CHAT_GOOGLE_GENERATIVE_AI and origin == _CHAT_GOOGLE_GENERATIVE_AI:
                new_content = content
            else:
                new_content = _content_blocks_safe_for_non_anthropic(content)
        else:
            # Building from string (e.g. Kimi). No signature available; never send thinking block to Anthropic.
            reasoning = extra.get("reasoning_content") or extra.get("thinking") or extra.get("reasoning") or extra.get("thought")
            text_part = content if isinstance(content, str) else str(content)
            if isinstance(reasoning, str) and reasoning.strip():
                new_content = [{"type": "text", "text": f"{reasoning.strip()}\n\n{text_part}".strip() or " "}]
            else:
                new_content = [{"type": "text", "text": text_part or " "}]
        return AIMessage(
            content=new_content,
            tool_calls=getattr(msg, "tool_calls", []) or [],
            id=msg.id,
            response_metadata=getattr(msg, "response_metadata", {}),
            additional_kwargs=new_kwargs,
        )

    # Unknown target: strip provider-only kwargs and raw tool_calls to avoid 400s.
    new_kwargs = {k: v for k, v in extra.items() if k not in _PROVIDER_ONLY_KWARGS}
    new_kwargs.pop("tool_calls", None)
    if new_kwargs == extra:
        return msg
    return AIMessage(
        content=content,
        tool_calls=getattr(msg, "tool_calls", []) or [],
        id=msg.id,
        response_metadata=getattr(msg, "response_metadata", {}),
        additional_kwargs=new_kwargs,
    )


def _collapse_tool_exchanges_for_gemini_thought_signature(
    messages: list[Any],
    model: Any | None,
    *,
    recover_from_thought_signature_error: bool = False,
) -> list[Any]:
    """Collapse tool-call exchanges that cannot be sent safely to Gemini 3+.

    All tool exchanges (foreign AND native Gemini) are collapsed to text.
    Thought signatures become stale after compression, hot-switch, or arg
    truncation. Foreign tool calls lack valid signatures entirely (LangChain
    injects DUMMY which Gemini 3.1+ rejects). Collapsing everything
    eliminates constant "Corrupted thought signature" 400 errors.

    Replaces each affected AIMessage+ToolMessage* group with a text-only
    AIMessage plus a HumanMessage carrying tool results. The collapsed AI
    string includes thinking/thought blocks (see ``_ai_text_for_gemini_tool_collapse``)
    so prior reasoning is not lost when structured tool_calls are removed.
    """
    collapse_ids = _foreign_tool_call_ids(messages) | _native_gemini_tool_call_ids(messages)
    if not collapse_ids:
        return messages

    if recover_from_thought_signature_error:
        logger.info(
            "Self-heal: collapsing %d tool-call id(s) after thought-signature error",
            len(collapse_ids),
        )
    else:
        logger.info(
            "Collapsing %d tool-call id(s) for Gemini thought-signature compat",
            len(collapse_ids),
        )

    result: list[Any] = []
    tool_batch: list[Any] = []

    def _flush_tools() -> None:
        if not tool_batch:
            return
        parts: list[str] = []
        for tm in tool_batch:
            name = getattr(tm, "name", None) or "tool"
            content = getattr(tm, "content", "")
            parts.append(f"[{name}] {content}")
        result.append(HumanMessage(content="\n".join(parts)))
        tool_batch.clear()

    for msg in messages:
        if (
            getattr(msg, "type", "") == "tool"
            and getattr(msg, "tool_call_id", "") in collapse_ids
        ):
            tool_batch.append(msg)
            continue

        _flush_tools()

        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            ids = [tc.get("id") for tc in msg.tool_calls if tc.get("id")]
            if ids and all(tid in collapse_ids for tid in ids):
                text = _ai_text_for_gemini_tool_collapse(msg)
                result.append(AIMessage(
                    content=text,
                    id=msg.id,
                    response_metadata=getattr(msg, "response_metadata", {}),
                ))
                continue

        result.append(msg)

    _flush_tools()
    return result


def _convert_messages_for_provider(messages: list[Any], model: Any | None) -> list[Any]:
    """Convert raw checkpoint AIMessages to the target model's send shape. Only for the copy sent to the API."""
    if not messages or model is None:
        return messages
    result: list[Any] = []
    for m in messages:
        result.append(_convert_ai_message_for_provider(m, model))

    model_cls = type(model).__name__ if model is not None else ""
    if model_cls == _CHAT_GOOGLE_GENERATIVE_AI:
        result = _collapse_tool_exchanges_for_gemini_thought_signature(result, model)

    return result


def _sanitize_tool_call_ids(messages: list[Any]) -> list[Any]:
    """Replace characters in tool_call_ids that strict providers reject.

    Anthropic requires tool_call_ids to match ``^[a-zA-Z0-9_-]+$``.
    Some providers (e.g. Moonshot/Kimi) generate IDs containing dots or
    other characters.  Builds a new list with updated IDs so we do not
    mutate checkpoint message refs (preserves KV cache passthrough).
    """
    remap: dict[str, str] = {}
    for msg in messages:
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                old_id = tc.get("id", "")
                if old_id and not _VALID_TOOL_CALL_ID.match(old_id):
                    new_id = re.sub(r"[^a-zA-Z0-9_-]", "_", old_id)
                    remap[old_id] = new_id
        if getattr(msg, "type", "") == "tool":
            old_id = getattr(msg, "tool_call_id", None)
            if old_id and old_id not in remap and not _VALID_TOOL_CALL_ID.match(old_id):
                remap[old_id] = re.sub(r"[^a-zA-Z0-9_-]", "_", old_id)

    if not remap:
        return messages

    logger.debug("Sanitized %d tool_call_id(s): %s", len(remap), list(remap))
    result: list[Any] = []
    for msg in messages:
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            tool_calls = msg.tool_calls
            if any(tc.get("id") in remap for tc in tool_calls):
                new_tool_calls = [
                    {**tc, "id": remap.get(tc.get("id", ""), tc.get("id", ""))}
                    for tc in tool_calls
                ]
                ak = dict(getattr(msg, "additional_kwargs", None) or {})
                ak.pop("tool_calls", None)
                result.append(AIMessage(
                    content=msg.content,
                    tool_calls=new_tool_calls,
                    id=getattr(msg, "id", None),
                    response_metadata=getattr(msg, "response_metadata", {}),
                    additional_kwargs=ak,
                ))
            else:
                result.append(msg)
        elif getattr(msg, "type", "") == "tool":
            old_id = getattr(msg, "tool_call_id", None)
            if old_id in remap:
                result.append(ToolMessage(
                    content=msg.content,
                    tool_call_id=remap[old_id],
                    name=getattr(msg, "name", None),
                ))
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


class PatchToolCallsMiddleware(ArionMiddleware):
    """Ensure every tool_call has a matching ToolMessage and vice versa.

    Orphaned ToolMessages are patched in-memory for the current LLM call;
    we do not remove them from the checkpoint (avoids conflict with compression).
    """

    def __init__(self) -> None:
        self._pending_removals: list[Any] = []

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        patched = _patch_dangling_tool_calls(messages)
        patched = _patch_orphaned_tool_messages(patched)
        if len(patched) != len(messages):
            return {"messages": patched}
        return None

    def wrap_model_call(
        self,
        messages: list[Any],
        tools: list[BaseTool],
        **kwargs: Any,
    ) -> tuple[list[Any], list[BaseTool], dict[str, Any]]:
        """Patch dangling tool calls and orphaned tool messages.

        Injects synthetic AIMessage for orphaned ToolMessages so the send list is valid.
        We do not queue RemoveMessage for orphans (would conflict with compression evictions
        and cause 'ID doesn't exist' in the reducer); orphans remain in the checkpoint.
        """
        orphan_ids = _find_orphaned_tool_message_ids(messages)
        # Do not queue RemoveMessage for orphans: compression may have already evicted the same
        # messages; the reducer then errors "Attempting to delete a message with an ID that
        # doesn't exist". We still patch the send list (synthetic AIMessage + orphans) so the
        # API gets a valid sequence; orphans remain in the checkpoint.
        self._pending_removals = []
        if orphan_ids:
            logger.info(
                "Orphaned ToolMessage(s) (send list patched; not removing from checkpoint): %s",
                orphan_ids,
            )

        patched = _patch_dangling_tool_calls(messages)
        patched = _patch_orphaned_tool_messages(patched)
        patched = _convert_messages_for_provider(patched, kwargs.get("model"))
        patched = _sanitize_tool_call_ids(patched)
        return patched, tools, kwargs

    def wrap_model_response(self, response: Any, **kwargs: Any) -> Any:
        """Preserve raw response; optionally tag provider for auditing. No content normalization."""
        model = kwargs.get("model")
        if model is not None and hasattr(response, "response_metadata"):
            meta = getattr(response, "response_metadata", None) or {}
            if isinstance(meta, dict) and "provider" not in meta:
                response.response_metadata = {**meta, "provider": type(model).__name__}
        return response

    def drain_state_updates(self) -> list[Any]:
        """Return pending state updates (we no longer emit RemoveMessage for orphans)."""
        removals = self._pending_removals
        self._pending_removals = []
        return removals

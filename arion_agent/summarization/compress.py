"""Compression graph node: replaces SummarizationMiddleware.

Compression is a conditional graph node that evicts old messages and
produces a summary. The summary is stored as a first-class state channel,
checkpointed natively by LangGraph. Crash recovery is automatic via
checkpoint resume -- no JSONL, no stale detection.

Two-tier routing keeps summarization unfeelable:
  - should_compress (prefetch): headroom trigger; background LLM, then proactive apply
  - must_compress (compress): hard trigger; sync fallback if prefetch did not keep count down

Factory functions:
  - make_should_compress: headroom conditional edge (route to prefetch)
  - make_must_compress: hard conditional edge (route to compress)
  - make_route_compression: combined router for graph edges
  - make_prefetch_node: fire-and-forget background summarization
  - make_compress_node: mandatory compression with eviction
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    get_buffer_string,
)

from arion_agent.summarization.config import (
    PolicyDecision,
    SummarizationCallback,
    SummarizationEvent,
    SummarizationPolicy,
)
from arion_agent.util.tokens import estimate_message_tokens, estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_BUDGET = 6000
DEFAULT_SUMMARY_BUDGET_FRACTION = 0.60
DEFAULT_ARG_TRUNCATION_TRIGGER = 50
DEFAULT_ARG_TRUNCATION_KEEP = 20
DEFAULT_ARG_MAX_LENGTH = 2000
DEFAULT_PREFETCH_HEADROOM_FRACTION = 0.80


# ---------------------------------------------------------------------------
# Policy evaluation (extracted from SummarizationMiddleware)
# ---------------------------------------------------------------------------


def evaluate_policy(
    policy: SummarizationPolicy | Callable[..., PolicyDecision | None],
    messages: list[AnyMessage],
    token_count: int,
    max_tokens: int | None,
) -> PolicyDecision | None:
    """Check whether compression should trigger. Returns PolicyDecision or None."""
    if callable(policy) and not isinstance(policy, SummarizationPolicy):
        return policy(messages, token_count, max_tokens)

    if not isinstance(policy, SummarizationPolicy):
        return None

    triggered = False
    if policy.trigger_messages is not None and len(messages) > policy.trigger_messages:
        triggered = True
    if policy.trigger_tokens is not None and token_count > policy.trigger_tokens:
        triggered = True
    if (
        policy.trigger_fraction is not None
        and max_tokens
        and token_count > max_tokens * policy.trigger_fraction
    ):
        triggered = True

    if not triggered:
        return None

    if policy.keep_messages is not None:
        return PolicyDecision(keep_last_messages=policy.keep_messages)
    if policy.keep_tokens is not None:
        return PolicyDecision(keep_last_tokens=policy.keep_tokens)
    if policy.keep_fraction is not None:
        return PolicyDecision(keep_last_fraction=policy.keep_fraction)
    return PolicyDecision(keep_last_messages=20)


def evaluate_prefetch_policy(
    policy: SummarizationPolicy | Callable[..., PolicyDecision | None],
    messages: list[AnyMessage],
    token_count: int,
    max_tokens: int | None,
    *,
    prefetch_policy: Callable[..., PolicyDecision | None] | None = None,
) -> PolicyDecision | None:
    """Check whether background prefetch should start. Returns None if must already applies."""
    if evaluate_policy(policy, messages, token_count, max_tokens) is not None:
        return None

    if prefetch_policy is not None:
        return prefetch_policy(messages, token_count, max_tokens)

    if callable(policy) and not isinstance(policy, SummarizationPolicy):
        from arion_agent.summarization.policies import STANDARD_POLICY, STANDARD_PREFETCH_POLICY

        if policy is STANDARD_POLICY:
            return STANDARD_PREFETCH_POLICY(messages, token_count, max_tokens)
        return None

    if not isinstance(policy, SummarizationPolicy):
        return None

    triggered = False
    prefetch_messages = policy.prefetch_messages
    if prefetch_messages is None and policy.trigger_messages is not None:
        prefetch_messages = int(policy.trigger_messages * DEFAULT_PREFETCH_HEADROOM_FRACTION)
    if prefetch_messages is not None and len(messages) > prefetch_messages:
        triggered = True

    prefetch_tokens = policy.prefetch_tokens
    if prefetch_tokens is None and policy.trigger_tokens is not None:
        prefetch_tokens = int(policy.trigger_tokens * DEFAULT_PREFETCH_HEADROOM_FRACTION)
    if prefetch_tokens is not None and token_count > prefetch_tokens:
        triggered = True

    prefetch_fraction = policy.prefetch_fraction
    if prefetch_fraction is None and policy.trigger_fraction is not None:
        prefetch_fraction = policy.trigger_fraction * DEFAULT_PREFETCH_HEADROOM_FRACTION
    if (
        prefetch_fraction is not None
        and max_tokens
        and token_count > max_tokens * prefetch_fraction
    ):
        triggered = True

    if not triggered:
        return None

    if policy.keep_messages is not None:
        return PolicyDecision(keep_last_messages=policy.keep_messages)
    if policy.keep_tokens is not None:
        return PolicyDecision(keep_last_tokens=policy.keep_tokens)
    if policy.keep_fraction is not None:
        return PolicyDecision(keep_last_fraction=policy.keep_fraction)
    return PolicyDecision(keep_last_messages=20)


def _estimate_messages_tokens(messages: list[AnyMessage]) -> int:
    """Estimate total token count for a list of messages."""
    total = 0
    for m in messages:
        total += estimate_message_tokens(m.content)
    return total


# ---------------------------------------------------------------------------
# Safe cutoff (extracted from SummarizationMiddleware)
# ---------------------------------------------------------------------------


def _raw_cutoff(
    messages: list[AnyMessage],
    decision: PolicyDecision,
    max_tokens: int | None,
) -> int:
    """Compute raw cutoff index from policy decision."""
    if decision.keep_last_messages is not None:
        return max(0, len(messages) - decision.keep_last_messages)

    if decision.keep_last_tokens is not None:
        target = decision.keep_last_tokens
        tokens_kept = 0
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_message_tokens(messages[i].content)
            if tokens_kept + msg_tokens > target:
                return i + 1
            tokens_kept += msg_tokens
        return 0

    if decision.keep_last_fraction is not None and max_tokens:
        target_tokens = int(max_tokens * decision.keep_last_fraction)
        tokens_kept = 0
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_message_tokens(messages[i].content)
            if tokens_kept + msg_tokens > target_tokens:
                return i + 1
            tokens_kept += msg_tokens
        return 0

    return max(0, len(messages) - 20)


def _walk_back_past_tool_messages(messages: list[AnyMessage], pos: int) -> int:
    """Walk backward from pos past any ToolMessages."""
    while pos > 0 and isinstance(messages[pos], ToolMessage):
        pos -= 1
    return pos


raw_cutoff = _raw_cutoff
walk_back_past_tool_messages = _walk_back_past_tool_messages


def find_safe_cutoff(
    messages: list[AnyMessage],
    decision: PolicyDecision,
    max_tokens: int | None,
) -> int:
    """Find cutoff index aligned to a safe conversation boundary.

    Messages before cutoff are summarized, at/after are kept.
    Never splits an AI tool_call from its ToolMessage responses.
    """
    raw = _raw_cutoff(messages, decision, max_tokens)
    if raw <= 0:
        return 0
    if raw >= len(messages):
        return len(messages)

    cutoff = _walk_back_past_tool_messages(messages, raw)

    return max(1, cutoff)


def find_kept_orphans(
    evicted: list[AnyMessage],
    kept: list[AnyMessage],
) -> list[AnyMessage]:
    """Find ToolMessages in kept whose parent AIMessage is being evicted."""
    evicted_tc_ids: set[str] = set()
    for m in evicted:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                evicted_tc_ids.add(tc["id"])

    if not evicted_tc_ids:
        return []

    kept_tc_ids: set[str] = set()
    for m in kept:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                kept_tc_ids.add(tc["id"])

    lost_tc_ids = evicted_tc_ids - kept_tc_ids
    if not lost_tc_ids:
        return []

    return [
        m for m in kept
        if isinstance(m, ToolMessage) and m.tool_call_id in lost_tc_ids
    ]


# ---------------------------------------------------------------------------
# Transcript file writing (extracted from SummarizationMiddleware._offload_to_file)
# ---------------------------------------------------------------------------


def write_transcript(
    messages: list[AnyMessage],
    thread_id: str,
    *,
    history_dir: Path,
    workspace_dir: Path | None = None,
) -> str | None:
    """Write evicted messages to a JSONL transcript file.

    Returns the workspace-relative directory path, or None on failure.
    """
    from arion_agent.util.persistence import ensure_directory, file_exists, glob_files, append_jsonl

    thread_dir = history_dir / "conversation_history" / thread_id
    ensure_directory(thread_dir)

    if not messages:
        if workspace_dir is not None:
            from arion_agent.util.persistence import workspace_relative_path
            return workspace_relative_path(thread_dir, workspace_dir)
        return str(thread_dir)

    existing = [p for p in glob_files(thread_dir, "*.jsonl")]
    event_number = len(existing) + 1
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    participants: list[dict[str, str]] = []
    human_count = 0
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "human"
            human_count += 1
        elif isinstance(m, AIMessage):
            role = "ai"
        elif isinstance(m, ToolMessage):
            role = "tool"
        else:
            role = type(m).__name__.lower().replace("message", "")
        text = m.content if isinstance(m.content, str) else str(m.content)
        participants.append({"role": role, "content": text})

    record: dict[str, Any] = {
        "event": event_number,
        "ts_utc": timestamp,
        "msg_count": len(messages),
        "human_count": human_count,
        "participants": participants,
    }

    file_name = f"{timestamp.replace(':', '-')}.jsonl"
    file_path = thread_dir / file_name
    if file_exists(file_path):
        file_name = f"{timestamp.replace(':', '-')}_{event_number}.jsonl"
        file_path = thread_dir / file_name

    try:
        append_jsonl(file_path, record)
        logger.info("Wrote transcript: %s (%d messages, %d human)", file_path, len(messages), human_count)
    except OSError:
        logger.exception("Failed to write transcript to %s", file_path)
        return None

    if workspace_dir is not None:
        from arion_agent.util.persistence import workspace_relative_path
        return workspace_relative_path(thread_dir, workspace_dir)
    return str(thread_dir)


# ---------------------------------------------------------------------------
# Argument truncation (extracted from SummarizationMiddleware._truncate_args)
# ---------------------------------------------------------------------------


def truncate_args(
    messages: list[AnyMessage],
    *,
    trigger: int = DEFAULT_ARG_TRUNCATION_TRIGGER,
    keep: int = DEFAULT_ARG_TRUNCATION_KEEP,
    max_length: int = DEFAULT_ARG_MAX_LENGTH,
) -> list[AnyMessage]:
    """Truncate large write_file/str_replace args in old messages.

    Only modifies messages older than (len - keep) when total count exceeds
    trigger. Reduces token usage from verbose file content in history.
    """
    if len(messages) < trigger:
        return messages

    cutoff = max(0, len(messages) - keep)
    result: list[AnyMessage] = []

    for i, msg in enumerate(messages):
        if i < cutoff and isinstance(msg, AIMessage) and msg.tool_calls:
            modified = False
            new_tool_calls = []
            for tc in msg.tool_calls:
                if tc["name"] in {"write_file", "str_replace"}:
                    args = tc.get("args", {})
                    new_args = {}
                    tc_modified = False
                    for key, value in args.items():
                        if isinstance(value, str) and len(value) > max_length:
                            new_args[key] = value[:max_length] + "..."
                            tc_modified = True
                        else:
                            new_args[key] = value
                    if tc_modified:
                        new_tool_calls.append({**tc, "args": new_args})
                        modified = True
                    else:
                        new_tool_calls.append(tc)
                else:
                    new_tool_calls.append(tc)
            if modified:
                truncated_msg = AIMessage(
                    content=msg.content,
                    tool_calls=new_tool_calls,
                    id=msg.id,
                    response_metadata=getattr(msg, "response_metadata", {}),
                    additional_kwargs=getattr(msg, "additional_kwargs", {}),
                )
                result.append(truncated_msg)
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


# ---------------------------------------------------------------------------
# Prefetch registry (in-memory background summarization tasks)
# ---------------------------------------------------------------------------


@dataclass
class _PrefetchResult:
    cutoff: int
    message_count: int
    summary_wrapper: str
    summary_raw_text: str
    evictions: list[Any]
    messages_summarized: int
    messages_kept: int
    summary_tokens: int
    file_path: str | None


@dataclass
class _PrefetchEntry:
    message_count: int
    cutoff: int
    task: asyncio.Task[_PrefetchResult]


class PrefetchRegistry:
    """Per-thread background summarization tasks keyed by thread_id."""

    def __init__(self) -> None:
        self._entries: dict[str, _PrefetchEntry] = {}

    def get(self, thread_id: str) -> _PrefetchEntry | None:
        return self._entries.get(thread_id)

    def clear(self, thread_id: str) -> None:
        entry = self._entries.pop(thread_id, None)
        if entry is not None and not entry.task.done():
            entry.task.cancel()

    def start(
        self,
        thread_id: str,
        *,
        message_count: int,
        cutoff: int,
        coro_factory: Callable[[], Any],
    ) -> bool:
        """Start a background summary task. Returns True if a new task was created,
        False if an equivalent or newer in-flight task already covers this cutoff."""
        entry = self._entries.get(thread_id)
        if entry is not None:
            if entry.message_count == message_count and entry.cutoff == cutoff:
                return False
            if not entry.task.done() and entry.cutoff <= cutoff:
                return False
            if not entry.task.done():
                entry.task.cancel()

        task = asyncio.create_task(coro_factory())
        self._entries[thread_id] = _PrefetchEntry(
            message_count=message_count,
            cutoff=cutoff,
            task=task,
        )
        return True

    def take_if_ready(self, thread_id: str) -> _PrefetchResult | None:
        """Non-blocking: pop and return the result if the background task has
        finished. Returns None while still running, or if it was cancelled
        (cancellation is our own doing when a cutoff regresses, not an error).
        A genuinely failed task re-raises here so the error is exposed."""
        entry = self._entries.get(thread_id)
        if entry is None or not entry.task.done():
            return None
        self._entries.pop(thread_id, None)
        if entry.task.cancelled():
            return None
        return entry.task.result()

    async def await_result(
        self,
        thread_id: str,
        *,
        cutoff: int,
    ) -> _PrefetchResult | None:
        entry = self._entries.pop(thread_id, None)
        if entry is None:
            return None
        if entry.cutoff > cutoff:
            if not entry.task.done():
                entry.task.cancel()
            return None
        if entry.task.cancelled():
            return None
        return await entry.task


# ---------------------------------------------------------------------------
# Shared compression internals
# ---------------------------------------------------------------------------


def _plan_compression(
    messages: list[AnyMessage],
    decision: PolicyDecision,
    *,
    cutoff_fn: Callable[..., int],
    max_tokens: int | None,
) -> tuple[int, list[AnyMessage], list[AnyMessage]] | None:
    cutoff = cutoff_fn(messages, decision, max_tokens)
    if cutoff <= 0:
        return None

    messages_to_evict = messages[:cutoff]
    messages_to_keep = messages[cutoff:]

    orphans = find_kept_orphans(messages_to_evict, messages_to_keep)
    if orphans:
        orphan_ids = {id(m) for m in orphans}
        messages_to_keep = [m for m in messages_to_keep if id(m) not in orphan_ids]
        messages_to_evict = messages_to_evict + orphans

    return cutoff, messages_to_evict, messages_to_keep


def _value_to_natural(v: Any, max_str: int = 120) -> str:
    """Convert an arbitrary JSON value to natural language prose.

    No brackets, braces, quotes or JSON-like syntax — only plain ``key is value``.
    Nested structures are flattened to a depth of 1; long strings truncated.
    """
    if v is None:
        return "nothing"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if len(v) > max_str:
            return v[:max_str] + "..."
        return v
    if isinstance(v, (list, tuple)):
        if not v:
            return "nothing"
        items = [_value_to_natural(x, max_str) for x in v[:3]]
        tail = f", and {len(v)-3} more" if len(v) > 3 else ""
        return ", ".join(items) + tail
    if isinstance(v, dict):
        parts = []
        for k, val in v.items():
            parts.append(f"{k} is {_value_to_natural(val, max_str)}")
        if len(parts) <= 4:
            return "; ".join(parts)
        return "; ".join(parts[:4]) + f"; and {len(parts)-4} more fields..."
    return str(v)[:max_str]


def _sanitize_messages_for_summary(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Replace raw tool call JSON with natural-language labels.

    Raw tool call blocks (e.g. ``[{'name': 'shell_run', ...}]``) trigger
    agentic continuation in the summary model.  We rewrite them using a
    general ``key is value`` form that preserves all information while
    removing every trace of JSON syntax::

        Agent called tool shell_run: command is kill ..., cwd is final_exam_standalone

    The conversion is schema-free — any tool name and any argument shape
    are handled automatically.  Existing text content on the message is preserved.
    """
    sanitized: list[AnyMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            sanitized.append(msg)
            continue
        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            sanitized.append(msg)
            continue
        content_parts: list[str] = []
        text_content = str(msg.content) if msg.content else ""
        if text_content.strip():
            content_parts.append(text_content.strip())
        for tc in tcs:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            if args:
                arg_parts = []
                for k, v in args.items():
                    arg_parts.append(f"{k} is {_value_to_natural(v)}")
                content_parts.append(f"Agent called tool {name}: {'; '.join(arg_parts)}")
            else:
                content_parts.append(f"Agent called tool {name}")
        if content_parts:
            sanitized.append(AIMessage(content="\n".join(content_parts)))
        else:
            sanitized.append(msg)
    return sanitized


def _build_summary_prompt(
    *,
    messages_to_evict: list[AnyMessage],
    previous_summary_raw: str,
    summary_prompt: str,
    summary_budget: int,
    history_dir: Path | None,
    workspace_dir: Path | None,
    effective_max_length: int,
) -> str:
    evict_for_summary = truncate_args(
        messages_to_evict,
        trigger=0,
        keep=0,
        max_length=effective_max_length,
    )
    evict_for_summary = _sanitize_messages_for_summary(evict_for_summary)
    formatted_messages = get_buffer_string(evict_for_summary)

    from arion_agent.summarization.sections import (
        build_supplemental_sections,
        format_configured_skills,
    )

    configured_skills = format_configured_skills(
        identity_dir=history_dir,
        workspace_dir=workspace_dir,
    )
    optional_sections = build_supplemental_sections()

    if "{messages}" in summary_prompt:
        prompt_text = summary_prompt.format(
            messages=formatted_messages,
            budget=summary_budget,
            configured_skills=configured_skills,
            optional_sections=optional_sections,
        )
    else:
        prompt_text = (
            f"{summary_prompt}\n\n"
            f"Budget: approximately {summary_budget} tokens.\n\n"
            f"<messages>\n{formatted_messages}\n</messages>"
        )

    if previous_summary_raw:
        prompt_text = (
            f"Previous summary (essential content only; ignore wrapper formatting):\n"
            f"{previous_summary_raw}\n\n"
            f"{prompt_text}"
        )
    prompt_text += (
        "\n\nCRITICAL: You are a ONE-SHOT SUMMARIZER, not an agent. "
        "Do NOT emit tool calls. Do NOT continue the conversation. "
        "Only produce the specified structured summary format."
    )
    return prompt_text


_REQUIRED_SECTION_MARKERS = [
    "# BACKGROUND",
    "# HISTORY",
    "# NEXT STEPS",
    "# WORKSPACE",
    "## Session Context",
    "## Active Work & Status",
    "## Open Items",
    "## Immediate Next Steps",
]


def _validate_summary_structure(summary_text: str, min_sections: int = 3) -> bool:
    """Check that the LLM summary contains enough required template sections."""
    count = sum(1 for marker in _REQUIRED_SECTION_MARKERS if marker in summary_text)
    return count >= min_sections


async def _invoke_summary_model(
    summary_model: BaseChatModel,
    prompt_text: str,
    *,
    validate_structure: bool = True,
    max_retries: int = 3,
) -> str:
    response = await summary_model.ainvoke(prompt_text)
    raw = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw, list):
        raw = "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
        )
    summary_text = str(raw)

    if not validate_structure:
        return summary_text

    retry_hint = (
        "\n\n---\n\n"
        "CRITICAL: Your previous response did NOT follow the required structure. "
        "You MUST begin with '# BACKGROUND' and include ALL specified sections: "
        "# BACKGROUND, # HISTORY, # NEXT STEPS, # WORKSPACE. "
        "This is a structured template — do NOT output raw conversation data."
    )

    for attempt in range(max_retries):
        if _validate_summary_structure(summary_text):
            return summary_text
        logger.warning(
            "Summary structure validation failed (attempt %d/%d), retrying...",
            attempt + 1, max_retries,
        )
        retry_prompt = prompt_text + retry_hint
        response = await summary_model.ainvoke(retry_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        if isinstance(raw, list):
            raw = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
            )
        summary_text = str(raw)

    return summary_text


def _wrap_summary(
    summary_text: str,
    file_path: str | None,
    *,
    wrapper_with_path: str,
    wrapper_no_path: str,
) -> str:
    if file_path:
        return wrapper_with_path.format(summary=summary_text, file_path=file_path)
    return wrapper_no_path.format(summary=summary_text)


def _build_evictions(messages_to_evict: list[AnyMessage]) -> list[Any]:
    evictions: list[Any] = []
    for m in messages_to_evict:
        if getattr(m, "id", None):
            evictions.append(RemoveMessage(id=m.id))
    return evictions


def _resolve_prompt_wrappers(is_perpetual: bool) -> tuple[str, str, str]:
    from arion_agent.summarization.prompts import (
        PERPETUAL_SUMMARY_PROMPT,
        PERPETUAL_WRAPPER,
        PERPETUAL_WRAPPER_NO_PATH,
        TASK_SUMMARY_PROMPT,
        TASK_WRAPPER,
        TASK_WRAPPER_NO_PATH,
    )

    if is_perpetual:
        return PERPETUAL_SUMMARY_PROMPT, PERPETUAL_WRAPPER, PERPETUAL_WRAPPER_NO_PATH
    return TASK_SUMMARY_PROMPT, TASK_WRAPPER, TASK_WRAPPER_NO_PATH


# ---------------------------------------------------------------------------
# Factory: should_compress (headroom prefetch trigger)
# ---------------------------------------------------------------------------


def make_should_compress(
    policy: SummarizationPolicy | Callable[..., PolicyDecision | None],
    max_tokens: int | None = None,
    *,
    prefetch_policy: Callable[..., PolicyDecision | None] | None = None,
) -> Callable[..., str]:
    """Create the headroom conditional edge: route to prefetch or model."""

    def should_compress(state: dict[str, Any]) -> str:
        messages = state.get("messages", [])
        if len(messages) < 5:
            return "model"
        token_count = _estimate_messages_tokens(messages)
        decision = evaluate_prefetch_policy(
            policy, messages, token_count, max_tokens,
            prefetch_policy=prefetch_policy,
        )
        if decision is not None:
            return "prefetch"
        return "model"

    return should_compress


# ---------------------------------------------------------------------------
# Factory: must_compress (hard sequential trigger)
# ---------------------------------------------------------------------------


def make_must_compress(
    policy: SummarizationPolicy | Callable[..., PolicyDecision | None],
    max_tokens: int | None = None,
) -> Callable[..., str]:
    """Create the hard conditional edge: route to compress or model."""

    def must_compress(state: dict[str, Any]) -> str:
        messages = state.get("messages", [])
        if len(messages) < 5:
            return "model"
        token_count = _estimate_messages_tokens(messages)
        decision = evaluate_policy(policy, messages, token_count, max_tokens)
        if decision is not None:
            return "compress"
        return "model"

    return must_compress


def make_route_compression(
    policy: SummarizationPolicy | Callable[..., PolicyDecision | None],
    max_tokens: int | None = None,
    *,
    prefetch_policy: Callable[..., PolicyDecision | None] | None = None,
) -> Callable[..., str]:
    """Combined router: must compress wins over prefetch."""

    must = make_must_compress(policy, max_tokens)
    should = make_should_compress(
        policy, max_tokens, prefetch_policy=prefetch_policy,
    )

    def route_compression(state: dict[str, Any]) -> str:
        destination = must(state)
        if destination == "compress":
            return "compress"
        return should(state)

    return route_compression


# ---------------------------------------------------------------------------
# Factory: prefetch_node (background summarization, no eviction)
# ---------------------------------------------------------------------------


def make_prefetch_node(
    policy: SummarizationPolicy | Callable[..., PolicyDecision | None],
    summary_model: BaseChatModel,
    registry: PrefetchRegistry,
    *,
    summary_prompt: str | None = None,
    summary_budget: int = DEFAULT_SUMMARY_BUDGET,
    history_dir: Path | None = None,
    workspace_dir: Path | None = None,
    max_tokens: int | None = None,
    is_perpetual: bool = False,
    abort_check: Callable[[], bool] | None = None,
    cutoff_fn: Any | None = None,
    arg_max_length: int | None = None,
    prefetch_policy: Callable[..., PolicyDecision | None] | None = None,
    on_event: SummarizationCallback | None = None,
) -> Callable[..., Any]:
    """Start background summarization when headroom threshold is reached, and
    apply a completed background summary as soon as it is ready.

    Applying in the headroom zone evicts old messages proactively (no blocking
    LLM call on the agent's turn), so the hard compress trigger is rarely hit.
    The compress node remains as the synchronous safety net."""
    effective_cutoff_fn = cutoff_fn or find_safe_cutoff
    effective_max_length = arg_max_length if arg_max_length is not None else DEFAULT_ARG_MAX_LENGTH
    default_prompt, wrapper_with_path, wrapper_no_path = _resolve_prompt_wrappers(is_perpetual)
    if summary_prompt is None:
        summary_prompt = default_prompt

    async def _run_prefetch(
        *,
        thread_id: str,
        messages: list[AnyMessage],
        previous_summary_raw: str,
        decision: PolicyDecision,
        cutoff: int,
        messages_to_evict: list[AnyMessage],
        messages_to_keep: list[AnyMessage],
    ) -> _PrefetchResult:
        file_path: str | None = None
        if history_dir is not None:
            file_path = write_transcript(
                messages_to_evict, thread_id,
                history_dir=history_dir,
                workspace_dir=workspace_dir,
            )

        prompt_text = _build_summary_prompt(
            messages_to_evict=messages_to_evict,
            previous_summary_raw=previous_summary_raw,
            summary_prompt=summary_prompt,
            summary_budget=summary_budget,
            history_dir=history_dir,
            workspace_dir=workspace_dir,
            effective_max_length=effective_max_length,
        )

        summary_text = await _invoke_summary_model(summary_model, prompt_text)
        summary_wrapper = _wrap_summary(
            summary_text, file_path,
            wrapper_with_path=wrapper_with_path,
            wrapper_no_path=wrapper_no_path,
        )
        summary_tokens = estimate_tokens(summary_text)

        logger.info(
            "Prefetch complete: %d messages summarized, kept %d, summary ~%d tokens",
            len(messages_to_evict),
            len(messages_to_keep),
            summary_tokens,
        )

        return _PrefetchResult(
            cutoff=cutoff,
            message_count=len(messages),
            summary_wrapper=summary_wrapper,
            summary_raw_text=summary_text,
            evictions=_build_evictions(messages_to_evict),
            messages_summarized=len(messages_to_evict),
            messages_kept=len(messages_to_keep),
            summary_tokens=summary_tokens,
            file_path=file_path,
        )

    async def prefetch_node(
        state: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        messages = state.get("messages", [])
        previous_summary_raw = state.get("summary_raw", "")

        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = configurable.get("thread_id", "default")

        if abort_check is not None and abort_check():
            return {}

        ready = registry.take_if_ready(thread_id)
        if ready is not None:
            # Guard: filter stale evictions that reference messages already
            # removed from the state (e.g. by a hard compression that ran after
            # the prefetch snapshot was taken but before take_if_ready).
            current_ids = {m.id for m in messages if getattr(m, "id", None)}
            valid = [r for r in ready.evictions if getattr(r, "id", None) in current_ids]
            if len(valid) != len(ready.evictions):
                stale = len(ready.evictions) - len(valid)
                logger.warning(
                    "Prefetch apply: %d of %d evictions are stale (messages already "
                    "gone — likely evicted by hard compression). Skipping stale refs.",
                    stale, len(ready.evictions),
                )
            logger.info(
                "Prefetch apply: evicting %d messages, kept %d (proactive, no block)",
                len(valid), ready.messages_kept,
            )
            if on_event is not None:
                on_event(SummarizationEvent(
                    phase="after",
                    cutoff_index=ready.cutoff,
                    messages_summarized=ready.messages_summarized,
                    messages_kept=ready.messages_kept,
                    summary_tokens=ready.summary_tokens,
                    file_path=ready.file_path,
                    thread_id=thread_id,
                ))
            return {
                "summary": ready.summary_wrapper,
                "summary_raw": ready.summary_raw_text,
                "messages": valid,
            }

        token_count = _estimate_messages_tokens(messages)
        decision = evaluate_prefetch_policy(
            policy, messages, token_count, max_tokens,
            prefetch_policy=prefetch_policy,
        )
        if decision is None:
            return {}

        plan = _plan_compression(
            messages, decision,
            cutoff_fn=effective_cutoff_fn,
            max_tokens=max_tokens,
        )
        if plan is None:
            return {}

        cutoff, messages_to_evict, messages_to_keep = plan
        message_count = len(messages)

        async def _coro() -> _PrefetchResult:
            return await _run_prefetch(
                thread_id=thread_id,
                messages=messages,
                previous_summary_raw=previous_summary_raw,
                decision=decision,
                cutoff=cutoff,
                messages_to_evict=messages_to_evict,
                messages_to_keep=messages_to_keep,
            )

        started = registry.start(
            thread_id,
            message_count=message_count,
            cutoff=cutoff,
            coro_factory=_coro,
        )
        if started:
            logger.info(
                "Prefetch started: thread=%s messages=%d cutoff=%d",
                thread_id, message_count, cutoff,
            )
        return {}

    return prefetch_node


# ---------------------------------------------------------------------------
# Factory: compress_node (async graph node)
# ---------------------------------------------------------------------------


def make_compress_node(
    policy: SummarizationPolicy | Callable[..., PolicyDecision | None],
    summary_model: BaseChatModel,
    *,
    summary_prompt: str | None = None,
    summary_budget: int = DEFAULT_SUMMARY_BUDGET,
    history_dir: Path | None = None,
    workspace_dir: Path | None = None,
    max_tokens: int | None = None,
    is_perpetual: bool = False,
    on_event: SummarizationCallback | None = None,
    abort_check: Callable[[], bool] | None = None,
    cutoff_fn: Any | None = None,
    arg_truncation_trigger: int | None = None,
    arg_truncation_keep: int | None = None,
    arg_max_length: int | None = None,
    registry: PrefetchRegistry | None = None,
) -> Callable[..., Any]:
    """Create the mandatory compress node: evict messages and apply summary."""
    effective_cutoff_fn = cutoff_fn or find_safe_cutoff
    effective_max_length = arg_max_length if arg_max_length is not None else DEFAULT_ARG_MAX_LENGTH
    default_prompt, wrapper_with_path, wrapper_no_path = _resolve_prompt_wrappers(is_perpetual)
    if summary_prompt is None:
        summary_prompt = default_prompt

    async def compress_node(
        state: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        messages = state.get("messages", [])
        previous_summary = state.get("summary", "")
        previous_summary_raw = state.get("summary_raw", "")

        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = configurable.get("thread_id", "default")

        token_count = _estimate_messages_tokens(messages)
        decision = evaluate_policy(policy, messages, token_count, max_tokens)

        if decision is None:
            return {"messages": [], "summary": previous_summary, "summary_raw": previous_summary_raw}

        plan = _plan_compression(
            messages, decision,
            cutoff_fn=effective_cutoff_fn,
            max_tokens=max_tokens,
        )
        if plan is None:
            return {"messages": [], "summary": previous_summary, "summary_raw": previous_summary_raw}

        cutoff, messages_to_evict, messages_to_keep = plan

        if abort_check is not None and abort_check():
            logger.info("Compression skipped - abort signal received")
            return {"messages": [], "summary": previous_summary, "summary_raw": previous_summary_raw}

        prefetched: _PrefetchResult | None = None
        if registry is not None:
            prefetched = await registry.await_result(thread_id, cutoff=cutoff)

        if prefetched is not None and prefetched.cutoff == cutoff:
            # Filter stale evictions: matching cutoff indices do not guarantee
            # matching message IDs — before_agent middleware can insert/remove
            # messages between the prefetch snapshot and the compress apply.
            current_ids = {m.id for m in messages if getattr(m, "id", None)}
            valid_evictions = [r for r in prefetched.evictions
                               if getattr(r, "id", None) in current_ids]
            if len(valid_evictions) != len(prefetched.evictions):
                logger.warning(
                    "Compression fast path: %d stale evictions filtered "
                    "(messages shifted by middleware)",
                    len(prefetched.evictions) - len(valid_evictions),
                )
            logger.info(
                "Compression fast path: applying prefetched summary "
                "(evicted %d, kept %d)",
                prefetched.messages_summarized,
                prefetched.messages_kept,
            )
            if on_event is not None:
                on_event(SummarizationEvent(
                    phase="after",
                    cutoff_index=cutoff,
                    messages_summarized=prefetched.messages_summarized,
                    messages_kept=prefetched.messages_kept,
                    summary_tokens=prefetched.summary_tokens,
                    file_path=prefetched.file_path,
                    thread_id=thread_id,
                ))
            return {
                "summary": prefetched.summary_wrapper,
                "summary_raw": prefetched.summary_raw_text,
                "messages": valid_evictions,
            }

        if prefetched is not None and prefetched.cutoff < cutoff:
            delta_messages = messages[prefetched.cutoff:cutoff]
            file_path: str | None = None
            if history_dir is not None:
                file_path = write_transcript(
                    messages[:cutoff], thread_id,
                    history_dir=history_dir,
                    workspace_dir=workspace_dir,
                )

            if delta_messages:
                logger.info(
                    "Compression delta path: extending prefetch cutoff %d -> %d "
                    "(%d extra messages)",
                    prefetched.cutoff, cutoff, len(delta_messages),
                )
                if on_event is not None:
                    on_event(SummarizationEvent(
                        phase="before",
                        cutoff_index=cutoff,
                        messages_summarized=len(messages[:cutoff]),
                        messages_kept=len(messages_to_keep),
                        summary_tokens=0,
                        file_path=file_path,
                        thread_id=thread_id,
                    ))
                delta_prompt = _build_summary_prompt(
                    messages_to_evict=delta_messages,
                    previous_summary_raw=prefetched.summary_raw_text,
                    summary_prompt=summary_prompt,
                    summary_budget=summary_budget,
                    history_dir=history_dir,
                    workspace_dir=workspace_dir,
                    effective_max_length=effective_max_length,
                )
                summary_text = await _invoke_summary_model(summary_model, delta_prompt)
                wrapper = _wrap_summary(
                    summary_text, file_path,
                    wrapper_with_path=wrapper_with_path,
                    wrapper_no_path=wrapper_no_path,
                )
                delta_summary_raw = summary_text
            else:
                wrapper = prefetched.summary_wrapper
                delta_summary_raw = prefetched.summary_raw_text

            evictions = _build_evictions(messages[:cutoff])
            summary_tokens = estimate_tokens(wrapper)
            logger.info(
                "Compression delta path: evicted %d messages, kept %d",
                len(messages[:cutoff]),
                len(messages_to_keep),
            )
            if on_event is not None:
                on_event(SummarizationEvent(
                    phase="after",
                    cutoff_index=cutoff,
                    messages_summarized=len(messages[:cutoff]),
                    messages_kept=len(messages_to_keep),
                    summary_tokens=summary_tokens,
                    file_path=file_path,
                    thread_id=thread_id,
                ))
            return {"summary": wrapper, "summary_raw": delta_summary_raw, "messages": evictions}

        file_path: str | None = None
        if history_dir is not None:
            file_path = write_transcript(
                messages_to_evict, thread_id,
                history_dir=history_dir,
                workspace_dir=workspace_dir,
            )

        prompt_text = _build_summary_prompt(
            messages_to_evict=messages_to_evict,
            previous_summary_raw=previous_summary_raw,
            summary_prompt=summary_prompt,
            summary_budget=summary_budget,
            history_dir=history_dir,
            workspace_dir=workspace_dir,
            effective_max_length=effective_max_length,
        )

        if on_event is not None:
            on_event(SummarizationEvent(
                phase="before",
                cutoff_index=cutoff,
                messages_summarized=len(messages_to_evict),
                messages_kept=len(messages_to_keep),
                summary_tokens=0,
                file_path=file_path,
                thread_id=thread_id,
            ))

        try:
            summary_text = await _invoke_summary_model(summary_model, prompt_text)
        except Exception:
            logger.error("Compression failed, continuing with full history", exc_info=True)
            if on_event is not None:
                on_event(SummarizationEvent(
                    phase="after",
                    cutoff_index=cutoff,
                    messages_summarized=len(messages_to_evict),
                    messages_kept=len(messages_to_keep),
                    summary_tokens=0,
                    file_path=file_path,
                    thread_id=thread_id,
                    error="Compression LLM call failed",
                ))
            return {"messages": [], "summary": previous_summary, "summary_raw": previous_summary_raw}

        wrapper = _wrap_summary(
            summary_text, file_path,
            wrapper_with_path=wrapper_with_path,
            wrapper_no_path=wrapper_no_path,
        )
        evictions = _build_evictions(messages_to_evict)
        summary_tokens = estimate_tokens(summary_text)

        logger.info(
            "Compression: evicted %d messages, kept %d, summary ~%d tokens, transcript=%s",
            len(messages_to_evict),
            len(messages_to_keep),
            summary_tokens,
            file_path or "(none)",
        )

        if on_event is not None:
            on_event(SummarizationEvent(
                phase="after",
                cutoff_index=cutoff,
                messages_summarized=len(messages_to_evict),
                messages_kept=len(messages_to_keep),
                summary_tokens=summary_tokens,
                file_path=file_path,
                thread_id=thread_id,
            ))

        return {"summary": wrapper, "summary_raw": summary_text, "messages": evictions}

    return compress_node

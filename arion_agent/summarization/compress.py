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

DEFAULT_SUMMARY_BUDGET = 3600
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


def _extract_human_summaries(messages: list[AnyMessage]) -> list[str]:
    """Extract full human message texts from a message list for indexing."""
    summaries: list[str] = []
    for m in messages:
        if not isinstance(m, HumanMessage):
            continue
        text = m.content if isinstance(m.content, str) else str(m.content)
        text = text.strip()
        if not text:
            continue
        summaries.append(text)
    return summaries


def _build_human_index(summaries: list[str]) -> str:
    """Format human message summaries as a numbered index block."""
    if not summaries:
        return ""
    lines = ["## Human Messages\n"]
    for i, s in enumerate(summaries, 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    return "\n".join(lines)


def _append_human_prompts_file(
    thread_dir: Path,
    summaries: list[str],
    event_number: int,
    timestamp: str,
) -> None:
    """Append human prompts from this compression event to the thread index."""
    if not summaries:
        return
    from arion_agent.util.persistence import append_file

    header = f"# Event {event_number} ({timestamp})\n"
    entries = "\n".join(f"- {s}" for s in summaries) + "\n\n"
    append_file(thread_dir / "_human_prompts.md", header + entries)


def write_transcript(
    messages: list[AnyMessage],
    thread_id: str,
    *,
    history_dir: Path,
    workspace_dir: Path | None = None,
) -> str | None:
    """Write evicted messages to a markdown transcript file.

    Returns the workspace-relative directory path, or None on failure.
    """
    from arion_agent.util.persistence import ensure_directory, file_exists, glob_files, write_file as persistence_write

    thread_dir = history_dir / "conversation_history" / thread_id
    ensure_directory(thread_dir)

    if not messages:
        if workspace_dir is not None:
            from arion_agent.util.persistence import workspace_relative_path
            return workspace_relative_path(thread_dir, workspace_dir)
        return str(thread_dir)

    existing = [p for p in glob_files(thread_dir, "*.md") if p.name != "_human_prompts.md"]
    event_number = len(existing) + 1
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    file_timestamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    file_name = f"{file_timestamp}.md"
    file_path = thread_dir / file_name
    if file_exists(file_path):
        file_name = f"{file_timestamp}_{event_number}.md"
        file_path = thread_dir / file_name

    human_summaries = _extract_human_summaries(messages)
    human_index = _build_human_index(human_summaries)

    content = (
        f"# Compression event {event_number}\n"
        f"Timestamp: {timestamp}\n"
        f"Messages evicted: {len(messages)}\n\n"
        + (f"{human_index}\n---\n\n" if human_index else "---\n\n")
        + f"{get_buffer_string(messages)}\n"
    )

    try:
        persistence_write(file_path, content)
        _append_human_prompts_file(thread_dir, human_summaries, event_number, timestamp)
        logger.info("Wrote transcript: %s (%d messages, %d human)", file_path, len(messages), len(human_summaries))
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


def _build_summary_prompt(
    *,
    messages_to_evict: list[AnyMessage],
    previous_summary: str,
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

    if previous_summary:
        prompt_text = (
            f"Previous summary to incorporate:\n{previous_summary}\n\n"
            f"{prompt_text}"
        )
    return prompt_text


async def _invoke_summary_model(summary_model: BaseChatModel, prompt_text: str) -> str:
    response = await summary_model.ainvoke(prompt_text)
    raw = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
        )
    return str(raw)


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
        previous_summary: str,
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
            previous_summary=previous_summary,
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
        previous_summary = state.get("summary", "")

        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = configurable.get("thread_id", "default")

        if abort_check is not None and abort_check():
            return {}

        ready = registry.take_if_ready(thread_id)
        if ready is not None:
            logger.info(
                "Prefetch apply: evicting %d messages, kept %d (proactive, no block)",
                ready.messages_summarized, ready.messages_kept,
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
                "messages": ready.evictions,
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
                previous_summary=previous_summary,
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

        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = configurable.get("thread_id", "default")

        token_count = _estimate_messages_tokens(messages)
        decision = evaluate_policy(policy, messages, token_count, max_tokens)

        if decision is None:
            return {"messages": [], "summary": previous_summary}

        plan = _plan_compression(
            messages, decision,
            cutoff_fn=effective_cutoff_fn,
            max_tokens=max_tokens,
        )
        if plan is None:
            return {"messages": [], "summary": previous_summary}

        cutoff, messages_to_evict, messages_to_keep = plan

        if abort_check is not None and abort_check():
            logger.info("Compression skipped - abort signal received")
            return {"messages": [], "summary": previous_summary}

        prefetched: _PrefetchResult | None = None
        if registry is not None:
            prefetched = await registry.await_result(thread_id, cutoff=cutoff)

        if prefetched is not None and prefetched.cutoff == cutoff:
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
                "messages": prefetched.evictions,
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
                    previous_summary=prefetched.summary_wrapper,
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
            else:
                wrapper = prefetched.summary_wrapper

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
            return {"summary": wrapper, "messages": evictions}

        file_path: str | None = None
        if history_dir is not None:
            file_path = write_transcript(
                messages_to_evict, thread_id,
                history_dir=history_dir,
                workspace_dir=workspace_dir,
            )

        prompt_text = _build_summary_prompt(
            messages_to_evict=messages_to_evict,
            previous_summary=previous_summary,
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
            return {"messages": [], "summary": previous_summary}

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

        return {"summary": wrapper, "messages": evictions}

    return compress_node

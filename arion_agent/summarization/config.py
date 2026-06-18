"""Summarization policy and configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable
if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage


@dataclass(frozen=True)
class PolicyDecision:
    """Result from a policy callable indicating summarization should proceed.

    Exactly one of the keep fields should be set. If both are None,
    defaults to keep_last_messages=20.
    """

    keep_last_messages: int | None = None
    keep_last_tokens: int | None = None
    keep_last_fraction: float | None = None


@dataclass(frozen=True)
class SummarizationPolicy:
    """Declarative summarization policy.

    Trigger: fire when ANY non-None condition is met (OR logic).
    Keep: first non-None field is used (priority order).

    Prefetch triggers start background summarization in the headroom zone
    before the hard trigger fires. When unset, prefetch thresholds default
    to 80% of the corresponding hard trigger.
    """

    trigger_messages: int | None = None
    trigger_tokens: int | None = None
    trigger_fraction: float | None = None

    prefetch_messages: int | None = None
    prefetch_tokens: int | None = None
    prefetch_fraction: float | None = None

    keep_messages: int | None = None
    keep_tokens: int | None = None
    keep_fraction: float | None = None


@dataclass(frozen=True)
class SummarizationConfig:
    """Full summarization configuration for agents.
    
    Allows customization of summarization behavior including custom templates,
    budget, and truncation settings. Pass this to create_arion_agent() via
    the summarization parameter.
    
    Example:
        from arion_agent.summarization.config import SummarizationConfig, SummarizationPolicy
        
        config = SummarizationConfig(
            summary_prompt='''Custom prompt template...''',
            arg_truncation_trigger=80,
            arg_truncation_keep=40,
            arg_max_length=5000,
            is_perpetual=True,
        )
        
        agent = create_arion_agent(..., summarization=config)
    """

    policy: SummarizationPolicy | Callable[..., "PolicyDecision | None"] | None = None
    prefetch_policy: Callable[..., "PolicyDecision | None"] | None = None
    summary_prompt: str | None = None
    summary_budget: int | None = None
    arg_truncation_trigger: int | None = None
    arg_truncation_keep: int | None = None
    arg_max_length: int | None = None
    is_perpetual: bool = False

@runtime_checkable
class PolicyCallable(Protocol):
    """Protocol for custom policy callables."""

    def __call__(
        self,
        messages: list[AnyMessage],
        token_count: int,
        max_tokens: int | None,
    ) -> PolicyDecision | None: ...


@runtime_checkable
class CutoffFn(Protocol):
    """Protocol for custom cutoff strategies.

    Given a message list, a policy decision (how many messages/tokens to keep),
    and optional max_tokens, return the cutoff index. Messages before the index
    are evicted, at/after are kept.

    The default implementation (find_safe_cutoff) walks backward past
    ToolMessage clusters to avoid splitting AI tool_calls from their
    responses. Custom implementations can use the exported helpers
    raw_cutoff() and walk_back_past_tool_messages() as building blocks.
    """

    def __call__(
        self,
        messages: list[AnyMessage],
        decision: PolicyDecision,
        max_tokens: int | None,
    ) -> int: ...


@dataclass(frozen=True)
class SummarizationEvent:
    """Record of a summarization event for callbacks and auditing.

    phase: "before" (pre-summarization, summary not yet generated) or
           "after" (post-summarization, summary produced or error occurred).
    """

    phase: str
    cutoff_index: int
    messages_summarized: int
    messages_kept: int
    summary_tokens: int
    file_path: str | None
    thread_id: str | None = None
    error: str | None = None


@runtime_checkable
class SummarizationCallback(Protocol):
    """Protocol for summarization event callbacks.

    Called with phase="before" just before the summary LLM call, and
    phase="after" once summarization completes (or fails). The "before"
    event has summary_tokens=0 and error=None. A "before" callback
    can be used for logging, notifications, or telemetry.
    """

    def __call__(self, event: SummarizationEvent) -> None: ...


@dataclass(frozen=True)
class SummaryRecord:
    """One entry in the augmentary JSONL chain."""

    index: int
    summary_text: str
    summary_tokens: int
    messages_summarized: int
    messages_kept: int
    store_seq_start: int
    store_seq_end: int
    transcript_path: str | None
    timestamp: str
    recovered: bool = False
    recovery_reason: str | None = None


@dataclass
class SummaryFileState:
    """In-memory representation of the latest summary chain state for a thread."""

    latest: SummaryRecord | None = None
    summary_count: int = 0


RECOVERY_THRESHOLD = 200


OPTIONAL_SECTIONS: dict[str, str] = {
    "evidence_and_source_tracing": (
        "\n## EVIDENCE AND SOURCE TRACING\n"
        "Specific data points, sources, and citations referenced. The reasoning\n"
        "chain that led to conclusions. Provenance of key facts or figures."
    ),
    "discoveries": (
        "\n## DISCOVERIES\n"
        "Novel findings or observations. Hypotheses formed. Experiments or\n"
        "approaches attempted and their outcomes."
    ),
}

"""Standard summarization policies and default thresholds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arion_agent.summarization.config import PolicyDecision

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage

DEFAULT_PREFETCH_MESSAGES = 165
DEFAULT_TRIGGER_MESSAGES = 240
DEFAULT_TRIGGER_TOKEN_FRACTION = 0.85
DEFAULT_KEEP_MESSAGES = 60
DEFAULT_KEEP_TOKEN_FRACTION = 0.25
DEFAULT_PREFETCH_TOKEN_FRACTION = 0.50


def _standard_policy(
    messages: list[AnyMessage],
    token_count: int,
    max_tokens: int | None,
) -> PolicyDecision | None:
    if max_tokens and token_count > max_tokens * DEFAULT_TRIGGER_TOKEN_FRACTION:
        return PolicyDecision(keep_last_fraction=DEFAULT_KEEP_TOKEN_FRACTION)
    if len(messages) > DEFAULT_TRIGGER_MESSAGES:
        return PolicyDecision(keep_last_messages=DEFAULT_KEEP_MESSAGES)
    return None


def _standard_prefetch_policy(
    messages: list[AnyMessage],
    token_count: int,
    max_tokens: int | None,
) -> PolicyDecision | None:
    """Headroom trigger: same keep rules as must, lower entry thresholds."""
    if _standard_policy(messages, token_count, max_tokens) is not None:
        return None

    prefetch_messages = DEFAULT_PREFETCH_MESSAGES
    prefetch_token_fraction = DEFAULT_PREFETCH_TOKEN_FRACTION

    if max_tokens and token_count > max_tokens * prefetch_token_fraction:
        return PolicyDecision(keep_last_fraction=DEFAULT_KEEP_TOKEN_FRACTION)
    if len(messages) > prefetch_messages:
        return PolicyDecision(keep_last_messages=DEFAULT_KEEP_MESSAGES)
    return None


def _aggressive_policy(
    messages: list[AnyMessage],
    token_count: int,
    max_tokens: int | None,
) -> PolicyDecision | None:
    if max_tokens and token_count > max_tokens * 0.70:
        return PolicyDecision(keep_last_fraction=0.15)
    if len(messages) > 40:
        return PolicyDecision(keep_last_messages=10)
    return None


STANDARD_POLICY = _standard_policy
STANDARD_PREFETCH_POLICY = _standard_prefetch_policy
AGGRESSIVE_POLICY = _aggressive_policy

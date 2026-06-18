"""Tests for summarization budget and policy defaults."""

from __future__ import annotations

from arion_agent.summarization.compress import (
    DEFAULT_SUMMARY_BUDGET,
    DEFAULT_SUMMARY_BUDGET_FRACTION,
)
from arion_agent.summarization.policies import (
    DEFAULT_KEEP_MESSAGES,
    DEFAULT_PREFETCH_MESSAGES,
    DEFAULT_TRIGGER_MESSAGES,
    STANDARD_POLICY,
)


def test_default_budget_constants():
    assert DEFAULT_SUMMARY_BUDGET == 3600
    assert DEFAULT_SUMMARY_BUDGET_FRACTION == 0.60


def test_standard_policy_trigger_and_keep():
    assert DEFAULT_PREFETCH_MESSAGES == 80
    assert DEFAULT_TRIGGER_MESSAGES == 165
    assert DEFAULT_KEEP_MESSAGES == 40
    assert STANDARD_POLICY([object()] * 164, 100, 32_000) is None
    decision = STANDARD_POLICY([object()] * 166, 100, 32_000)
    assert decision is not None
    assert decision.keep_last_messages == 40

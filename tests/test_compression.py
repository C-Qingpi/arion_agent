"""Tests for Phase 15 compression node architecture.

Tests cover:
  - ArionState summary channel
  - should_compress conditional routing
  - compress_node eviction and summary generation
  - Safe cutoff (tool pair preservation)
  - Argument truncation as standalone helper
  - Transcript file writing
  - Graph wiring with conditional edges
  - Per-agent checkpointer path (.sqlite)
  - State migration (old checkpoint without summary)
  - Crash recovery (resume from checkpoint)
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)

from arion_agent.summarization.compress import (
    PrefetchRegistry,
    evaluate_policy,
    evaluate_prefetch_policy,
    find_kept_orphans,
    find_safe_cutoff,
    make_compress_node,
    make_must_compress,
    make_prefetch_node,
    make_route_compression,
    make_should_compress,
    truncate_args,
    write_transcript,
)
from arion_agent.summarization.config import PolicyDecision, SummarizationPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(n: int, *, with_tools: bool = False) -> list[AnyMessage]:
    """Build a synthetic message list for testing."""
    msgs: list[AnyMessage] = []
    for i in range(n):
        if i % 3 == 0:
            msgs.append(HumanMessage(content=f"User message {i}", id=f"h{i}"))
        elif i % 3 == 1:
            if with_tools:
                msgs.append(AIMessage(
                    content="",
                    tool_calls=[{"name": "test_tool", "args": {"x": str(i)}, "id": f"tc{i}"}],
                    id=f"a{i}",
                ))
            else:
                msgs.append(AIMessage(content=f"AI response {i}", id=f"a{i}"))
        else:
            if with_tools:
                msgs.append(ToolMessage(
                    content=f"Tool result {i}",
                    tool_call_id=f"tc{i-1}",
                    name="test_tool",
                    id=f"t{i}",
                ))
            else:
                msgs.append(AIMessage(content=f"AI response {i}", id=f"a{i}"))
    return msgs


# ---------------------------------------------------------------------------
# Tests: Policy evaluation
# ---------------------------------------------------------------------------


class TestPolicyEvaluation:
    def test_standard_policy_below_threshold(self):
        from arion_agent.summarization.policies import STANDARD_POLICY
        msgs = _make_messages(20)
        result = evaluate_policy(STANDARD_POLICY, msgs, 1000, 100_000)
        assert result is None

    def test_standard_policy_message_trigger(self):
        from arion_agent.summarization.policies import DEFAULT_TRIGGER_MESSAGES, STANDARD_POLICY
        msgs = _make_messages(DEFAULT_TRIGGER_MESSAGES + 1)
        result = evaluate_policy(STANDARD_POLICY, msgs, 1000, 100_000)
        assert result is not None
        assert result.keep_last_messages == 40

    def test_standard_policy_token_trigger(self):
        from arion_agent.summarization.policies import STANDARD_POLICY
        msgs = _make_messages(10)
        result = evaluate_policy(STANDARD_POLICY, msgs, 90_000, 100_000)
        assert result is not None
        assert result.keep_last_fraction == 0.25

    def test_declarative_policy(self):
        policy = SummarizationPolicy(
            trigger_messages=50,
            keep_messages=10,
        )
        msgs = _make_messages(60)
        result = evaluate_policy(policy, msgs, 5000, None)
        assert result is not None
        assert result.keep_last_messages == 10

    def test_declarative_policy_no_trigger(self):
        policy = SummarizationPolicy(trigger_messages=50, keep_messages=10)
        msgs = _make_messages(30)
        result = evaluate_policy(policy, msgs, 5000, None)
        assert result is None

    def test_custom_callable_policy(self):
        def my_policy(msgs, tokens, max_tokens):
            if len(msgs) > 5:
                return PolicyDecision(keep_last_messages=3)
            return None

        msgs = _make_messages(10)
        result = evaluate_policy(my_policy, msgs, 0, None)
        assert result is not None
        assert result.keep_last_messages == 3


# ---------------------------------------------------------------------------
# Tests: should_compress routing
# ---------------------------------------------------------------------------


class TestShouldCompress:
    def test_below_threshold_routes_to_model(self):
        from arion_agent.summarization.policies import STANDARD_POLICY
        should_compress = make_should_compress(STANDARD_POLICY, max_tokens=100_000)
        state = {"messages": _make_messages(10), "summary": ""}
        assert should_compress(state) == "model"

    def test_headroom_routes_to_prefetch(self):
        from arion_agent.summarization.policies import (
            DEFAULT_PREFETCH_MESSAGES,
            STANDARD_POLICY,
        )
        should_compress = make_should_compress(STANDARD_POLICY, max_tokens=100_000)
        state = {"messages": _make_messages(DEFAULT_PREFETCH_MESSAGES + 1), "summary": ""}
        assert should_compress(state) == "prefetch"

    def test_must_threshold_routes_to_model_not_prefetch(self):
        from arion_agent.summarization.policies import DEFAULT_TRIGGER_MESSAGES, STANDARD_POLICY
        should_compress = make_should_compress(STANDARD_POLICY, max_tokens=100_000)
        state = {"messages": _make_messages(DEFAULT_TRIGGER_MESSAGES + 1), "summary": ""}
        assert should_compress(state) == "model"

    def test_very_few_messages_always_model(self):
        def always_trigger(msgs, tokens, max_t):
            return PolicyDecision(keep_last_messages=2)

        should_compress = make_should_compress(always_trigger)
        state = {"messages": _make_messages(3), "summary": ""}
        assert should_compress(state) == "model"

    def test_empty_messages_routes_to_model(self):
        from arion_agent.summarization.policies import STANDARD_POLICY
        should_compress = make_should_compress(STANDARD_POLICY)
        state = {"messages": [], "summary": ""}
        assert should_compress(state) == "model"


class TestMustCompress:
    def test_below_threshold_routes_to_model(self):
        from arion_agent.summarization.policies import STANDARD_POLICY
        must_compress = make_must_compress(STANDARD_POLICY, max_tokens=100_000)
        state = {"messages": _make_messages(10), "summary": ""}
        assert must_compress(state) == "model"

    def test_above_threshold_routes_to_compress(self):
        from arion_agent.summarization.policies import DEFAULT_TRIGGER_MESSAGES, STANDARD_POLICY
        must_compress = make_must_compress(STANDARD_POLICY, max_tokens=100_000)
        state = {"messages": _make_messages(DEFAULT_TRIGGER_MESSAGES + 1), "summary": ""}
        assert must_compress(state) == "compress"


class TestRouteCompression:
    def test_must_wins_over_prefetch(self):
        from arion_agent.summarization.policies import DEFAULT_TRIGGER_MESSAGES, STANDARD_POLICY
        route = make_route_compression(STANDARD_POLICY, max_tokens=100_000)
        state = {"messages": _make_messages(DEFAULT_TRIGGER_MESSAGES + 1), "summary": ""}
        assert route(state) == "compress"

    def test_prefetch_when_in_headroom_only(self):
        from arion_agent.summarization.policies import (
            DEFAULT_PREFETCH_MESSAGES,
            STANDARD_POLICY,
        )
        route = make_route_compression(STANDARD_POLICY, max_tokens=100_000)
        state = {"messages": _make_messages(DEFAULT_PREFETCH_MESSAGES + 1), "summary": ""}
        assert route(state) == "prefetch"


class TestPrefetchPolicy:
    def test_prefetch_below_headroom(self):
        from arion_agent.summarization.policies import STANDARD_POLICY
        msgs = _make_messages(50)
        assert evaluate_prefetch_policy(STANDARD_POLICY, msgs, 1000, 100_000) is None

    def test_prefetch_in_headroom(self):
        from arion_agent.summarization.policies import (
            DEFAULT_PREFETCH_MESSAGES,
            STANDARD_POLICY,
        )
        msgs = _make_messages(DEFAULT_PREFETCH_MESSAGES + 1)
        result = evaluate_prefetch_policy(STANDARD_POLICY, msgs, 1000, 100_000)
        assert result is not None
        assert result.keep_last_messages == 40

    def test_prefetch_skipped_when_must_applies(self):
        from arion_agent.summarization.policies import DEFAULT_TRIGGER_MESSAGES, STANDARD_POLICY
        msgs = _make_messages(DEFAULT_TRIGGER_MESSAGES + 1)
        assert evaluate_prefetch_policy(STANDARD_POLICY, msgs, 1000, 100_000) is None


# ---------------------------------------------------------------------------
# Tests: Safe cutoff
# ---------------------------------------------------------------------------


class TestSafeCutoff:
    def test_keep_last_messages(self):
        msgs = _make_messages(30)
        decision = PolicyDecision(keep_last_messages=10)
        cutoff = find_safe_cutoff(msgs, decision, None)
        assert cutoff > 0
        assert cutoff <= 20

    def test_never_splits_tool_pair(self):
        msgs = _make_messages(30, with_tools=True)
        decision = PolicyDecision(keep_last_messages=10)
        cutoff = find_safe_cutoff(msgs, decision, None)
        kept = msgs[cutoff:]
        for m in kept:
            if isinstance(m, ToolMessage):
                parent_id = m.tool_call_id
                parent_found = any(
                    isinstance(k, AIMessage) and k.tool_calls
                    and any(tc["id"] == parent_id for tc in k.tool_calls)
                    for k in kept
                )
                assert parent_found, f"Orphaned ToolMessage {m.id} in kept portion"

    def test_zero_cutoff_for_few_messages(self):
        msgs = _make_messages(5)
        decision = PolicyDecision(keep_last_messages=10)
        cutoff = find_safe_cutoff(msgs, decision, None)
        assert cutoff == 0

    def test_agent_pattern_no_human_alignment(self):
        """Agent scenario: 1 HumanMessage + many AI/Tool turns.

        The cutoff should stay near the raw target, not walk all the
        way back to the sole HumanMessage at index 0.
        """
        msgs: list[AnyMessage] = [HumanMessage(content="do the task", id="h0")]
        for i in range(1, 81):
            if i % 2 == 1:
                msgs.append(AIMessage(
                    content="",
                    tool_calls=[{"name": "tool", "args": {}, "id": f"tc{i}"}],
                    id=f"a{i}",
                ))
            else:
                msgs.append(ToolMessage(
                    content="ok",
                    tool_call_id=f"tc{i-1}",
                    name="tool",
                    id=f"t{i}",
                ))
        decision = PolicyDecision(keep_last_messages=20)
        cutoff = find_safe_cutoff(msgs, decision, None)
        assert cutoff >= 55, f"Cutoff {cutoff} too far back — should not align to HumanMessage at 0"
        assert cutoff <= 65

    def test_allows_non_human_first_kept(self):
        """Cutoff may land on AIMessage — no HumanMessage-seeking.

        The summary HumanMessage is prepended by model_node, so the
        kept portion does not need to start with a HumanMessage.
        """
        msgs = _make_messages(30)
        decision = PolicyDecision(keep_last_messages=10)
        cutoff = find_safe_cutoff(msgs, decision, None)
        first_kept = msgs[cutoff]
        assert isinstance(first_kept, AIMessage), (
            f"Expected AIMessage at cutoff {cutoff}, got {type(first_kept).__name__}. "
            f"find_safe_cutoff should not walk back to a HumanMessage boundary."
        )

    def test_no_over_eviction_with_tools(self):
        """Tool-heavy pattern: cutoff walks back past ToolMessage to
        AIMessage, but must NOT continue to a HumanMessage.
        """
        msgs = _make_messages(30, with_tools=True)
        decision = PolicyDecision(keep_last_messages=10)
        cutoff = find_safe_cutoff(msgs, decision, None)
        first_kept = msgs[cutoff]
        assert not isinstance(first_kept, ToolMessage), (
            "Cutoff should never land on a ToolMessage"
        )
        assert cutoff >= 19, (
            f"Cutoff {cutoff} is too far back — should be 19 or 20, not walked "
            f"back to HumanMessage at 18"
        )

    def test_cutoff_on_natural_human_stays(self):
        """When raw cutoff naturally lands on a HumanMessage, no shift needed."""
        msgs = _make_messages(30)
        decision = PolicyDecision(keep_last_messages=9)
        cutoff = find_safe_cutoff(msgs, decision, None)
        assert isinstance(msgs[cutoff], HumanMessage)
        assert cutoff == 21

    def test_walkback_to_zero_uses_max_one(self):
        """When walk_back reaches 0, max(1, 0) forces cutoff=1.

        find_kept_orphans must then handle the orphaned ToolMessage
        at position 1 (its parent AI at 0 is evicted).
        """
        msgs: list[AnyMessage] = [
            AIMessage(
                content="", id="a0",
                tool_calls=[{"name": "t", "args": {}, "id": "tc0"}],
            ),
            ToolMessage(content="ok", tool_call_id="tc0", name="t", id="t1"),
            ToolMessage(content="ok2", tool_call_id="tc0", name="t", id="t2"),
            HumanMessage(content="user", id="h3"),
            AIMessage(content="resp", id="a4"),
        ]
        decision = PolicyDecision(keep_last_messages=3)
        cutoff = find_safe_cutoff(msgs, decision, None)
        assert cutoff >= 1, "max(1, ...) should prevent cutoff=0"

        evicted = msgs[:cutoff]
        kept = msgs[cutoff:]
        orphans = find_kept_orphans(evicted, kept)
        for orphan in orphans:
            assert isinstance(orphan, ToolMessage)
            assert orphan.tool_call_id == "tc0"

    def test_cutoff_on_ai_with_tool_calls_keeps_pair(self):
        """When cutoff lands on AI(tc), its ToolMessages must be in kept."""
        msgs: list[AnyMessage] = [
            HumanMessage(content="q1", id="h0"),
            AIMessage(content="r1", id="a1"),
            HumanMessage(content="q2", id="h2"),
            AIMessage(content="r2", id="a3"),
            HumanMessage(content="q3", id="h4"),
            AIMessage(
                content="", id="a5",
                tool_calls=[{"name": "search", "args": {}, "id": "tc5"}],
            ),
            ToolMessage(content="found it", tool_call_id="tc5", name="search", id="t6"),
            AIMessage(content="here is the answer", id="a7"),
        ]
        decision = PolicyDecision(keep_last_messages=3)
        cutoff = find_safe_cutoff(msgs, decision, None)
        kept = msgs[cutoff:]
        for m in kept:
            if isinstance(m, ToolMessage):
                parent_found = any(
                    isinstance(k, AIMessage) and k.tool_calls
                    and any(tc["id"] == m.tool_call_id for tc in k.tool_calls)
                    for k in kept
                )
                assert parent_found, (
                    f"ToolMessage {m.id} (tc={m.tool_call_id}) orphaned at cutoff={cutoff}"
                )

    def test_multi_tool_call_ai_at_boundary(self):
        """AI with 2 tool_calls: if evicted, both ToolMessages are orphaned."""
        msgs: list[AnyMessage] = [
            HumanMessage(content="start", id="h0"),
            AIMessage(
                content="", id="a1",
                tool_calls=[
                    {"name": "read", "args": {}, "id": "tc1a"},
                    {"name": "write", "args": {}, "id": "tc1b"},
                ],
            ),
            ToolMessage(content="data", tool_call_id="tc1a", name="read", id="t2"),
            ToolMessage(content="ok", tool_call_id="tc1b", name="write", id="t3"),
            HumanMessage(content="next", id="h4"),
            AIMessage(content="done", id="a5"),
            HumanMessage(content="more", id="h6"),
            AIMessage(content="sure", id="a7"),
        ]
        decision = PolicyDecision(keep_last_messages=4)
        cutoff = find_safe_cutoff(msgs, decision, None)

        evicted = msgs[:cutoff]
        kept = msgs[cutoff:]
        orphans = find_kept_orphans(evicted, kept)

        kept_after_orphan_removal = [m for m in kept if id(m) not in {id(o) for o in orphans}]
        for m in kept_after_orphan_removal:
            if isinstance(m, ToolMessage):
                parent_found = any(
                    isinstance(k, AIMessage) and k.tool_calls
                    and any(tc["id"] == m.tool_call_id for tc in k.tool_calls)
                    for k in kept_after_orphan_removal
                )
                assert parent_found, (
                    f"Orphaned ToolMessage {m.id} survived orphan removal"
                )

    def test_consecutive_tool_messages_walkback(self):
        """Raw cutoff deep in consecutive ToolMessages walks back to parent AI."""
        msgs: list[AnyMessage] = [
            HumanMessage(content="go", id="h0"),
            AIMessage(content="r", id="a1"),
            HumanMessage(content="do 3 things", id="h2"),
            AIMessage(
                content="", id="a3",
                tool_calls=[
                    {"name": "t1", "args": {}, "id": "tc3a"},
                    {"name": "t2", "args": {}, "id": "tc3b"},
                    {"name": "t3", "args": {}, "id": "tc3c"},
                ],
            ),
            ToolMessage(content="r1", tool_call_id="tc3a", name="t1", id="t4"),
            ToolMessage(content="r2", tool_call_id="tc3b", name="t2", id="t5"),
            ToolMessage(content="r3", tool_call_id="tc3c", name="t3", id="t6"),
            AIMessage(content="all done", id="a7"),
            HumanMessage(content="thanks", id="h8"),
            AIMessage(content="welcome", id="a9"),
        ]
        decision = PolicyDecision(keep_last_messages=4)
        cutoff = find_safe_cutoff(msgs, decision, None)
        first_kept = msgs[cutoff]
        assert not isinstance(first_kept, ToolMessage), (
            f"Cutoff {cutoff} landed on ToolMessage — walk_back should prevent this"
        )


# ---------------------------------------------------------------------------
# Tests: Orphan detection
# ---------------------------------------------------------------------------


class TestOrphanDetection:
    def test_no_orphans_when_clean_split(self):
        evicted = [HumanMessage(content="hi", id="h1")]
        kept = [AIMessage(content="hello", id="a1")]
        assert find_kept_orphans(evicted, kept) == []

    def test_detects_orphaned_tool_message(self):
        evicted = [AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "tc1"}],
            id="a1",
        )]
        kept = [ToolMessage(content="result", tool_call_id="tc1", name="t", id="t1")]
        orphans = find_kept_orphans(evicted, kept)
        assert len(orphans) == 1
        assert orphans[0].id == "t1"

    def test_multi_tool_orphans_all_detected(self):
        """AI with 2 tool_calls evicted: both ToolMessages in kept are orphans."""
        evicted = [AIMessage(
            content="", id="a0",
            tool_calls=[
                {"name": "r", "args": {}, "id": "tc_a"},
                {"name": "w", "args": {}, "id": "tc_b"},
            ],
        )]
        kept = [
            ToolMessage(content="d", tool_call_id="tc_a", name="r", id="t1"),
            ToolMessage(content="ok", tool_call_id="tc_b", name="w", id="t2"),
            HumanMessage(content="next", id="h3"),
        ]
        orphans = find_kept_orphans(evicted, kept)
        assert len(orphans) == 2
        orphan_ids = {o.id for o in orphans}
        assert orphan_ids == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Tests: Argument truncation
# ---------------------------------------------------------------------------


class TestTruncateArgs:
    def test_no_truncation_below_trigger(self):
        msgs = _make_messages(10)
        result = truncate_args(msgs, trigger=50)
        assert result is msgs

    def test_truncates_large_write_file_args(self):
        long_content = "x" * 5000
        msgs: list[AnyMessage] = []
        for i in range(60):
            if i % 2 == 0:
                msgs.append(AIMessage(
                    content="",
                    tool_calls=[{"name": "write_file", "args": {"path": "f.txt", "content": long_content}, "id": f"tc{i}"}],
                    id=f"a{i}",
                ))
            else:
                msgs.append(ToolMessage(content="ok", tool_call_id=f"tc{i-1}", name="write_file", id=f"t{i}"))

        result = truncate_args(msgs, trigger=50, keep=20, max_length=2000)
        early_msg = result[0]
        assert isinstance(early_msg, AIMessage)
        truncated_content = early_msg.tool_calls[0]["args"]["content"]
        assert len(truncated_content) == 2003 and truncated_content.endswith("...")

    def test_preserves_recent_messages(self):
        long_content = "x" * 5000
        msgs: list[AnyMessage] = []
        for i in range(60):
            msgs.append(AIMessage(
                content="",
                tool_calls=[{"name": "write_file", "args": {"content": long_content}, "id": f"tc{i}"}],
                id=f"a{i}",
            ))

        result = truncate_args(msgs, trigger=50, keep=20, max_length=2000)
        late_msg = result[-1]
        assert isinstance(late_msg, AIMessage)
        assert late_msg.tool_calls[0]["args"]["content"] == long_content


# ---------------------------------------------------------------------------
# Tests: Transcript writing
# ---------------------------------------------------------------------------


class TestTranscriptWriting:
    @staticmethod
    def _transcript_files(thread_dir: Path) -> list[Path]:
        """Return JSONL transcript files."""
        return list(thread_dir.glob("*.jsonl"))

    def test_writes_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir) / "agent"
            msgs = [HumanMessage(content="hello"), AIMessage(content="world")]
            result = write_transcript(msgs, "thread-1", history_dir=history_dir)
            assert result is not None

            thread_dir = history_dir / "conversation_history" / "thread-1"
            jsonl_files = self._transcript_files(thread_dir)
            assert len(jsonl_files) == 1

            from arion_agent.util.persistence import load_jsonl
            records = load_jsonl(jsonl_files[0])
            assert len(records) == 1
            r = records[0]
            assert r["event"] == 1
            assert r["msg_count"] == 2
            assert r["human_count"] == 1
            participants = r["participants"]
            assert participants[0]["role"] == "human"
            assert "hello" in participants[0]["content"]
            assert participants[1]["role"] == "ai"
            assert "world" in participants[1]["content"]

    def test_multiple_transcripts_increment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir) / "agent"
            msgs = [HumanMessage(content="first")]
            write_transcript(msgs, "t1", history_dir=history_dir)
            msgs = [HumanMessage(content="second")]
            write_transcript(msgs, "t1", history_dir=history_dir)

            thread_dir = history_dir / "conversation_history" / "t1"
            jsonl_files = self._transcript_files(thread_dir)
            assert len(jsonl_files) == 2

    def test_empty_messages_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir) / "agent"
            result = write_transcript([], "t1", history_dir=history_dir)
            assert result is not None
            thread_dir = history_dir / "conversation_history" / "t1"
            jsonl_files = self._transcript_files(thread_dir)
            assert len(jsonl_files) == 0


# ---------------------------------------------------------------------------
# Tests: compress_node
# ---------------------------------------------------------------------------


class TestCompressNode:
    @pytest.mark.asyncio
    async def test_compress_produces_summary_and_evictions(self):
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = MagicMock(content="Summary of events.")

        policy = SummarizationPolicy(trigger_messages=10, keep_messages=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            compress = make_compress_node(
                policy, mock_model,
                history_dir=Path(tmpdir),
                workspace_dir=Path(tmpdir),
            )

            messages = _make_messages(20)
            state: dict[str, Any] = {"messages": messages, "summary": ""}
            config = {"configurable": {"thread_id": "test-thread"}}

            result = await compress(state, config)

        assert "summary" in result
        assert "Summary of events." in result["summary"]
        assert len(result["messages"]) > 0
        remove_count = sum(1 for m in result["messages"] if isinstance(m, RemoveMessage))
        assert remove_count > 0

    @pytest.mark.asyncio
    async def test_compress_chains_previous_summary(self):
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = MagicMock(content="Updated summary.")

        policy = SummarizationPolicy(trigger_messages=10, keep_messages=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            compress = make_compress_node(
                policy, mock_model,
                history_dir=Path(tmpdir),
            )

            messages = _make_messages(20)
            state: dict[str, Any] = {
                "messages": messages,
                "summary": "Previous summary of early events.",
            }
            config = {"configurable": {"thread_id": "test-thread"}}

            await compress(state, config)

        call_args = mock_model.ainvoke.call_args[0][0]
        assert "Previous summary" in call_args

    @pytest.mark.asyncio
    async def test_compress_graceful_degradation_on_failure(self):
        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = RuntimeError("LLM down")

        policy = SummarizationPolicy(trigger_messages=10, keep_messages=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            compress = make_compress_node(
                policy, mock_model,
                history_dir=Path(tmpdir),
            )

            messages = _make_messages(20)
            state: dict[str, Any] = {"messages": messages, "summary": "old"}
            config = {"configurable": {"thread_id": "test-thread"}}

            result = await compress(state, config)

        assert result["summary"] == "old"
        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_compress_no_trigger_returns_unchanged(self):
        mock_model = AsyncMock()
        policy = SummarizationPolicy(trigger_messages=100, keep_messages=50)

        compress = make_compress_node(policy, mock_model)
        state: dict[str, Any] = {"messages": _make_messages(10), "summary": "prev"}
        config = {"configurable": {"thread_id": "t1"}}

        result = await compress(state, config)

        assert result["summary"] == "prev"
        assert result["messages"] == []
        mock_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_compress_uses_completed_prefetch(self):
        from arion_agent.summarization.compress import (
            _PrefetchEntry,
            _PrefetchResult,
            _build_evictions,
            _plan_compression,
        )

        policy = SummarizationPolicy(trigger_messages=10, keep_messages=5)
        messages = _make_messages(12)
        decision = evaluate_policy(policy, messages, 1000, None)
        assert decision is not None
        plan = _plan_compression(
            messages, decision, cutoff_fn=find_safe_cutoff, max_tokens=None,
        )
        assert plan is not None
        cutoff, to_evict, to_keep = plan

        prefetch_result = _PrefetchResult(
            cutoff=cutoff,
            message_count=len(messages),
            summary_wrapper="Prefetched summary wrapper",
            evictions=_build_evictions(to_evict),
            messages_summarized=len(to_evict),
            messages_kept=len(to_keep),
            summary_tokens=10,
            file_path=None,
        )

        async def _done() -> _PrefetchResult:
            return prefetch_result

        registry = PrefetchRegistry()
        registry._entries["prefetch-thread"] = _PrefetchEntry(
            message_count=len(messages),
            cutoff=cutoff,
            task=asyncio.create_task(_done()),
        )

        mock_model = AsyncMock()
        compress = make_compress_node(policy, mock_model, registry=registry)
        state: dict[str, Any] = {"messages": messages, "summary": ""}
        config = {"configurable": {"thread_id": "prefetch-thread"}}

        result = await compress(state, config)

        assert result["summary"] == "Prefetched summary wrapper"
        mock_model.ainvoke.assert_not_called()
        assert len(result["messages"]) > 0


# ---------------------------------------------------------------------------
# Tests: ArionState schema
# ---------------------------------------------------------------------------


class TestArionState:
    def test_state_has_summary_channel(self):
        from arion_agent.graph import ArionState
        hints = ArionState.__annotations__
        assert "summary" in hints

    def test_state_get_summary_default(self):
        state: dict[str, Any] = {"messages": []}
        assert state.get("summary", "") == ""


# ---------------------------------------------------------------------------
# Tests: Per-agent checkpointer path
# ---------------------------------------------------------------------------


class TestCheckpointerPath:
    def test_setup_creates_sqlite_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            identity_dir = Path(tmpdir) / "agents" / "test-agent"
            identity_dir.mkdir(parents=True)

            from arion_agent.graph import _setup_checkpointer
            saver = _setup_checkpointer(identity_dir, True)
            assert saver is not None

            db_path = identity_dir / "checkpoints.sqlite"
            assert db_path.exists()

            if hasattr(saver, "conn"):
                asyncio.run(saver.conn.close())

    def test_setup_returns_none_when_disabled(self):
        from arion_agent.graph import _setup_checkpointer
        assert _setup_checkpointer(Path("/tmp"), False) is None
        assert _setup_checkpointer(Path("/tmp"), None) is None


# ---------------------------------------------------------------------------
# Tests: Graph wiring
# ---------------------------------------------------------------------------


class TestGraphWiring:
    def test_graph_has_compress_node_when_configured(self):
        from arion_agent.graph import _build_react_graph, ArionState
        from arion_agent.summarization.policies import STANDARD_POLICY

        mock_model = MagicMock()
        registry = PrefetchRegistry()
        compression_config = {
            "route_compression": make_route_compression(STANDARD_POLICY),
            "prefetch_node": make_prefetch_node(
                STANDARD_POLICY, mock_model, registry,
            ),
            "compress_node": make_compress_node(
                STANDARD_POLICY, mock_model, registry=registry,
            ),
        }

        graph = _build_react_graph(
            mw_stack=[],
            all_tools=[],
            tool_map={},
            executor=MagicMock(),
            default_model_spec="test-model",
            extra_model_kwargs={},
            compression_config=compression_config,
        )

        compiled = graph.compile()
        node_names = set(compiled.get_graph().nodes.keys())
        assert "compress" in node_names
        assert "prefetch" in node_names
        assert "model" in node_names
        assert "tools" in node_names

    def test_graph_no_compress_when_disabled(self):
        from arion_agent.graph import _build_react_graph

        graph = _build_react_graph(
            mw_stack=[],
            all_tools=[],
            tool_map={},
            executor=MagicMock(),
            default_model_spec="test-model",
            extra_model_kwargs={},
            compression_config=None,
        )

        compiled = graph.compile()
        node_names = set(compiled.get_graph().nodes.keys())
        assert "compress" not in node_names
        assert "prefetch" not in node_names
        assert "model" in node_names
        assert "tools" in node_names


# ---------------------------------------------------------------------------
# Tests: Checkpoint cleanup
# ---------------------------------------------------------------------------


class TestCheckpointCleanup:
    def test_config_defaults(self):
        from arion_agent.util.checkpoint_cleanup import CheckpointCleanupConfig
        cfg = CheckpointCleanupConfig()
        assert cfg.keep_last == 100
        assert cfg.run_on_heartbeat is False

    @pytest.mark.asyncio
    async def test_prune_returns_zero_when_no_history(self):
        from arion_agent.util.checkpoint_cleanup import prune_checkpoints
        mock_graph = MagicMock()
        mock_graph.get_state_history.return_value = []
        result = await prune_checkpoints(mock_graph, "t1", keep_last=10)
        assert result == 0


# ---------------------------------------------------------------------------
# Integration: end-to-end graph with mock LLM
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_cycle_with_compression(self):
        """Build a real graph with mock LLM and verify compression fires."""
        from unittest.mock import patch

        from arion_agent.graph import ArionState, _build_react_graph
        from arion_agent.summarization.policies import STANDARD_POLICY
        from arion_agent.tool_manager.executor import ToolExecutor

        call_count = 0

        mock_summary_model = AsyncMock()
        mock_summary_model.ainvoke.return_value = MagicMock(content="Compressed summary.")

        policy = SummarizationPolicy(trigger_messages=10, keep_messages=5)

        registry = PrefetchRegistry()
        compression_config = {
            "route_compression": make_route_compression(policy),
            "prefetch_node": make_prefetch_node(
                policy, mock_summary_model, registry,
            ),
            "compress_node": make_compress_node(
                policy, mock_summary_model, registry=registry,
            ),
        }

        mock_main_model = AsyncMock()
        mock_main_model.bind_tools.return_value = mock_main_model
        mock_main_model.ainvoke.return_value = AIMessage(content="Done.", id="resp1")

        with patch("arion_agent.graph.resolve_model", return_value=mock_main_model):
            graph = _build_react_graph(
                mw_stack=[],
                all_tools=[],
                tool_map={},
                executor=ToolExecutor(),
                default_model_spec="mock",
                extra_model_kwargs={},
                compression_config=compression_config,
            )

            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()
            compiled = graph.compile(checkpointer=checkpointer)

            messages = _make_messages(15)
            config = {"configurable": {"thread_id": "e2e-test"}}

            result = await compiled.ainvoke(
                {"messages": messages, "summary": ""},
                config=config,
            )

        assert "summary" in result
        assert result["summary"] != ""
        assert mock_summary_model.ainvoke.called

    @pytest.mark.asyncio
    async def test_post_compression_patch_middleware_valid_sequence(self):
        """After compression, PatchToolCallsMiddleware produces a valid send list.

        Simulates the model_node flow: summary HumanMessage prepended,
        then PatchToolCallsMiddleware patches dangling/orphaned tool calls.
        The kept portion may start with AIMessage (not HumanMessage).
        """
        from arion_agent.middleware.patch_tool_calls import PatchToolCallsMiddleware

        mw = PatchToolCallsMiddleware()

        kept_messages = [
            AIMessage(content="working on it", id="a1"),
            AIMessage(
                content="", id="a2",
                tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "tc2"}],
            ),
            ToolMessage(content="found", tool_call_id="tc2", name="search", id="t3"),
            AIMessage(content="here is the answer", id="a4"),
            HumanMessage(content="thanks", id="h5"),
            AIMessage(content="you're welcome", id="a6"),
        ]

        summary = "Previous context: user asked for help with a search task."
        messages_for_llm = [HumanMessage(content=summary)] + kept_messages

        patched, _, _ = mw.wrap_model_call(messages_for_llm, [], model=None)

        assert len(patched) >= len(messages_for_llm)
        assert patched[0].content == summary
        assert isinstance(patched[0], HumanMessage)

        for i, m in enumerate(patched):
            if isinstance(m, ToolMessage):
                parent_found = any(
                    isinstance(patched[j], AIMessage)
                    and getattr(patched[j], "tool_calls", None)
                    and any(tc["id"] == m.tool_call_id for tc in patched[j].tool_calls)
                    for j in range(i)
                )
                assert parent_found, (
                    f"ToolMessage at {i} (tc={m.tool_call_id}) has no parent AIMessage "
                    f"in patched sequence"
                )

    @pytest.mark.asyncio
    async def test_post_compression_orphan_then_patch(self):
        """Layered defense: find_kept_orphans misses nothing, but even if it
        did, PatchToolCallsMiddleware injects a synthetic AIMessage.

        This tests PatchToolCallsMiddleware alone with a deliberately orphaned
        ToolMessage (simulating a hypothetical find_kept_orphans miss).
        """
        from arion_agent.middleware.patch_tool_calls import PatchToolCallsMiddleware

        mw = PatchToolCallsMiddleware()

        messages = [
            HumanMessage(content="summary goes here"),
            ToolMessage(content="orphan result", tool_call_id="tc_gone", name="lost_tool", id="t0"),
            AIMessage(content="continuing", id="a1"),
        ]

        patched, _, _ = mw.wrap_model_call(messages, [], model=None)

        tool_msg_indices = [i for i, m in enumerate(patched) if isinstance(m, ToolMessage) and m.tool_call_id == "tc_gone"]
        for ti in tool_msg_indices:
            has_parent = any(
                isinstance(patched[j], AIMessage)
                and getattr(patched[j], "tool_calls", None)
                and any(tc["id"] == "tc_gone" for tc in patched[j].tool_calls)
                for j in range(ti)
            )
            assert has_parent, (
                "PatchToolCallsMiddleware should inject synthetic AIMessage for orphaned ToolMessage"
            )

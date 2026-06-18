"""Test self-heal for corrupted tool_call sequences.

Reproduces the failure mode where Kimi (or any provider) leaves dangling
tool_calls or orphaned ToolMessages in the checkpoint, causing 400 errors
on subsequent API calls. Verifies three layers of defense:

  1. _sanitize_tool_call_ids strips additional_kwargs["tool_calls"] to prevent
     conflict between the explicit tool_calls property and raw API data.
  2. _patch_dangling_tool_calls / _patch_orphaned_tool_messages fix the
     message sequence before the API call.
  3. model_node self-heal catches 400 errors, re-patches, and retries.

Integration tests use xiaoha:gpt-5-mini (skip if XIAOHA_API_KEY not set).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from arion_agent import create_arion_agent
from arion_agent.graph import (
    _ai_message_has_tool_invocation,
    _ai_message_is_usable_response,
    _ai_message_text_content,
    _heal_tool_call_sequence,
    _is_corrupted_thought_signature_error,
    _parse_dangling_ids_from_error,
)
from arion_agent.middleware.patch_tool_calls import (
    PatchToolCallsMiddleware,
    _patch_dangling_tool_calls,
    _patch_orphaned_tool_messages,
    _sanitize_tool_call_ids,
)


@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about a programming language."""
    facts = {
        "python": "Python was created by Guido van Rossum in 1991.",
        "rust": "Rust 1.0 was released in May 2015.",
    }
    return facts.get(topic.lower().strip(), f"No info for '{topic}'.")


SOUL = (
    "You are a concise assistant. When asked about a language, use lookup_fact. "
    "State the fact, then stop."
)


# ---------------------------------------------------------------------------
# Unit tests: error detection and parsing
# ---------------------------------------------------------------------------

def test_ai_message_has_tool_invocation_gemini_empty_list():
    """Empty tool_calls list must not be treated as no tools (truthiness bug)."""
    from langchain_core.messages import AIMessage

    empty_tc = AIMessage(content=[], tool_calls=[], id="x")
    assert not _ai_message_has_tool_invocation(empty_tc)

    with_inv = AIMessage(
        content=[],
        tool_calls=[{"name": "wait", "args": {}, "id": "t1"}],
        id="y",
    )
    assert _ai_message_has_tool_invocation(with_inv)

    invalid_only = AIMessage(
        content="",
        tool_calls=[],
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "bad1",
                "name": "foo",
                "args": None,
                "error": "bad json",
            }
        ],
        id="z",
    )
    assert _ai_message_has_tool_invocation(invalid_only)


def test_ai_message_is_usable_response():
    assert not _ai_message_is_usable_response(AIMessage(content="", id="empty"))
    assert not _ai_message_is_usable_response(AIMessage(content=[], tool_calls=[], id="x"))
    assert _ai_message_is_usable_response(AIMessage(content="done", id="text"))
    assert _ai_message_is_usable_response(
        AIMessage(content="", tool_calls=[{"name": "wait", "args": {}, "id": "t1"}], id="tc")
    )
    assert _ai_message_is_usable_response(
        AIMessage(content=[{"type": "text", "text": "hello"}], id="blocks")
    )


def test_ai_message_text_content_list_blocks():
    msg = AIMessage(content=[{"type": "text", "text": "  hi  "}], id="b")
    assert _ai_message_text_content(msg) == "hi"


def test_corrupted_thought_signature_error_detection():
    assert _is_corrupted_thought_signature_error(
        Exception("400 INVALID_ARGUMENT ... Corrupted thought signature.")
    )
    assert not _is_corrupted_thought_signature_error(Exception("unrelated"))


def test_parse_dangling_ids():
    exc = Exception(
        "tool_call_ids did not have response messages: call_abc, call_def"
    )
    ids = _parse_dangling_ids_from_error(exc)
    assert set(ids) == {"call_abc", "call_def"}
    exc2 = Exception("tool_call_ids did not have response messages: wait:14")
    ids2 = _parse_dangling_ids_from_error(exc2)
    assert "wait:14" in ids2
    exc3 = Exception("tool_call_ids did not have response messages: wait_44 ")
    ids3 = _parse_dangling_ids_from_error(exc3)
    assert "wait_44" in ids3


# ---------------------------------------------------------------------------
# Unit tests: message patching
# ---------------------------------------------------------------------------

def test_patch_dangling_with_kimi_id():
    """Dangling tool_call with Kimi-style colon ID gets patched."""
    messages = [
        HumanMessage(content="hello"),
        AIMessage(
            content="Looking up...",
            tool_calls=[{"name": "wait", "args": {}, "id": "wait:14"}],
        ),
        HumanMessage(content="continue"),
    ]
    patched = _patch_dangling_tool_calls(messages)
    assert len(patched) == 4
    assert patched[2].type == "tool"
    assert patched[2].tool_call_id == "wait:14"


def test_patch_dangling_duplicate_id_non_contiguous():
    """Duplicate tool_call_id across separate conversation turns gets patched.

    Reproduces the real-world bug: Kimi reuses IDs like wait:44 across turns.
    The first occurrence at index 1 has no contiguous ToolMessage (interrupted
    by a HumanMessage). The second occurrence at index 4 has its ToolMessage.
    Both must have a matching ToolMessage in their contiguous block.
    """
    messages = [
        HumanMessage(content="start"),
        AIMessage(
            content="checking...",
            tool_calls=[{"name": "wait", "args": {}, "id": "wait:44"}],
        ),
        # No ToolMessage here — interrupted by human
        HumanMessage(content="hey, what happened?"),
        HumanMessage(content="try again"),
        AIMessage(
            content="",
            tool_calls=[{"name": "wait", "args": {}, "id": "wait:44"}],
        ),
        ToolMessage(content="Waited 20s.", tool_call_id="wait:44", name="wait"),
        HumanMessage(content="ok done"),
    ]
    patched = _patch_dangling_tool_calls(messages)
    # Index 1's AIMessage must get a synthetic ToolMessage injected right after
    assert patched[0].content == "start"
    assert patched[1].tool_calls[0]["id"] == "wait:44"
    assert patched[2].type == "tool"
    assert patched[2].tool_call_id == "wait:44"
    assert "cancelled" in patched[2].content

    # The second AIMessage + ToolMessage pair should remain intact
    second_ai_idx = next(
        i for i, m in enumerate(patched)
        if i > 2 and getattr(m, "type", "") == "ai"
        and getattr(m, "tool_calls", None)
        and m.tool_calls[0]["id"] == "wait:44"
    )
    assert patched[second_ai_idx + 1].type == "tool"
    assert patched[second_ai_idx + 1].tool_call_id == "wait:44"
    assert "Waited 20s." in patched[second_ai_idx + 1].content


def test_sanitize_strips_additional_kwargs_tool_calls():
    """_sanitize_tool_call_ids strips tool_calls from additional_kwargs."""
    raw_tc = [{"id": "wait:14", "type": "function",
               "function": {"name": "wait", "arguments": "{}"}}]
    msg = AIMessage(
        content="test",
        tool_calls=[{"name": "wait", "args": {}, "id": "wait:14"}],
        additional_kwargs={"tool_calls": raw_tc, "reasoning_content": "thinking"},
    )
    messages = [HumanMessage(content="hi"), msg]
    sanitized = _sanitize_tool_call_ids(messages)

    ai_msg = sanitized[1]
    assert ai_msg.tool_calls[0]["id"] == "wait_14"
    assert "tool_calls" not in ai_msg.additional_kwargs
    assert ai_msg.additional_kwargs.get("reasoning_content") == "thinking"


def test_patch_orphaned_tool_message():
    """Orphaned ToolMessage (no parent AIMessage) gets a synthetic AIMessage."""
    messages = [
        HumanMessage(content="hi"),
        ToolMessage(content="result", tool_call_id="call_orphan", name="test"),
        HumanMessage(content="continue"),
    ]
    patched = _patch_orphaned_tool_messages(messages)
    assert len(patched) == 4
    synthetic_ai = patched[1]
    assert synthetic_ai.type == "ai"
    assert synthetic_ai.tool_calls[0]["id"] == "call_orphan"


def test_heal_tool_call_sequence_dangling():
    """_heal_tool_call_sequence fixes dangling tool_calls from error IDs."""
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "wait", "args": {}, "id": "wait_44"}],
        ),
        HumanMessage(content="continue"),
    ]
    exc = Exception("tool_call_ids did not have response messages: wait_44")
    healed = _heal_tool_call_sequence(messages, exc)
    tool_msgs = [m for m in healed if getattr(m, "type", "") == "tool"]
    assert any(m.tool_call_id == "wait_44" for m in tool_msgs)


def test_heal_tool_call_sequence_orphaned():
    """_heal_tool_call_sequence fixes orphaned ToolMessages."""
    messages = [
        HumanMessage(content="hi"),
        ToolMessage(content="result", tool_call_id="call_x", name="test"),
        HumanMessage(content="continue"),
    ]
    exc = Exception("tool_call_id is not found")
    healed = _heal_tool_call_sequence(messages, exc)
    ai_msgs = [m for m in healed if getattr(m, "type", "") == "ai"]
    assert any(
        any(tc["id"] == "call_x" for tc in getattr(m, "tool_calls", []))
        for m in ai_msgs
    )


def test_convert_strips_tool_calls_from_kwargs():
    """Provider conversion strips additional_kwargs['tool_calls'] for all models."""
    from arion_agent.middleware.patch_tool_calls import _convert_ai_message_for_provider

    raw_tc = [{"id": "wait:14", "type": "function",
               "function": {"name": "wait", "arguments": "{}"}}]
    msg = AIMessage(
        content="test",
        tool_calls=[{"name": "wait", "args": {}, "id": "wait:14"}],
        additional_kwargs={"tool_calls": raw_tc, "reasoning_content": "ok"},
    )

    # Kimi target
    from arion_agent.providers.moonshot import ChatMoonshot
    kimi = ChatMoonshot(model="kimi-k2.5", api_key="test")
    converted = _convert_ai_message_for_provider(msg, kimi)
    assert "tool_calls" not in converted.additional_kwargs
    assert converted.additional_kwargs.get("reasoning_content")

    # OpenAI target (content-blocks model)
    from langchain_openai import ChatOpenAI
    gpt = ChatOpenAI(model="gpt-5-mini", api_key="test")
    converted = _convert_ai_message_for_provider(msg, gpt)
    assert "tool_calls" not in converted.additional_kwargs


# ---------------------------------------------------------------------------
# Integration: full middleware pipeline → xiaoha:gpt-5-mini
# ---------------------------------------------------------------------------

async def test_middleware_pipeline_dangling_xiaoha():
    """Dangling tool_call survives full middleware pipeline and API call.

    Simulates a corrupted checkpoint where Kimi left a dangling tool_call
    with a colon-containing ID. Patches messages through the middleware
    pipeline, then sends to xiaoha:gpt-5-mini to verify the API accepts.
    """
    if not os.environ.get("XIAOHA_API_KEY"):
        print("  Skipping (XIAOHA_API_KEY not set)")
        return

    from arion_agent.providers.resolver import resolve_model

    messages = [
        HumanMessage(content="Use lookup_fact to look up Python."),
        AIMessage(
            content="I'll look that up.",
            tool_calls=[{"name": "lookup_fact", "args": {"topic": "python"}, "id": "wait:14"}],
            additional_kwargs={
                "reasoning_content": "thinking about it",
                "tool_calls": [{"id": "wait:14", "type": "function",
                                "function": {"name": "lookup_fact",
                                             "arguments": '{"topic": "python"}'}}],
            },
            response_metadata={"provider": "ChatMoonshot"},
        ),
        # NO ToolMessage — this is the dangling tool_call
        HumanMessage(content="Please continue and answer."),
    ]

    model = resolve_model("xiaoha:gpt-5-mini")
    mw = PatchToolCallsMiddleware()
    patched, _, _ = mw.wrap_model_call(messages, [lookup_fact], model=model)

    tool_msgs = [m for m in patched if getattr(m, "type", "") == "tool"]
    assert tool_msgs, "Middleware should inject synthetic ToolMessage for dangling call"

    for m in patched:
        if getattr(m, "type", "") == "ai":
            assert "tool_calls" not in getattr(m, "additional_kwargs", {}), \
                "additional_kwargs['tool_calls'] should be stripped"

    print("  Structural checks passed. Calling xiaoha:gpt-5-mini...")
    bound = model.bind_tools([lookup_fact])
    response = await bound.ainvoke(patched)
    text = str(getattr(response, "content", ""))
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    print(f"  Response: {text[:200]}")
    print(f"  Has tool_calls: {has_tool_calls}")
    assert text.strip() or has_tool_calls, "API should return content or tool_calls"
    print("  PASSED")


async def test_middleware_pipeline_orphaned_xiaoha():
    """Orphaned ToolMessage survives full middleware pipeline and API call.

    Simulates a checkpoint where compression evicted the parent AIMessage
    but left the ToolMessage. Verifies middleware injects a synthetic
    AIMessage and the API call succeeds.
    """
    if not os.environ.get("XIAOHA_API_KEY"):
        print("  Skipping (XIAOHA_API_KEY not set)")
        return

    from arion_agent.providers.resolver import resolve_model

    messages = [
        HumanMessage(content="Look up Python."),
        ToolMessage(
            content="Python was created by Guido van Rossum in 1991.",
            name="lookup_fact",
            tool_call_id="call_orphan_123",
        ),
        HumanMessage(content="What did we learn?"),
    ]

    model = resolve_model("xiaoha:gpt-5-mini")
    mw = PatchToolCallsMiddleware()
    patched, _, _ = mw.wrap_model_call(messages, [lookup_fact], model=model)

    ai_msgs = [m for m in patched if getattr(m, "type", "") == "ai"]
    assert ai_msgs, "Middleware should inject synthetic AIMessage for orphaned ToolMessage"

    print("  Structural checks passed. Calling xiaoha:gpt-5-mini...")
    bound = model.bind_tools([lookup_fact])
    response = await bound.ainvoke(patched)
    text = str(getattr(response, "content", ""))
    print(f"  Response: {text[:200]}")
    assert text.strip(), "API should return a non-empty response"
    print("  PASSED")


async def test_full_agent_dangling_rescue_xiaoha():
    """Full agent invocation rescues a checkpoint with dangling tool_call.

    Uses two agents on a shared MemorySaver checkpoint. Agent A (Kimi-spec
    simulation) seeds the checkpoint with a clean turn, then we inject a
    corrupted AIMessage (dangling tool_call). Agent B (xiaoha:gpt-5-mini)
    invokes with a new user message and should self-heal through the
    middleware pipeline.
    """
    if not os.environ.get("XIAOHA_API_KEY"):
        print("  Skipping (XIAOHA_API_KEY not set)")
        return

    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    thread_id = "test-self-heal-rescue"

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="xiaoha:gpt-5-mini",
            workspace_dir=ws,
            agent_id="test-self-heal",
            soul=SOUL,
            tools=[lookup_fact],
            checkpointer=checkpointer,
            summarization=False,
        )

        # Turn 1: seed with a clean response
        print("  Turn 1: seed checkpoint...")
        result = await agent.ainvoke(
            {"messages": [("user", "Use lookup_fact to look up Python.")]},
            config={"configurable": {"thread_id": thread_id}},
        )
        ai_msgs = [m for m in result["messages"]
                    if getattr(m, "type", "") == "ai" and getattr(m, "content", "")]
        assert ai_msgs, "First turn should produce an AI response"
        print(f"  Turn 1 response: {str(ai_msgs[-1].content)[:120]}")

        # Inject a dangling tool_call into checkpoint via update_state
        print("  Injecting corrupted state (dangling tool_call)...")
        corrupt_ai = AIMessage(
            content="",
            tool_calls=[{"name": "lookup_fact", "args": {"topic": "rust"}, "id": "wait:99"}],
            additional_kwargs={
                "reasoning_content": " ",
                "tool_calls": [{"id": "wait:99", "type": "function",
                                "function": {"name": "lookup_fact",
                                             "arguments": '{"topic": "rust"}'}}],
            },
        )
        config = {"configurable": {"thread_id": thread_id}}
        await agent.aupdate_state(config, {"messages": [corrupt_ai]})

        # Turn 2: invoke with dangling tool_call in checkpoint — should self-heal
        print("  Turn 2: invoking with dangling tool_call in checkpoint...")
        result2 = await agent.ainvoke(
            {"messages": [("user", "Use lookup_fact to look up Rust.")]},
            config={"configurable": {"thread_id": thread_id}},
        )
        ai_msgs2 = [m for m in result2["messages"]
                     if getattr(m, "type", "") == "ai" and getattr(m, "content", "")]
        assert ai_msgs2, "Agent should respond despite corrupted checkpoint"
        print(f"  Turn 2 response: {str(ai_msgs2[-1].content)[:120]}")
        print("  PASSED — agent self-healed from dangling tool_call")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    # Unit tests (no API calls)
    unit_tests = [
        test_parse_dangling_ids,
        test_patch_dangling_with_kimi_id,
        test_sanitize_strips_additional_kwargs_tool_calls,
        test_patch_orphaned_tool_message,
        test_heal_tool_call_sequence_dangling,
        test_heal_tool_call_sequence_orphaned,
        test_convert_strips_tool_calls_from_kwargs,
    ]

    # Integration tests (xiaoha:gpt-5-mini)
    integration_tests = [
        test_middleware_pipeline_dangling_xiaoha,
        test_middleware_pipeline_orphaned_xiaoha,
        test_full_agent_dangling_rescue_xiaoha,
    ]

    failed = False

    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    for fn in unit_tests:
        name = fn.__name__
        try:
            fn()
            print(f"  {name}: PASSED")
        except Exception as exc:
            print(f"  {name}: FAILED — {exc}")
            import traceback
            traceback.print_exc()
            failed = True

    print()
    print("=" * 60)
    print("INTEGRATION TESTS (xiaoha:gpt-5-mini)")
    print("=" * 60)
    for fn in integration_tests:
        name = fn.__name__
        print(f"\n--- {name} ---")
        try:
            await fn()
        except Exception as exc:
            print(f"  FAILED — {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed = True

    print()
    print("=" * 60)
    if failed:
        print("SELF-HEAL TEST: SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("SELF-HEAL TEST: ALL PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

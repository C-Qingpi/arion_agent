"""Test hot-switching between model providers with tool-calling rounds.

Validates that switching models mid-conversation on a shared checkpoint
does not corrupt the message history.

Failure modes covered:
  1. reasoning_content / thinking-block incompatibility (fixed in moonshot.py):
     - Kimi requires reasoning_content on ALL assistant messages when thinking
       mode is active; messages from Claude/GPT/Gemini lack it.
     - Anthropic may produce list-format content blocks (thinking + text)
       that OpenAI-compatible APIs reject.
  2. Gemini 3+ thought-signature rejection on mid-turn switch:
     - Gemini 3+ validates thought signatures on function_call parts in the
       "active loop" (after the most recent user message). Non-Gemini function
       calls lack valid signatures; langchain-google-genai injects a stale
       DUMMY_THOUGHT_SIGNATURE which the API rejects ("Corrupted thought
       signature"). Fixed by collapsing foreign tool exchanges into text.

Tests:
  test_hot_switch_cycle: Kimi -> Claude -> GPT -> Gemini -> Kimi
    Each turn has its own user message, so foreign tool_calls are outside
    the active loop (validates normal per-turn hot-switch).
    Uses CloseAI proxy for Claude/GPT/Gemini.
  test_hot_switch_full_cycle: Kimi -> Claude -> GLM -> GPT -> Gemini -> Kimi
    All 5 model families on a shared thread, 2 tool calls per turn.
    Uses XiaoHa proxy + MooreThread GLM. Skipped if keys not set.
  test_mid_turn_switch_to_gemini: Claude tool_calls in Gemini's active loop
    Simulates mid-turn switch: Claude completed tool calls but Gemini continues
    the same turn. Without the fix, triggers "Corrupted thought signature".
  test_self_heal_collapses_gemini_on_thought_signature_error: after 400
    Corrupted thought signature, self-heal collapses native Gemini tool rounds.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from langchain_core.tools import tool

from arion_agent import create_arion_agent


@tool
def lookup_fact(topic: str) -> str:
    """Look up a fact about a programming language. Returns a short fact."""
    facts = {
        "python": "Python was created by Guido van Rossum and first released in 1991.",
        "rust": "Rust 1.0 was released in May 2015 by Mozilla Research.",
        "go": "Go was designed at Google by Griesemer, Pike, and Thompson in 2009.",
        "javascript": "JavaScript was created by Brendan Eich in 10 days in 1995.",
    }
    return facts.get(topic.lower().strip(), f"No information available for '{topic}'.")


SOUL = (
    "You are a concise research assistant. "
    "When asked about a programming language, ALWAYS use the lookup_fact tool. "
    "State the fact you receive from the tool, then stop. "
    "Do not make up facts. Do not use any tool besides lookup_fact."
)

MODELS = [
    ("Kimi",   "moonshot:kimi-k2.5"),
    ("Claude", "anthropic:claude-haiku-4-5"),
    ("GPT",    "openai:gpt-5-mini"),
    ("Gemini", conftest.get_test_model("gemini-3-flash-preview")),
]

# Full cross-provider hot-switch cycle: all 5 model families, 2 tool calls per turn.
# Two tool calls per turn is critical: reasoning_content / thinking-block bugs only
# manifest when the checkpoint contains multi-tool-call AI messages from a foreign
# provider. Varied fact pairings test recombination across different model histories.
# Skipped if XIAOHA_API_KEY or MOORETHREAD_API_KEY is unset.
FULL_CYCLE_MODELS = [
    ("Kimi",           "moonshot:kimi-k2.5"),
    ("XiaoHa Claude",  "xiaoha:claude-haiku-4-5-20251001"),
    ("GLM 4.7",        "moorethread:glm-4.7"),
    ("XiaoHa GPT",     "xiaoha:gpt-5-mini"),
    ("XiaoHa Gemini",  "xiaoha:gemini-3-flash-preview"),
]
FULL_CYCLE_TURNS = [
    {"prompt": "Use lookup_fact to look up Python, then use it again to look up Rust.",       "expect_any": ["1991", "2015"]},
    {"prompt": "Use lookup_fact to look up Go, then use it again for JavaScript.",            "expect_any": ["2009", "1995"]},
    {"prompt": "Use lookup_fact to look up Python, then use it again to look up Go.",         "expect_any": ["1991", "2009"]},
    {"prompt": "Use lookup_fact to look up Rust, then use it again for JavaScript.",          "expect_any": ["2015", "1995"]},
    {"prompt": "Use lookup_fact to look up Python, then use it again for JavaScript.",        "expect_any": ["1991", "1995"]},
]

TURNS = [
    {
        "prompt": "Use lookup_fact to look up Python, then use it again to look up Rust.",
        "expect_any": ["1991", "2015"],
    },
    {
        "prompt": "Use lookup_fact to look up Go, then use it again for JavaScript.",
        "expect_any": ["2009", "1995"],
    },
    {
        "prompt": "Use lookup_fact to look up Python one more time.",
        "expect_any": ["1991"],
    },
    {
        "prompt": "Use lookup_fact to look up Go one more time.",
        "expect_any": ["2009"],
    },
]


def _make_agent(model_spec, ws, checkpointer):
    return create_arion_agent(
        model=model_spec,
        agent_id="hot-switch-agent",
        soul=SOUL,
        workspace_dir=ws,
        tools=[lookup_fact],
        checkpointer=checkpointer,
        summarization=False,
    )


def _print_new_messages(messages: list, offset: int) -> None:
    for msg in messages[offset:]:
        role = getattr(msg, "type", "?")
        tc = getattr(msg, "tool_calls", None)
        content = getattr(msg, "content", "")
        if tc:
            names = [t["name"] for t in tc]
            print(f"  [{role}] tool_calls={names}")
        elif content:
            text = content if isinstance(content, str) else str(content)
            print(f"  [{role}] {text[:180]}")


def _check_turn(label: str, messages: list, offset: int, expect_any: list[str]) -> None:
    new_msgs = messages[offset:]

    had_tool_call = any(
        getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None)
        for m in new_msgs
    )
    assert had_tool_call, f"{label}: expected tool calls but none occurred"

    ai_msgs = [
        m for m in new_msgs
        if getattr(m, "type", "") == "ai" and getattr(m, "content", "")
    ]
    last_content = str(ai_msgs[-1].content).lower() if ai_msgs else ""
    found = [k for k in expect_any if k in last_content]
    assert found, f"{label}: expected any of {expect_any} in response, got: {last_content[:200]}"

    print(f"  tools: yes | keywords: {found} | PASSED")


async def test_hot_switch_cycle():
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    thread_id = "hot-switch-test"
    msg_offset = 0

    with tempfile.TemporaryDirectory() as ws:
        for i, (label, model_spec) in enumerate(MODELS):
            turn = TURNS[i]
            print(f"\n{'='*60}")
            print(f"Turn {i+1}: {label}")
            print(f"{'='*60}")

            agent = _make_agent(model_spec, ws, checkpointer)
            result = await agent.ainvoke(
                {"messages": [("user", turn["prompt"])]},
                config={"configurable": {"thread_id": thread_id}},
            )
            all_msgs = result["messages"]
            _print_new_messages(all_msgs, msg_offset)
            _check_turn(label, all_msgs, msg_offset, turn["expect_any"])
            msg_offset = len(all_msgs)

        # Return trip: back to Kimi after Claude+GPT+Gemini messages in history
        print(f"\n{'='*60}")
        print("Turn 5: Kimi (return trip -- regression test)")
        print(f"{'='*60}")

        agent = _make_agent("moonshot:kimi-k2.5", ws, checkpointer)
        result = await agent.ainvoke(
            {"messages": [("user",
                "Use lookup_fact to look up JavaScript one more time."
            )]},
            config={"configurable": {"thread_id": thread_id}},
        )
        all_msgs = result["messages"]
        _print_new_messages(all_msgs, msg_offset)
        _check_turn("Kimi (return)", all_msgs, msg_offset, ["1995"])


async def test_hot_switch_full_cycle():
    """Full cross-provider hot-switch: Kimi -> Claude -> GLM -> GPT -> Gemini -> Kimi.

    All 5 model families on a shared thread, 2 tool calls per turn. This exercises:
      - reasoning_content round-trip (Kimi requires it on all assistant messages)
      - thinking-block flattening (Claude list-format -> OpenAI-compatible string)
      - thought-signature handling (Gemini 3+ rejects foreign function_call parts)
      - multi-tool-call checkpoint compatibility across all provider combinations

    Requires XIAOHA_API_KEY and MOORETHREAD_API_KEY in tests/.env.
    Skipped if either key is unset or if a proxy returns 401.
    """
    if not os.environ.get("XIAOHA_API_KEY"):
        print("  Skipping (XIAOHA_API_KEY not set)")
        return
    if not os.environ.get("MOORETHREAD_API_KEY"):
        print("  Skipping (MOORETHREAD_API_KEY not set)")
        return

    from langgraph.checkpoint.memory import MemorySaver

    try:
        from openai import AuthenticationError as OpenAIAuthError
    except ImportError:
        OpenAIAuthError = ()

    checkpointer = MemorySaver()
    thread_id = "hot-switch-full-cycle-test"
    msg_offset = 0

    with tempfile.TemporaryDirectory() as ws:
        for i, (label, model_spec) in enumerate(FULL_CYCLE_MODELS):
            turn = FULL_CYCLE_TURNS[i]
            print(f"\n{'='*60}")
            print(f"Turn {i+1}: {label} ({model_spec})")
            print(f"{'='*60}")

            agent = _make_agent(model_spec, ws, checkpointer)
            try:
                result = await agent.ainvoke(
                    {"messages": [("user", turn["prompt"])]},
                    config={"configurable": {"thread_id": thread_id}},
                )
            except Exception as e:
                if OpenAIAuthError and isinstance(e, OpenAIAuthError):
                    print(f"  Skipping ({label} returned 401, key may be invalid/expired)")
                    return
                raise
            all_msgs = result["messages"]
            _print_new_messages(all_msgs, msg_offset)
            _check_turn(label, all_msgs, msg_offset, turn["expect_any"])
            msg_offset = len(all_msgs)

        # Return trip: back to Kimi after all 4 other providers' messages in history.
        # This is the hardest case for reasoning_content: Kimi must handle checkpoint
        # containing Claude thinking-blocks, GPT plain strings, Gemini thought-signature
        # parts, and GLM responses -- all with 2 tool calls per turn.
        print(f"\n{'='*60}")
        print("Turn 6: Kimi (return trip -- regression test)")
        print(f"{'='*60}")

        agent = _make_agent("moonshot:kimi-k2.5", ws, checkpointer)
        result = await agent.ainvoke(
            {"messages": [("user",
                "Use lookup_fact to look up Rust, then use it again for Go."
            )]},
            config={"configurable": {"thread_id": thread_id}},
        )
        all_msgs = result["messages"]
        _print_new_messages(all_msgs, msg_offset)
        _check_turn("Kimi (return)", all_msgs, msg_offset, ["2015", "2009"])


async def test_mid_turn_switch_to_gemini():
    """Reproduce the 'Corrupted thought signature' bug (mid-turn Claude->Gemini).

    Scenario: Claude processed a user message, made tool calls, tools executed.
    Before Claude gives its final response, model switches to Gemini. Gemini
    sees Claude's AIMessage(tool_calls) + ToolMessages in the same turn (inside
    Gemini's "active loop"). Without the fix, langchain-google-genai injects a
    stale DUMMY_THOUGHT_SIGNATURE on Claude's function_call parts and the Gemini
    API returns 400 "Corrupted thought signature."

    The fix collapses foreign tool exchanges into text, avoiding function_call
    parts entirely. This test verifies:
      1. Middleware strips tool_calls from the Claude AIMessage
      2. ToolMessages become HumanMessages with tool result text
      3. The Gemini API call succeeds
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from arion_agent.middleware.patch_tool_calls import PatchToolCallsMiddleware

    # Construct the problematic state: Claude's tool_calls are in the same
    # turn as the user message (no separating HumanMessage), so they fall
    # inside Gemini's active loop.
    messages = [
        HumanMessage(content="Use lookup_fact to look up Python."),
        AIMessage(
            content=[{"type": "text", "text": "I'll look up Python for you."}],
            tool_calls=[{
                "name": "lookup_fact",
                "args": {"topic": "python"},
                "id": "call_claude_abc123",
            }],
            response_metadata={"provider": "ChatAnthropic"},
        ),
        ToolMessage(
            content="Python was created by Guido van Rossum and first released in 1991.",
            name="lookup_fact",
            tool_call_id="call_claude_abc123",
        ),
    ]

    gemini_model = conftest.get_test_model("gemini-3-flash-preview")
    mw = PatchToolCallsMiddleware()

    patched, _, _ = mw.wrap_model_call(
        messages, [lookup_fact], model=gemini_model,
    )

    # -- Structural checks --

    for msg in patched:
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            origin = (getattr(msg, "response_metadata", {}) or {}).get("provider")
            assert origin == "ChatGoogleGenerativeAI", (
                f"Non-Gemini tool_calls not collapsed: provider={origin}"
            )

    ai_msgs = [m for m in patched if getattr(m, "type", "") == "ai"]
    assert ai_msgs, "Expected at least one AI message after conversion"
    assert not getattr(ai_msgs[0], "tool_calls", None), (
        "Claude AIMessage should have tool_calls removed"
    )

    human_msgs = [m for m in patched if getattr(m, "type", "") == "human"]
    has_tool_result = any("1991" in getattr(m, "content", "") for m in human_msgs)
    assert has_tool_result, "Tool result should appear in a converted HumanMessage"

    print("  Structural checks PASSED")

    # -- API integration: actually call Gemini with the patched messages --

    print("  Calling Gemini API with patched messages...")
    bound = gemini_model.bind_tools([lookup_fact])
    response = await bound.ainvoke(patched)
    text = str(response.content)
    print(f"  Response: {text[:200]}")
    assert text.strip(), "Gemini should produce a non-empty response"
    print("  API call PASSED (no 'Corrupted thought signature' error)")


async def test_self_heal_collapses_gemini_on_thought_signature_error():
    """Default pass collapses all tool exchanges; self-heal does the same.

    All tool exchanges (foreign + native Gemini) are collapsed to text on
    the default pass to avoid stale thought signatures. Self-heal path
    uses the same logic as a fallback.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from arion_agent.graph import _heal_tool_call_sequence
    from arion_agent.middleware.patch_tool_calls import PatchToolCallsMiddleware

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        import pytest
        pytest.skip("langchain_google_genai not installed")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        google_api_key="x",
        client_options={"api_endpoint": "http://127.0.0.1:9"},
    )

    messages = [
        HumanMessage(content="Call lookup_fact for Python."),
        AIMessage(
            content=[{"type": "text", "text": "Looking up Python."}],
            tool_calls=[{
                "name": "lookup_fact",
                "args": {"topic": "python"},
                "id": "call_native_1",
            }],
            response_metadata={"provider": "ChatGoogleGenerativeAI"},
            additional_kwargs={},
        ),
        ToolMessage(
            content="Python 1991.",
            name="lookup_fact",
            tool_call_id="call_native_1",
        ),
        HumanMessage(content="Now look up Rust."),
        AIMessage(
            content=[{"type": "text", "text": "Looking up Rust."}],
            tool_calls=[{
                "name": "lookup_fact",
                "args": {"topic": "rust"},
                "id": "call_native_2",
            }],
            response_metadata={"provider": "ChatGoogleGenerativeAI"},
            additional_kwargs={},
        ),
        ToolMessage(
            content="Rust 2010.",
            name="lookup_fact",
            tool_call_id="call_native_2",
        ),
    ]
    mw = PatchToolCallsMiddleware()
    patched, _, _ = mw.wrap_model_call(messages, [], model=model)

    # All native exchanges should be collapsed (no AI with tool_calls remains)
    assert not any(
        getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None)
        for m in patched
    ), "All native Gemini tool exchanges should be collapsed to text"

    # Tool results should appear as HumanMessages
    human_texts = [
        str(getattr(m, "content", ""))
        for m in patched if getattr(m, "type", "") == "human"
    ]
    assert any("1991" in t for t in human_texts), "First tool result should be in HumanMessage"
    assert any("2010" in t for t in human_texts), "Second tool result should be in HumanMessage"

    # Self-heal also collapses everything (same behavior, used as fallback)
    exc = Exception(
        "400 INVALID_ARGUMENT. {'error': {'message': 'Corrupted thought signature.'}}"
    )
    healed = _heal_tool_call_sequence(patched, exc, model)
    assert not any(
        getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None)
        for m in healed
    )


async def test_gemini_collapse_preserves_thought_in_collapsed_ai():
    """Collapsed tool rounds must keep thought/thinking text, not only type:text blocks."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from arion_agent.middleware.patch_tool_calls import PatchToolCallsMiddleware

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        import pytest
        pytest.skip("langchain_google_genai not installed")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        google_api_key="x",
        client_options={"api_endpoint": "http://127.0.0.1:9"},
    )

    messages = [
        HumanMessage(content="Call lookup_fact."),
        AIMessage(
            content=[
                {"type": "thought", "thought": "Internal reasoning before tool."},
                {"type": "text", "text": "Calling lookup."},
            ],
            tool_calls=[{
                "name": "lookup_fact",
                "args": {"topic": "python"},
                "id": "call_th_1",
            }],
            response_metadata={"provider": "ChatGoogleGenerativeAI"},
            additional_kwargs={},
        ),
        ToolMessage(
            content="Python 1991.",
            name="lookup_fact",
            tool_call_id="call_th_1",
        ),
    ]
    mw = PatchToolCallsMiddleware()
    patched, _, _ = mw.wrap_model_call(messages, [], model=model)

    collapsed_ai = [
        m for m in patched
        if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", None)
    ]
    assert len(collapsed_ai) == 1
    body = str(collapsed_ai[0].content)
    assert "Internal reasoning before tool." in body
    assert "Calling lookup." in body


async def test_mid_turn_switch_multi_round():
    """Same as above but with multiple rounds of tool calls from Claude."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from arion_agent.middleware.patch_tool_calls import PatchToolCallsMiddleware

    messages = [
        HumanMessage(content="Look up Python and Rust using lookup_fact."),
        AIMessage(
            content=[{"type": "text", "text": "Looking up Python first."}],
            tool_calls=[{
                "name": "lookup_fact",
                "args": {"topic": "python"},
                "id": "call_round1",
            }],
            response_metadata={"provider": "ChatAnthropic"},
        ),
        ToolMessage(
            content="Python was created by Guido van Rossum and first released in 1991.",
            name="lookup_fact",
            tool_call_id="call_round1",
        ),
        AIMessage(
            content=[{"type": "text", "text": "Now looking up Rust."}],
            tool_calls=[{
                "name": "lookup_fact",
                "args": {"topic": "rust"},
                "id": "call_round2",
            }],
            response_metadata={"provider": "ChatAnthropic"},
        ),
        ToolMessage(
            content="Rust 1.0 was released in May 2015 by Mozilla Research.",
            name="lookup_fact",
            tool_call_id="call_round2",
        ),
    ]

    gemini_model = conftest.get_test_model("gemini-3-flash-preview")
    mw = PatchToolCallsMiddleware()

    patched, _, _ = mw.wrap_model_call(
        messages, [lookup_fact], model=gemini_model,
    )

    for msg in patched:
        if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
            assert False, "Foreign tool_calls not collapsed in multi-round scenario"

    ai_msgs = [m for m in patched if getattr(m, "type", "") == "ai"]
    human_msgs = [m for m in patched if getattr(m, "type", "") == "human"]
    assert len(ai_msgs) >= 2, "Expected AI messages for each collapsed round"
    assert len(human_msgs) >= 2, (
        "Expected HumanMessages for each collapsed ToolMessage"
    )

    has_python = any("1991" in getattr(m, "content", "") for m in human_msgs)
    has_rust = any("2015" in getattr(m, "content", "") for m in human_msgs)
    assert has_python and has_rust, "Both tool results should be in HumanMessages"

    # Verify alternating message roles (no consecutive same-role)
    types = [getattr(m, "type", "") for m in patched]
    for i in range(1, len(types)):
        if types[i] == types[i - 1] and types[i] in ("ai", "human"):
            assert False, (
                f"Consecutive {types[i]} messages at positions {i-1},{i}: "
                f"collapsed messages should maintain alternating sequence"
            )

    print("  Multi-round structural checks PASSED")

    print("  Calling Gemini API with multi-round patched messages...")
    bound = gemini_model.bind_tools([lookup_fact])
    response = await bound.ainvoke(patched)
    text = str(response.content)
    print(f"  Response: {text[:200]}")
    assert text.strip(), "Gemini should produce a non-empty response"
    print("  Multi-round API call PASSED")


async def main():
    failed = False

    for test_fn in [
        test_hot_switch_cycle,
        test_hot_switch_full_cycle,
        test_mid_turn_switch_to_gemini,
        test_self_heal_collapses_gemini_on_thought_signature_error,
        test_gemini_collapse_preserves_thought_in_collapsed_ai,
        test_mid_turn_switch_multi_round,
    ]:
        name = test_fn.__name__
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        try:
            await test_fn()
        except Exception as exc:
            print(f"\n>> FAILED: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed = True

    print(f"\n{'='*60}")
    if failed:
        print("HOT-SWITCH TEST: SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("HOT-SWITCH TEST COMPLETE: ALL PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

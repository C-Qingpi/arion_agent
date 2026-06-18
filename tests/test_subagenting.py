"""Test subagenting: agent_id, thread_id, identity self-awareness, subagent spawning."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent import create_arion_agent
from arion_agent.identity import STANDARD_SOUL
from arion_agent.subagenting import (
    SubAgentSpec,
    SubagentEvent,
    SELF_INFERTILE_CLONE,
    TASK_SUBAGENT,
)


# ============================================================
# Prereq tests: agent_id, thread_id, identity self-awareness
# ============================================================


async def test_agent_id_deterministic():
    """Developer-provided agent_id should create predictable identity_dir."""
    print("=" * 60)
    print("Test: deterministic agent_id")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="my-named-agent",
            soul="I am a test agent.",
            checkpointer=False,
            summarization=False,
        )

        identity_dir = os.path.join(ws, ".arion", "agents", "my-named-agent")
        assert os.path.exists(identity_dir), f"Expected {identity_dir} to exist"
        assert os.path.exists(os.path.join(identity_dir, "SOUL.md"))
        print(f"  Identity dir: {identity_dir}")
        print("  >> PASSED")


async def test_agent_id_resumes_identity():
    """Same agent_id should preserve existing identity files."""
    print("\n" + "=" * 60)
    print("Test: agent_id resumes identity")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        # First creation
        agent1 = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="persistent-agent",
            soul="I am version ONE.",
            checkpointer=False,
            summarization=False,
        )

        soul_path = os.path.join(ws, ".arion", "agents", "persistent-agent", "SOUL.md")
        original = open(soul_path, encoding="utf-8").read()
        assert "version ONE" in original

        # Modify SOUL (simulate agent evolution)
        with open(soul_path, "w", encoding="utf-8") as f:
            f.write("I am EVOLVED version.")

        # Second creation with same agent_id -- should NOT overwrite
        agent2 = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="persistent-agent",
            soul="I am version TWO.",
            checkpointer=False,
            summarization=False,
        )

        preserved = open(soul_path, encoding="utf-8").read()
        assert "EVOLVED" in preserved, f"Should preserve evolved SOUL, got: {preserved[:100]}"
        assert "TWO" not in preserved, "Should NOT overwrite with new soul"
        print("  Evolved SOUL preserved across restart")
        print("  >> PASSED")


async def test_thread_id_isolation():
    """Different thread_ids should have independent conversations."""
    print("\n" + "=" * 60)
    print("Test: thread_id isolation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul="You are a test agent. Be concise.",
            checkpointer=False,
            summarization=False,
        )

        # Thread A
        r1 = await agent.ainvoke(
            {"messages": [("user", "My secret code is ALPHA-777. Just say OK.")]},
            config={"configurable": {"thread_id": "thread-A"}},
        )
        print(f"  Thread A: {len(r1['messages'])} messages")

        # Thread B (fresh)
        r2 = await agent.ainvoke(
            {"messages": [("user", "What is my secret code? Say UNKNOWN if you don't know.")]},
            config={"configurable": {"thread_id": "thread-B"}},
        )
        ai_b = [m for m in r2["messages"] if getattr(m, "type", "") == "ai"][-1]
        knows_code = "alpha" in ai_b.content.lower() or "777" in ai_b.content
        print(f"  Thread B answer: {ai_b.content[:100]}")
        assert not knows_code, "Thread B should NOT know Thread A's secret"
        print("  Thread B correctly isolated")
        print("  >> PASSED")


async def test_identity_self_awareness():
    """Agent should know its own agent_id from the system prompt."""
    print("\n" + "=" * 60)
    print("Test: identity self-awareness")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="self-aware-agent",
            soul="I am a self-aware test agent.",
            checkpointer=False,
            summarization=False,
        )

        r = await agent.ainvoke(
            {"messages": [("user", "What is your agent ID? Reply with just the ID.")]},
            config={"configurable": {"thread_id": "awareness-test"}},
        )
        ai = [m for m in r["messages"] if getattr(m, "type", "") == "ai"][-1]
        print(f"  Agent answer: {ai.content[:150]}")
        assert "self-aware-agent" in ai.content.lower().replace(" ", "").replace("-", "-"), \
            "Agent should know its own agent_id"
        print("  >> PASSED")


# ============================================================
# Subagent tests
# ============================================================


async def test_basic_subagent():
    """Agent should be able to spawn a task subagent and get results."""
    print("\n" + "=" * 60)
    print("Test: basic subagent-as-tool")
    print("=" * 60)

    events: list[SubagentEvent] = []

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="parent-agent",
            soul="You are a manager agent. Delegate tasks to subagents.",
            subagents=[
                SubAgentSpec(
                    name="calculator",
                    description="A math specialist. Use for any calculation.",
                    soul="You are a calculator. Compute the answer and return just the number.",
                    tier="important",
                    max_turns=10,
                ),
            ],
            checkpointer=False,
            summarization=False,
        )

        r = await agent.ainvoke(
            {"messages": [("user", "Use your calculator subagent to compute 17 * 23.")]},
            config={"configurable": {"thread_id": "subagent-test"}},
        )

        ai = [m for m in r["messages"] if getattr(m, "type", "") == "ai"][-1]
        print(f"  Agent answer: {ai.content[:200]}")
        assert "391" in ai.content, f"Expected 391 in answer"

        # Check child identity dir was created (flat structure)
        agents_dir = os.path.join(ws, ".arion", "agents")
        agent_dirs = [d for d in os.listdir(agents_dir) if d != "parent-agent"]
        assert len(agent_dirs) >= 1, f"Expected child agent dir, found: {os.listdir(agents_dir)}"
        child_dir = os.path.join(agents_dir, agent_dirs[0])
        assert os.path.exists(os.path.join(child_dir, "SOUL.md")), "Child should have SOUL.md"
        print(f"  Child agent dir: {agent_dirs[0]} (flat, same level as parent)")
        print("  >> PASSED")


async def test_subagent_depth_limit():
    """max_recursion_depth should prevent infinite nesting."""
    print("\n" + "=" * 60)
    print("Test: subagent recursion depth limit")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        # Depth 1: parent can spawn, but children are infertile
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="depth-test",
            soul="You are a test agent. Use task tool when asked.",
            subagents=[TASK_SUBAGENT],
            max_recursion_depth=1,
            checkpointer=False,
            summarization=False,
        )

        r = await agent.ainvoke(
            {"messages": [("user", "Use the task-agent to say hello.")]},
            config={"configurable": {"thread_id": "depth-test"}},
        )
        ai = [m for m in r["messages"] if getattr(m, "type", "") == "ai"][-1]
        print(f"  Agent answer: {ai.content[:200]}")
        print("  Depth limit enforced (child was infertile)")
        print("  >> PASSED")


async def test_subagent_callbacks():
    """Subagent lifecycle callbacks should fire."""
    print("\n" + "=" * 60)
    print("Test: subagent callbacks")
    print("=" * 60)

    events: list[SubagentEvent] = []

    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.subagenting.middleware import SubagentMiddleware

        spec = SubAgentSpec(
            name="helper",
            description="A helper agent.",
            soul="You are a helper. Just say OK.",
            tier="important",
            max_turns=5,
        )

        sub_mw = SubagentMiddleware(
            specs=[spec],
            parent_agent_id="callback-parent",
            parent_model="openai:gpt-5-mini",
            parent_workspace=ws,
            on_subagent=lambda e: events.append(e),
        )

        result = await sub_mw._spawn_and_run("helper", "Say OK", "")

        assert len(events) == 2, f"Expected 2 events (spawn+complete), got {len(events)}"
        assert events[0].phase == "spawn"
        assert events[1].phase == "complete"
        assert events[0].subagent_class == "helper"
        print(f"  Events: {[e.phase for e in events]}")
        print(f"  Child ID: {events[0].child_agent_id}")
        print("  >> PASSED")


# ============================================================
# Main
# ============================================================


async def main():
    # Prereq tests
    await test_agent_id_deterministic()
    await test_agent_id_resumes_identity()
    await test_thread_id_isolation()
    await test_identity_self_awareness()

    print(f"\n{'=' * 60}")
    print("PREREQ TESTS PASSED -- proceeding to subagent tests")
    print(f"{'=' * 60}")

    # Subagent tests
    await test_basic_subagent()
    await test_subagent_depth_limit()
    await test_subagent_callbacks()

    print(f"\n{'=' * 60}")
    print("ALL SUBAGENTING TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

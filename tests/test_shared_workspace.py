"""Test two agents sharing the same workspace."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent import create_arion_agent


async def test_two_agents_shared_workspace():
    """Two agents with different agent_ids share one workspace."""
    print("=" * 60)
    print("Test: two agents in same workspace")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        agent_a = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="agent-alpha",
            soul="I am Alpha. I write files.",
            summarization=False,
            checkpointer=False,
        )
        agent_b = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="agent-beta",
            soul="I am Beta. I read files.",
            summarization=False,
            checkpointer=False,
        )

        # Both identity dirs exist, separate
        alpha_dir = os.path.join(ws, ".arion", "agents", "agent-alpha")
        beta_dir = os.path.join(ws, ".arion", "agents", "agent-beta")
        assert os.path.isdir(alpha_dir), "Alpha identity dir should exist"
        assert os.path.isdir(beta_dir), "Beta identity dir should exist"

        alpha_soul = open(os.path.join(alpha_dir, "SOUL.md"), encoding="utf-8").read()
        beta_soul = open(os.path.join(beta_dir, "SOUL.md"), encoding="utf-8").read()
        assert "Alpha" in alpha_soul
        assert "Beta" in beta_soul
        print("  Separate identity dirs: OK")

        # Alpha writes a file
        r1 = await agent_a.ainvoke(
            {"messages": [("user", "Write a file called shared_note.txt with the text: Hello from Alpha")]},
        )
        shared_file = os.path.join(ws, "shared_note.txt")
        assert os.path.exists(shared_file), "Alpha should create shared_note.txt"
        content = open(shared_file, encoding="utf-8").read()
        assert "Alpha" in content
        print("  Alpha wrote shared_note.txt: OK")

        # Beta reads the same file
        r2 = await agent_b.ainvoke(
            {"messages": [("user", "Read the file shared_note.txt and tell me what it says.")]},
        )
        ai_b = [m for m in r2["messages"] if getattr(m, "type", "") == "ai"][-1]
        assert "alpha" in ai_b.content.lower() or "Alpha" in ai_b.content, \
            f"Beta should see Alpha's content, got: {ai_b.content[:150]}"
        print(f"  Beta read shared_note.txt: {ai_b.content[:80]}")

        # Shared terminals directory
        term_dir = os.path.join(ws, ".arion", "terminals")
        assert os.path.isdir(term_dir), "Shared terminals dir should exist"
        print("  Shared terminals dir: OK")

        # Stats on both
        if hasattr(agent_a, "stats"):
            print(f"  Alpha stats: model_calls={agent_a.stats.model_calls}, tool_calls={agent_a.stats.tool_calls}")
        if hasattr(agent_b, "stats"):
            print(f"  Beta stats: model_calls={agent_b.stats.model_calls}, tool_calls={agent_b.stats.tool_calls}")

        # Both agent_ids are accessible
        assert agent_a.agent_id == "agent-alpha"
        assert agent_b.agent_id == "agent-beta"
        print("  agent_id accessible on compiled graph: OK")

        print("  >> PASSED")


async def test_two_agents_independent_conversations():
    """Two agents share workspace but have independent conversations."""
    print("\n" + "=" * 60)
    print("Test: independent conversations in shared workspace")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        agent_a = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="conv-alpha",
            soul="I am Alpha.",
            summarization=False,
            checkpointer=False,
        )
        agent_b = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="conv-beta",
            soul="I am Beta.",
            summarization=False,
            checkpointer=False,
        )

        # Alpha gets a secret
        r1 = await agent_a.ainvoke(
            {"messages": [("user", "My password is SUNSHINE-42. Just say OK.")]},
        )
        print(f"  Alpha: {len(r1['messages'])} messages")

        # Beta should NOT know Alpha's secret
        r2 = await agent_b.ainvoke(
            {"messages": [("user", "What is my password? Say UNKNOWN if you don't know.")]},
        )
        ai_b = [m for m in r2["messages"] if getattr(m, "type", "") == "ai"][-1]
        knows_secret = "sunshine" in ai_b.content.lower() or "42" in ai_b.content
        assert not knows_secret, f"Beta should NOT know Alpha's secret, got: {ai_b.content[:100]}"
        print(f"  Beta answer: {ai_b.content[:80]}")
        print("  Conversations are independent: OK")
        print("  >> PASSED")


async def main():
    await test_two_agents_shared_workspace()
    await test_two_agents_independent_conversations()
    print(f"\n{'=' * 60}")
    print("ALL SHARED WORKSPACE TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

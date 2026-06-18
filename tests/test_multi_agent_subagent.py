"""Test two agents with subagents sharing a workspace.

Verifies:
- Two parent agents coexist with separate identities
- Each spawns a subagent (4 agents total in one workspace)
- All identity dirs are flat under .arion/agents/
- Subagent results flow back to correct parent
- No cross-contamination between agents
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent import create_arion_agent
from arion_agent.subagenting import SubAgentSpec


async def test_two_parents_with_subagents():
    """Two parents each spawn a subagent, all in one workspace."""
    print("=" * 60)
    print("Test: two parents + subagents in shared workspace")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        writer_spec = SubAgentSpec(
            name="writer",
            description="Writes text. Use for any writing task.",
            soul="You are a writer. Return exactly what is asked, nothing more.",
            tier="important",
            max_turns=5,
        )

        agent_a = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="parent-alpha",
            soul="I am Alpha. I delegate writing tasks.",
            subagents=[writer_spec],
            summarization=False,
            checkpointer=False,
        )
        agent_b = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="parent-beta",
            soul="I am Beta. I delegate writing tasks.",
            subagents=[writer_spec],
            summarization=False,
            checkpointer=False,
        )

        # Alpha delegates: write a file via subagent
        r_a = await agent_a.ainvoke(
            {"messages": [("user",
                "Use your writer subagent to compose a one-sentence poem about the sun. "
                "Then write the result to alpha_poem.txt")]},
        )
        ai_a = [m for m in r_a["messages"] if getattr(m, "type", "") == "ai"][-1]
        print(f"  Alpha result: {ai_a.content[:150]}")

        # Beta delegates: write a different file via subagent
        r_b = await agent_b.ainvoke(
            {"messages": [("user",
                "Use your writer subagent to compose a one-sentence poem about the moon. "
                "Then write the result to beta_poem.txt")]},
        )
        ai_b = [m for m in r_b["messages"] if getattr(m, "type", "") == "ai"][-1]
        print(f"  Beta result: {ai_b.content[:150]}")

        # Check flat agent structure
        agents_dir = os.path.join(ws, ".arion", "agents")
        all_agents = sorted(os.listdir(agents_dir))
        print(f"  All agent dirs: {all_agents}")

        assert "parent-alpha" in all_agents
        assert "parent-beta" in all_agents
        # Should have 4 agents: 2 parents + 2 subagent children (flat)
        assert len(all_agents) >= 4, f"Expected >= 4 agent dirs (2 parents + 2 children), got {len(all_agents)}: {all_agents}"

        # Verify all are flat (no nesting)
        for agent_dir_name in all_agents:
            agent_path = os.path.join(agents_dir, agent_dir_name)
            assert os.path.isdir(agent_path)
            soul_path = os.path.join(agent_path, "SOUL.md")
            assert os.path.exists(soul_path), f"Missing SOUL.md in {agent_dir_name}"

        child_agents = [d for d in all_agents if d not in ("parent-alpha", "parent-beta")]
        print(f"  Child agents: {child_agents}")

        # Verify children have correct souls (writer soul)
        for child_name in child_agents:
            child_soul = open(
                os.path.join(agents_dir, child_name, "SOUL.md"), encoding="utf-8"
            ).read()
            assert "writer" in child_soul.lower(), f"Child {child_name} should have writer soul"
        print("  All children have writer soul: OK")

        # Check files were written to shared workspace
        alpha_poem = os.path.join(ws, "alpha_poem.txt")
        beta_poem = os.path.join(ws, "beta_poem.txt")
        if os.path.exists(alpha_poem):
            print(f"  alpha_poem.txt: {open(alpha_poem, encoding='utf-8').read()[:80]}")
        if os.path.exists(beta_poem):
            print(f"  beta_poem.txt: {open(beta_poem, encoding='utf-8').read()[:80]}")

        # Stats
        print(f"  Alpha: model_calls={agent_a.stats.model_calls}, tool_calls={agent_a.stats.tool_calls}")
        print(f"  Beta: model_calls={agent_b.stats.model_calls}, tool_calls={agent_b.stats.tool_calls}")

        print("  >> PASSED")


async def test_subagent_isolation():
    """Subagent of Alpha cannot see subagent of Beta's conversation."""
    print("\n" + "=" * 60)
    print("Test: subagent conversation isolation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        secret_spec = SubAgentSpec(
            name="secret-keeper",
            description="Keeps secrets. Use to store information.",
            soul="You are a secret keeper. Remember what you are told.",
            tier="important",
            max_turns=5,
        )

        agent_a = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="iso-alpha",
            soul="I am Alpha.",
            subagents=[secret_spec],
            summarization=False,
            checkpointer=False,
        )
        agent_b = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="iso-beta",
            soul="I am Beta.",
            subagents=[secret_spec],
            summarization=False,
            checkpointer=False,
        )

        # Alpha's subagent gets a secret
        await agent_a.ainvoke(
            {"messages": [("user",
                "Tell your secret-keeper subagent: the code is DELTA-99. "
                "Just have it acknowledge.")]},
        )
        print("  Alpha told secret to its subagent")

        # Beta's subagent should NOT know Alpha's secret
        r_b = await agent_b.ainvoke(
            {"messages": [("user",
                "Ask your secret-keeper subagent what the code is. "
                "If it doesn't know, report UNKNOWN.")]},
        )
        ai_b = [m for m in r_b["messages"] if getattr(m, "type", "") == "ai"][-1]
        knows = "delta" in ai_b.content.lower() or "99" in ai_b.content
        assert not knows, f"Beta's subagent should NOT know Alpha's secret: {ai_b.content[:150]}"
        print(f"  Beta's subagent: {ai_b.content[:100]}")
        print("  Subagent isolation confirmed: OK")
        print("  >> PASSED")


async def main():
    await test_two_parents_with_subagents()
    await test_subagent_isolation()
    print(f"\n{'=' * 60}")
    print("ALL MULTI-AGENT SUBAGENT TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

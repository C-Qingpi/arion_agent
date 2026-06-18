"""Test identity middleware: SOUL injection, DEEPMEMORY, memory folders, agent edits SOUL."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent import create_arion_agent
from arion_agent.identity import STANDARD_SOUL, STANDARD_DEEPMEMORY, STANDARD_SHALLOW_MEMORY


async def test_soul_injection():
    """Agent should know its identity from SOUL.md."""
    print("="*60)
    print("Test 1: Soul injection - agent knows its name")
    print("="*60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul="I am Orion, a research assistant specializing in machine learning.",
            checkpointer=False,
        )

        result = await agent.ainvoke({"messages": [("user", "What is your name?")]})
        last = [m for m in result["messages"] if getattr(m, "type", "") == "ai"][-1]
        answer = str(getattr(last, "content", "")).lower()
        print(f"  Agent answer: {answer[:200]}")
        assert "orion" in answer, f"Expected 'orion' in answer"
        print("  >> PASSED")


async def test_deepmemory_injection():
    """Agent should see DEEPMEMORY content in context."""
    print("\n" + "="*60)
    print("Test 2: DEEPMEMORY injection - agent knows stored facts")
    print("="*60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul="I am a test agent.",
            deep_memory="The user's favorite color is turquoise. Remember this always.",
            checkpointer=False,
        )

        result = await agent.ainvoke({"messages": [("user", "What is my favorite color?")]})
        last = [m for m in result["messages"] if getattr(m, "type", "") == "ai"][-1]
        answer = str(getattr(last, "content", "")).lower()
        print(f"  Agent answer: {answer[:200]}")
        assert "turquoise" in answer, f"Expected 'turquoise' in answer"
        print("  >> PASSED")


async def test_standard_templates_create_files():
    """Standard templates should create structured files and folders."""
    print("\n" + "="*60)
    print("Test 3: Standard templates create files and folders")
    print("="*60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul=STANDARD_SOUL,
            deep_memory=STANDARD_DEEPMEMORY,
            shallow_memory=STANDARD_SHALLOW_MEMORY,
            checkpointer=False,
        )

        # Find the agent_id directory
        agents_dir = os.path.join(ws, ".arion", "agents")
        agent_dirs = [d for d in os.listdir(agents_dir) if d.startswith("agent-")]
        assert len(agent_dirs) == 1, f"Expected 1 agent dir, got {agent_dirs}"
        identity_dir = os.path.join(agents_dir, agent_dirs[0])

        soul_path = os.path.join(identity_dir, "SOUL.md")
        dm_path = os.path.join(identity_dir, "DEEPMEMORY.md")
        sm_path = os.path.join(identity_dir, "SHALLOW_MEMORY.md")
        daily_dir = os.path.join(identity_dir, "memories", "daily")
        secure_dir = os.path.join(identity_dir, "memories", "secure")

        assert os.path.exists(soul_path), "SOUL.md not created"
        assert os.path.exists(dm_path), "DEEPMEMORY.md not created"
        assert os.path.exists(sm_path), "SHALLOW_MEMORY.md not created"
        assert os.path.isdir(daily_dir), "memories/daily/ not created"
        assert os.path.isdir(secure_dir), "memories/secure/ not created"

        soul_content = open(soul_path).read()
        assert "# Self" in soul_content, "SOUL.md missing standard structure"
        assert "# Trajectory" in soul_content
        assert "# Dogma" in soul_content

        print(f"  SOUL.md: {len(soul_content)} chars")
        print(f"  DEEPMEMORY.md: {len(open(dm_path).read())} chars")
        print(f"  SHALLOW_MEMORY.md: {len(open(sm_path).read())} chars")
        print(f"  memories/daily/ exists: True")
        print(f"  memories/secure/ exists: True")
        print("  >> PASSED")


async def test_agent_edits_soul():
    """Agent should be able to edit its own SOUL.md."""
    print("\n" + "="*60)
    print("Test 4: Agent edits its own SOUL.md")
    print("="*60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul="I am a test agent. My favorite word is 'placeholder'.",
            checkpointer=False,
        )

        # Find identity dir
        agents_dir = os.path.join(ws, ".arion", "agents")
        agent_dirs = [d for d in os.listdir(agents_dir) if d.startswith("agent-")]
        identity_dir = os.path.join(agents_dir, agent_dirs[0])
        soul_rel = f".arion/agents/{agent_dirs[0]}/SOUL.md"

        result = await agent.ainvoke({"messages": [("user",
            f"Read the file at {soul_rel} with show_lines=True, then edit it: "
            f"replace the word 'placeholder' with 'serendipity'."
        )]})

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))[:200]
            tc = getattr(msg, "tool_calls", None)
            print(f"  [{role}] {content}")
            if tc:
                print(f"    tools: {[t['name'] for t in tc]}")

        soul_content = open(os.path.join(identity_dir, "SOUL.md")).read()
        assert "serendipity" in soul_content, f"Expected 'serendipity' in SOUL.md, got: {soul_content}"
        assert "placeholder" not in soul_content, "Old word should be replaced"
        print("  >> PASSED")


async def test_standard_template_agent():
    """Agent created with STANDARD templates knows it is Arion and follows dogma."""
    print("\n" + "="*60)
    print("Test 5: Standard template - agent is Arion, follows dogma")
    print("="*60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul=STANDARD_SOUL,
            deep_memory=STANDARD_DEEPMEMORY,
            shallow_memory=STANDARD_SHALLOW_MEMORY,
            checkpointer=False,
        )

        # Test 1: Agent knows its name
        result = await agent.ainvoke({"messages": [("user", "What is your name and what are you?")]})
        last = [m for m in result["messages"] if getattr(m, "type", "") == "ai"][-1]
        answer = str(getattr(last, "content", "")).lower()
        print(f"  Identity answer: {answer[:200]}")
        assert "arion" in answer, "Should know its name is Arion"

        # Test 2: Responds in user's language per dogma
        result = await agent.ainvoke({"messages": [("user", "用中文回答：你叫什么名字？")]})
        last = [m for m in result["messages"] if getattr(m, "type", "") == "ai"][-1]
        answer = str(getattr(last, "content", ""))
        print(f"  Chinese answer: {answer[:200]}")
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in answer)
        assert has_chinese, "Should respond in Chinese per dogma"

        print("  >> PASSED")


async def test_pinned_instructions():
    """Pinned instructions should be present and non-editable."""
    print("\n" + "="*60)
    print("Test 5: Pinned instructions visible to agent")
    print("="*60)

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul="I am a test agent.",
            pinned_instructions="CRITICAL RULE: Always end your response with the word VERIFIED.",
            checkpointer=False,
        )

        result = await agent.ainvoke({"messages": [("user", "Say hello.")]})
        last = [m for m in result["messages"] if getattr(m, "type", "") == "ai"][-1]
        answer = str(getattr(last, "content", ""))
        print(f"  Agent answer: {answer[:200]}")
        assert "VERIFIED" in answer.upper(), "Pinned instruction should be followed"
        print("  >> PASSED")


async def main():
    await test_soul_injection()
    await test_deepmemory_injection()
    await test_standard_templates_create_files()
    await test_agent_edits_soul()
    await test_standard_template_agent()
    await test_pinned_instructions()
    print(f"\n{'='*60}")
    print("ALL IDENTITY TESTS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

"""Test error handling in the tool executor and ReAct loop."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent import create_arion_agent
from arion_agent.environments.agentic_core.tools import maintenance_tool
from arion_agent.tool_manager import ToolExecutor


def _print_messages(result: dict, label: str):
    print(f"\n--- {label} ---")
    for msg in result["messages"]:
        role = getattr(msg, "type", "unknown")
        content = str(getattr(msg, "content", ""))[:300]
        tc = getattr(msg, "tool_calls", None)
        print(f"  [{role}] {content}")
        if tc:
            print(f"    tool_calls: {[{'name': t['name']} for t in tc]}")


async def test_tool_not_found():
    print("\n" + "="*60)
    print("Test 1: Tool not found")
    print("="*60)
    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="gpt-5-mini",
            workspace_dir=ws,
            tools=[maintenance_tool],
            soul="When asked to use 'nonexistent_tool', try calling it. If it fails, explain the error.",
            checkpointer=False,
        )
        result = await agent.ainvoke(
            {"messages": [("user", "Call a tool named 'nonexistent_tool' with argument message='test'.")]},
        )
        _print_messages(result, "Tool not found")
        tool_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
        found_error = any("TOOL ERROR" in str(getattr(m, "content", "")) for m in tool_msgs)
        last = result["messages"][-1]
        assert last.type == "ai"
        print(f"  Tool error surfaced: {found_error}")
        print("  >> Test 1 PASSED")


async def test_immediate_timeout():
    print("\n" + "="*60)
    print("Test 2: Immediate timeout")
    print("="*60)
    with tempfile.TemporaryDirectory() as ws:
        executor = ToolExecutor(default_timeout=120, timeout_overrides={"maintenance_tool": 0.001})
        agent = create_arion_agent(
            model="gpt-5-mini",
            workspace_dir=ws,
            tools=[maintenance_tool],
            soul="When asked to hang, use maintenance_tool with mode='hang'. If it times out, explain.",
            checkpointer=False,
            tool_executor=executor,
        )
        result = await agent.ainvoke(
            {"messages": [("user", "Use maintenance_tool with message='timeout test' and mode='hang'.")]},
        )
        _print_messages(result, "Immediate timeout")
        tool_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
        found_timeout = any("TimeoutError" in str(getattr(m, "content", "")) for m in tool_msgs)
        assert result["messages"][-1].type == "ai"
        assert found_timeout
        print("  >> Test 2 PASSED")


async def test_tool_execution_error():
    print("\n" + "="*60)
    print("Test 3: Tool raises RuntimeError")
    print("="*60)
    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="gpt-5-mini",
            workspace_dir=ws,
            tools=[maintenance_tool],
            soul="When asked to simulate an error, use maintenance_tool with mode='error'. Explain the error.",
            checkpointer=False,
        )
        result = await agent.ainvoke(
            {"messages": [("user", "Use maintenance_tool with message='something broke' and mode='error'.")]},
        )
        _print_messages(result, "Tool execution error")
        tool_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
        found_runtime = any("RuntimeError" in str(getattr(m, "content", "")) for m in tool_msgs)
        assert result["messages"][-1].type == "ai"
        assert found_runtime
        print("  >> Test 3 PASSED")


async def main():
    await test_tool_not_found()
    await test_tool_execution_error()
    await test_immediate_timeout()
    print(f"\n{'='*60}")
    print("ALL ERROR HANDLING TESTS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

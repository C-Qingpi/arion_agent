"""Test all 3 providers via CloseAI proxy: GPT, Claude, Gemini-3."""

from __future__ import annotations

import asyncio
import os
import sys

import tempfile

from arion_agent import create_arion_agent
from arion_agent.environments.agentic_core.tools import maintenance_tool

MODELS = [
    ("GPT", "gpt-5-mini"),
    ("Claude", "claude-sonnet-4-5"),
    ("Gemini3", "gemini-3-flash-preview"),
]

async def _test_provider(label: str, model_id: str):
    print(f"\n{'='*50}")
    print(f"Testing: {label} ({model_id})")
    print(f"{'='*50}")

    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model=model_id,
            workspace_dir=ws,
            tools=[maintenance_tool],
            soul="You are a test agent. When asked to echo, use maintenance_tool with the exact message given.",
            checkpointer=False,
        )

        result = await agent.ainvoke(
            {"messages": [("user", "Use maintenance_tool to echo 'hello from arion' with 0 delay")]},
        )

        for msg in result["messages"]:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            tc = getattr(msg, "tool_calls", None)
            preview = str(content)[:200] if content else ""
            print(f"  [{role}] {preview}")
            if tc:
                print(f"    tool_calls: {[t['name'] for t in tc]}")

        last = result["messages"][-1]
        assert "hello from arion" in str(getattr(last, "content", "")), f"{label}: Expected echo in final message"
        print(f"  >> {label} PASSED")


async def main():
    for label, model_id in MODELS:
        try:
            await _test_provider(label, model_id)
        except Exception as exc:
            print(f"  >> {label} FAILED: {exc}")


if __name__ == "__main__":
    if not os.environ.get("CLOSEAI_API_KEY"):
        print("Set CLOSEAI_API_KEY to run this test.")
        sys.exit(1)
    asyncio.run(main())

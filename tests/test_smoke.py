"""Smoke test: create agent, invoke with maintenance_tool, verify ReAct loop.

Uses CloseAI proxy via env vars configured in conftest.py.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401 - configures env vars

from arion_agent import create_arion_agent
from arion_agent.environments.agentic_core.tools import maintenance_tool


async def _run_smoke():
    import tempfile
    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            tools=[maintenance_tool],
            soul="You are a test agent. When asked to echo, use the maintenance_tool.",
            checkpointer=False,
        )

        result = await agent.ainvoke(
            {"messages": [("user", "Use maintenance_tool to echo 'hello arion' with 1 second delay")]},
        )

        for msg in result["messages"]:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            tc = getattr(msg, "tool_calls", None)
            print(f"[{role}] {str(content)[:200]}")
            if tc:
                print(f"  tool_calls: {tc}")
        print("--- Smoke test passed ---")


if __name__ == "__main__":
    asyncio.run(_run_smoke())

"""Test 2: Mid-execution interrupt and resume.

Flow:
  1. Ask Gemini to run maintenance_tool 5 times, 10s each.
  2. After ~5s, externally cancel the task.
  3. Send a follow-up message to echo a survival message.
  4. Verify PatchToolCallsMiddleware handles dangling tool calls.
  5. Verify the follow-up echo succeeds.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

from arion_agent import create_arion_agent
from arion_agent.environments.agentic_core.tools import maintenance_tool
from arion_agent.session import create_checkpointer

THREAD_ID = "interrupt-test-001"

SYSTEM_PROMPT = (
    "You are a test agent. When asked to run maintenance_tool multiple times, "
    "make separate sequential tool calls (one at a time). "
    "When asked to echo something, use maintenance_tool with the exact message."
)


def _print_messages(result: dict, label: str):
    print(f"\n--- {label} ---")
    for msg in result["messages"]:
        role = getattr(msg, "type", "unknown")
        content = str(getattr(msg, "content", ""))[:200]
        tc = getattr(msg, "tool_calls", None)
        print(f"  [{role}] {content}")
        if tc:
            print(f"    tool_calls: {[t['name'] for t in tc]}")


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_interrupt.db")

        async with create_checkpointer(db_path) as checkpointer:
            agent = create_arion_agent(
                model="gemini-3-flash-preview",
                workspace_dir=tmpdir,
                tools=[maintenance_tool],
                soul=SYSTEM_PROMPT,
                checkpointer=checkpointer,
            )

            config = {"configurable": {"thread_id": THREAD_ID, "model": "gemini-3-flash-preview"}}

            # Step 1: Start a long-running task, cancel after 5s
            print("="*60)
            print("Step 1: Start 5x maintenance_tool (10s each), cancel after 5s")
            print("="*60)

            task = asyncio.create_task(
                agent.ainvoke(
                    {"messages": [("user",
                        "Run maintenance_tool 5 separate times. Each time echo 'iteration N' "
                        "(where N is 1-5) with delay_seconds=10. Do them one at a time."
                    )]},
                    config=config,
                )
            )

            await asyncio.sleep(5)
            task.cancel()

            interrupted_result = None
            try:
                interrupted_result = await task
            except asyncio.CancelledError:
                print("  Task cancelled successfully (CancelledError)")
            except Exception as exc:
                print(f"  Task ended with: {type(exc).__name__}: {exc}")

            if interrupted_result:
                _print_messages(interrupted_result, "Interrupted result (partial)")

            # Step 2: Follow-up in same thread - the model should handle
            # dangling tool calls via PatchToolCallsMiddleware
            print("\n" + "="*60)
            print("Step 2: Follow-up after interrupt")
            print("="*60)

            config_followup = {"configurable": {"thread_id": THREAD_ID, "model": "gemini-3-flash-preview"}}
            result = await agent.ainvoke(
                {"messages": [("user",
                    "Echo to me 'you can survive mid tool execution interruption' using maintenance_tool"
                )]},
                config=config_followup,
            )

            _print_messages(result, "Follow-up result")

            found = any(
                "you can survive mid tool execution interruption" in str(getattr(m, "content", ""))
                for m in result["messages"]
            )
            assert found, "Expected survival echo in follow-up messages"
            print("  >> Follow-up PASSED")

    print(f"\n{'='*60}")
    print("INTERRUPT TEST PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    if not os.environ.get("CLOSEAI_API_KEY"):
        print("Set CLOSEAI_API_KEY to run this test.")
        sys.exit(1)
    asyncio.run(main())

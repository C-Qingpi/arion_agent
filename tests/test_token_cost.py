"""Measure token cost of a minimal agent invocation.

Creates an agent, sends "Reply with just OK", and prints the token usage
from the model response metadata. This shows how much context the tool
definitions consume before the agent even does anything.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent import create_arion_agent


async def main():
    with tempfile.TemporaryDirectory() as ws:
        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul="Reply with just OK.",
            checkpointer=False,
        )

        result = await agent.ainvoke({"messages": [("user", "Reply with just OK")]})

        print("="*60)
        print("Token Cost Analysis")
        print("="*60)

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))[:100]
            metadata = getattr(msg, "response_metadata", {})
            usage = getattr(msg, "usage_metadata", None) or metadata.get("token_usage", {})
            tc = getattr(msg, "tool_calls", None)

            print(f"\n[{role}] {content}")
            if tc:
                print(f"  tool_calls: {[t['name'] for t in tc]}")
            if usage:
                print(f"  usage: {usage}")

        # Extract final AI message usage
        ai_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "ai"]
        if ai_msgs:
            last_ai = ai_msgs[-1]
            usage = getattr(last_ai, "usage_metadata", None)
            metadata = getattr(last_ai, "response_metadata", {})

            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")

            if usage:
                input_tokens = getattr(usage, "input_tokens", None) or usage.get("input_tokens")
                output_tokens = getattr(usage, "output_tokens", None) or usage.get("output_tokens")
                total = getattr(usage, "total_tokens", None) or usage.get("total_tokens")
                print(f"Input tokens:  {input_tokens}")
                print(f"Output tokens: {output_tokens}")
                print(f"Total tokens:  {total}")
                if input_tokens:
                    print(f"\nThis is the baseline cost per invocation:")
                    print(f"  System prompt + tool definitions = ~{input_tokens - 10} input tokens")
                    print(f"  (subtract ~10 for the user message 'Reply with just OK')")
            else:
                print("No usage metadata found in response.")
                print(f"response_metadata keys: {list(metadata.keys())}")

            # Also show model info
            model_name = metadata.get("model_name") or metadata.get("model", "unknown")
            print(f"\nModel: {model_name}")


if __name__ == "__main__":
    asyncio.run(main())

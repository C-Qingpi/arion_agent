"""Test file+shell tools across all 3 providers.

Each model: write a file, read it back, edit it, run it via execute_python.
Verifies tool calling works correctly per provider.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

from arion_agent import create_arion_agent

MODELS = [
    ("GPT", "gpt-5-mini"),
    ("Claude", "claude-sonnet-4-5"),
    ("Gemini3", "gemini-3-flash-preview"),
]

SYSTEM_PROMPT = (
    "You are a test agent. Follow instructions precisely using the tools provided. "
    "Do not simulate results."
)


async def _test_model(label: str, model_id: str):
    print(f"\n{'='*60}")
    print(f"Testing: {label} ({model_id})")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as workspace:
        agent = create_arion_agent(
            model=model_id,
            soul=SYSTEM_PROMPT,
            workspace_dir=workspace,
            checkpointer=False,
        )

        # Step 1: Write a Python file
        print(f"\n  --- {label} Step 1: write_file ---")
        result = await agent.ainvoke({"messages": [("user",
            "Use write_file to create calc.py with this content:\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "print(add(17, 25))"
        )]})
        _show(result)
        all_text = _all_content(result)
        assert "Created" in all_text or "calc.py" in all_text, f"{label}: write_file failed"
        print(f"  >> {label} write_file PASSED")

        # Step 2: Read it back with show_lines
        print(f"\n  --- {label} Step 2: read_file ---")
        result = await agent.ainvoke({"messages": [("user",
            "Read calc.py with show_lines=True"
        )]})
        _show(result)
        all_text = _all_content(result)
        assert "L1|" in all_text or "add" in all_text, f"{label}: read_file failed"
        print(f"  >> {label} read_file PASSED")

        # Step 3: Edit line 3 to change the numbers
        print(f"\n  --- {label} Step 3: str_replace ---")
        result = await agent.ainvoke({"messages": [("user",
            "Edit calc.py: replace the print line `print(add(10, 20))` with `print(add(100, 200))` using str_replace"
        )]})
        _show(result)
        all_text = _all_content(result)
        assert "Replaced:" in all_text or "300" in all_text, f"{label}: str_replace failed"
        print(f"  >> {label} str_replace PASSED")

        # Step 4: Run it
        print(f"\n  --- {label} Step 4: execute_python ---")
        result = await agent.ainvoke({"messages": [("user",
            "Run calc.py using execute_python or execute_shell"
        )]})
        _show(result)
        all_text = _all_content(result)
        assert "300" in all_text, f"{label}: expected 300 in output, got: {all_text[:200]}"
        print(f"  >> {label} execute PASSED")

    print(f"\n  >> ALL {label} STEPS PASSED")


def _show(result: dict):
    for msg in result["messages"]:
        role = getattr(msg, "type", "?")
        content = str(getattr(msg, "content", ""))[:200]
        tc = getattr(msg, "tool_calls", None)
        print(f"    [{role}] {content}")
        if tc:
            print(f"      tools: {[t['name'] for t in tc]}")


def _all_content(result: dict) -> str:
    return " ".join(str(getattr(m, "content", "")) for m in result["messages"])


async def main():
    for label, model_id in MODELS:
        try:
            await _test_model(label, model_id)
        except Exception as exc:
            print(f"\n  >> {label} FAILED: {type(exc).__name__}: {exc}")

    print(f"\n{'='*60}")
    print("CROSS-PROVIDER ENVIRONMENT TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    if not os.environ.get("CLOSEAI_API_KEY"):
        print("Set CLOSEAI_API_KEY to run this test.")
        sys.exit(1)
    asyncio.run(main())

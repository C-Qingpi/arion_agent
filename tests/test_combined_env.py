"""Combined environment test: agent uses file + shell tools in a multi-step workflow.

Task: Agent creates a Python project, writes code, runs it, reads output, edits code.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

from arion_agent import create_arion_agent

SYSTEM_PROMPT = """You are a coding agent with workspace file and shell tools.
Follow instructions precisely. Use the tools provided - do not simulate results."""


async def main():
    with tempfile.TemporaryDirectory() as workspace:
        print(f"Workspace: {workspace}")

        agent = create_arion_agent(
            model="gpt-5-mini",
            soul=SYSTEM_PROMPT,
            workspace_dir=workspace,
            checkpointer=False,
        )

        # Step 1: Ask agent to create a project structure and write a script
        print("\n" + "="*60)
        print("Step 1: Create project and write a Python script")
        print("="*60)

        result = await agent.ainvoke({"messages": [("user",
            "Create a directory called 'myproject'. "
            "Then write a Python file at myproject/greet.py that defines a function greet(name) "
            "which returns 'Hello, {name}!' and a main block that prints greet('Arion')."
        )]})

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))[:200]
            tc = getattr(msg, "tool_calls", None)
            print(f"  [{role}] {content}")
            if tc:
                print(f"    tools: {[t['name'] for t in tc]}")

        # Step 2: Run the script
        print("\n" + "="*60)
        print("Step 2: Run the script")
        print("="*60)

        result = await agent.ainvoke({"messages": [("user",
            "Run myproject/greet.py using execute_python with file_path parameter."
        )]})

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))[:200]
            tc = getattr(msg, "tool_calls", None)
            print(f"  [{role}] {content}")
            if tc:
                print(f"    tools: {[t['name'] for t in tc]}")

        # Verify the output contains the expected greeting
        all_content = " ".join(str(getattr(m, "content", "")) for m in result["messages"])
        assert "Hello, Arion!" in all_content, "Expected 'Hello, Arion!' in execution output"
        print("  >> Step 2 PASSED: Script output verified")

        # Step 3: Read with line numbers, then edit
        print("\n" + "="*60)
        print("Step 3: Read with line numbers and edit the file")
        print("="*60)

        result = await agent.ainvoke({"messages": [("user",
            "Read myproject/greet.py with show_lines=True. "
            "Then edit it so the greet function returns 'Greetings, {name}! Welcome aboard.' "
            "instead of the original message. Use the line numbers from the read output."
        )]})

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))[:300]
            tc = getattr(msg, "tool_calls", None)
            print(f"  [{role}] {content}")
            if tc:
                print(f"    tools: {[t['name'] for t in tc]}")

        # Step 4: Run again to verify the edit
        print("\n" + "="*60)
        print("Step 4: Run edited script to verify")
        print("="*60)

        result = await agent.ainvoke({"messages": [("user",
            "Run myproject/greet.py again to verify the edit worked."
        )]})

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))[:200]
            tc = getattr(msg, "tool_calls", None)
            print(f"  [{role}] {content}")
            if tc:
                print(f"    tools: {[t['name'] for t in tc]}")

        all_content = " ".join(str(getattr(m, "content", "")) for m in result["messages"])
        assert "Greetings" in all_content and "Welcome aboard" in all_content, \
            "Expected edited greeting in output"
        print("  >> Step 4 PASSED: Edited script output verified")

        # Step 5: List files and clean up
        print("\n" + "="*60)
        print("Step 5: List workspace and delete the file")
        print("="*60)

        result = await agent.ainvoke({"messages": [("user",
            "List all files in the workspace recursively. Then delete myproject/greet.py."
        )]})

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))[:300]
            tc = getattr(msg, "tool_calls", None)
            print(f"  [{role}] {content}")
            if tc:
                print(f"    tools: {[t['name'] for t in tc]}")

        all_content = " ".join(str(getattr(m, "content", "")) for m in result["messages"])
        assert "recycle" in all_content.lower(), "Expected recycle bin confirmation"
        print("  >> Step 5 PASSED: File deleted to recycle bin")

    print(f"\n{'='*60}")
    print("ALL COMBINED ENVIRONMENT TESTS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    if not os.environ.get("CLOSEAI_API_KEY"):
        print("Set CLOSEAI_API_KEY to run this test.")
        sys.exit(1)
    asyncio.run(main())

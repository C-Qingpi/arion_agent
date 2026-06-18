"""Test browser environment: session, tools, and real agent integration."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent.environments.browser.config import BrowserConfig
from arion_agent.environments.browser.session import BrowserSession


async def test_browser_session_navigate():
    """Browser session can navigate and take snapshot."""
    print("=" * 60)
    print("Test: browser session navigate + snapshot")
    print("=" * 60)

    config = BrowserConfig(headless=True, stealth=False, humanize=False)
    session = BrowserSession(config)

    try:
        result = await session.navigate("https://example.com")
        print(f"  Navigate: {result}")
        assert "Example Domain" in result or "example" in result.lower()

        snap = await session.snapshot()
        print(f"  Snapshot: {snap[:200]}...")
        assert len(snap) > 10

        text = await session.read("h1")
        print(f"  Read h1: {text}")
        assert "Example" in text
    finally:
        await session.close()

    print("  >> PASSED")


async def test_browser_action():
    """browser_action tool works for navigate and read."""
    print("\n" + "=" * 60)
    print("Test: browser_action unified tool")
    print("=" * 60)

    config = BrowserConfig(headless=True, stealth=False, humanize=False)
    session = BrowserSession(config)

    try:
        result = await session.action("navigate", "https://httpbin.org/html")
        print(f"  Navigate: {result}")
        assert "Navigated" in result

        snap = await session.snapshot()
        assert len(snap) > 10
        print(f"  Snapshot length: {len(snap)} chars")

        scroll = await session.action("scroll", value="down")
        assert "Scrolled" in scroll
        print(f"  Scroll: {scroll}")
    finally:
        await session.close()

    print("  >> PASSED")


async def test_http_request():
    """http_request tool works for API endpoints."""
    print("\n" + "=" * 60)
    print("Test: http_request tool")
    print("=" * 60)

    from arion_agent.environments.browser.session import BrowserSession
    from arion_agent.environments.browser.tools import create_browser_tools

    config = BrowserConfig(headless=True)
    session = BrowserSession(config)
    tools = create_browser_tools(session)
    http_tool = [t for t in tools if t.name == "http_request"][0]

    result = await http_tool.ainvoke({
        "method": "GET",
        "url": "https://httpbin.org/json",
    })
    print(f"  Response: {result[:200]}")
    assert "HTTP 200" in result
    assert "slideshow" in result.lower()

    await session.close()
    print("  >> PASSED")


async def test_browser_console_and_eval():
    """Console logs and JS evaluation work."""
    print("\n" + "=" * 60)
    print("Test: browser console + eval_js")
    print("=" * 60)

    config = BrowserConfig(headless=True, stealth=False, humanize=False)
    session = BrowserSession(config)

    try:
        await session.navigate("https://example.com")

        js_result = await session.evaluate_js("document.title")
        print(f"  JS eval (title): {js_result}")
        assert "Example" in js_result

        console_output = await session.console()
        print(f"  Console: {console_output[:100]}")
    finally:
        await session.close()

    print("  >> PASSED")


async def test_browser_skills_seeded():
    """BrowserEnvironment seeds workspace-level skills on init."""
    print("\n" + "=" * 60)
    print("Test: browser skills seeded")
    print("=" * 60)

    from arion_agent.environments.browser import BrowserEnvironment, BrowserConfig

    with tempfile.TemporaryDirectory() as ws:
        browser_mw = BrowserEnvironment(
            BrowserConfig(headless=True),
            workspace_dir=ws,
        )

        skills_dir = os.path.join(ws, ".arion", "skills")
        expected = ["web-navigation", "form-filling", "web-scraping", "visual-debugging"]
        for skill_name in expected:
            skill_md = os.path.join(skills_dir, skill_name, "SKILL.md")
            assert os.path.exists(skill_md), f"Missing skill: {skill_name}"
            content = open(skill_md, encoding="utf-8").read()
            assert "---" in content, f"Skill {skill_name} missing frontmatter"

        print(f"  Skills seeded: {expected}")

        # Verify seed-if-absent (second init shouldn't overwrite)
        custom = os.path.join(skills_dir, "web-navigation", "SKILL.md")
        with open(custom, "w", encoding="utf-8") as f:
            f.write("# Custom override\n")

        BrowserEnvironment(BrowserConfig(headless=True), workspace_dir=ws)
        preserved = open(custom, encoding="utf-8").read()
        assert "Custom override" in preserved, "Should not overwrite existing skills"
        print("  Seed-if-absent: existing skills preserved")

    print("  >> PASSED")


async def test_agent_with_browser():
    """Full agent with browser environment visits a page."""
    print("\n" + "=" * 60)
    print("Integration: agent with browser visits example.com")
    print("=" * 60)

    from arion_agent import create_arion_agent
    from arion_agent.environments.browser import BrowserEnvironment, BrowserConfig

    with tempfile.TemporaryDirectory() as ws:
        browser_mw = BrowserEnvironment(BrowserConfig(headless=True, stealth=False, humanize=False))

        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            soul="You are a web browsing agent. Be concise.",
            middleware=[browser_mw],
            summarization=False,
            checkpointer=False,
        )

        r = await agent.ainvoke(
            {"messages": [("user",
                "Use browser_action to navigate to https://example.com, "
                "then use browser_snapshot to see the page, and tell me "
                "what the main heading says.")]},
        )

        ai = [m for m in r["messages"] if getattr(m, "type", "") == "ai"][-1]
        print(f"  Agent: {ai.content[:200]}")
        assert "example" in ai.content.lower() or "Example" in ai.content

    print("  >> PASSED")


async def main():
    await test_browser_session_navigate()
    await test_browser_action()
    await test_http_request()
    await test_browser_console_and_eval()
    await test_browser_skills_seeded()

    print(f"\n{'=' * 60}")
    print("UNIT TESTS PASSED -- proceeding to integration")
    print(f"{'=' * 60}")

    await test_agent_with_browser()

    print(f"\n{'=' * 60}")
    print("ALL BROWSER TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

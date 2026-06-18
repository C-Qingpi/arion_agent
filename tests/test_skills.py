"""Test skills system: parsing, scanning, middleware injection, real LLM."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent.skills import SkillMetadata, SkillMiddleware, parse_skill_md, scan_skills_directory


SAMPLE_SKILL_MD = """\
---
name: web-research
description: Structured approach to conducting thorough web research. Use when asked to research a topic.
license: MIT
---

# Web Research Skill

## When to Use
- User asks to research a topic

## Workflow
1. Clarify the research question
2. Search for primary sources
3. Synthesize into report
"""

SAMPLE_SKILL_MD_2 = """\
---
name: code-review
description: Systematic code review for bugs, style, and security.
---

# Code Review

## Steps
1. Read the code
2. Check for bugs
3. Suggest improvements
"""


def test_parse_skill_md():
    """Parse SKILL.md YAML frontmatter."""
    print("=" * 60)
    print("Test: parse SKILL.md")
    print("=" * 60)

    meta = parse_skill_md(SAMPLE_SKILL_MD, "/skills/web-research/SKILL.md", "web-research")
    assert meta is not None
    assert meta.name == "web-research"
    assert "thorough web research" in meta.description
    assert meta.license == "MIT"
    print(f"  name={meta.name}, desc={meta.description[:50]}...")
    print("  >> PASSED")


def test_parse_invalid():
    """Invalid SKILL.md should return None."""
    print("\n" + "=" * 60)
    print("Test: parse invalid SKILL.md")
    print("=" * 60)

    assert parse_skill_md("no frontmatter", "/x", "x") is None
    assert parse_skill_md("---\nfoo: bar\n---\n", "/x", "x") is None
    print("  Invalid files correctly rejected")
    print("  >> PASSED")


def test_parse_xml_format():
    """Parse XML-style frontmatter."""
    print("\n" + "=" * 60)
    print("Test: parse XML frontmatter")
    print("=" * 60)

    xml_skill = """\
<name>api-client</name>
<description>Generate API client code from OpenAPI specs.</description>

# API Client Skill

Steps:
1. Read the OpenAPI spec
2. Generate client code
"""
    meta = parse_skill_md(xml_skill, "/skills/api-client/SKILL.md", "api-client")
    assert meta is not None
    assert meta.name == "api-client"
    assert "API client" in meta.description
    print(f"  name={meta.name}, desc={meta.description[:50]}...")

    xml_attr = """\
<skill name="data-viz">
<description>Create data visualizations from CSV files.</description>
</skill>

# Data Viz
"""
    meta2 = parse_skill_md(xml_attr, "/skills/data-viz/SKILL.md", "data-viz")
    assert meta2 is not None
    assert meta2.name == "data-viz"
    print(f"  name={meta2.name} (attribute style)")
    print("  >> PASSED")


def test_scan_directory():
    """Scan a directory for skill folders."""
    print("\n" + "=" * 60)
    print("Test: scan skills directory")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        skills_dir = Path(ws) / "skills"
        (skills_dir / "web-research").mkdir(parents=True)
        (skills_dir / "web-research" / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
        (skills_dir / "code-review").mkdir(parents=True)
        (skills_dir / "code-review" / "SKILL.md").write_text(SAMPLE_SKILL_MD_2, encoding="utf-8")
        (skills_dir / "empty-dir").mkdir()

        results = scan_skills_directory(skills_dir)
        assert len(results) == 2, f"Expected 2 skills, got {len(results)}"
        names = {r.name for r in results}
        assert "web-research" in names
        assert "code-review" in names
        print(f"  Found: {names}")
        print("  >> PASSED")


def test_middleware_injection():
    """SkillMiddleware should inject skill info into system prompt based on directory."""
    print("\n" + "=" * 60)
    print("Test: middleware system prompt injection")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        imp_dir = identity_dir / "skills" / "important" / "web-research"
        gen_dir = identity_dir / "skills" / "generic" / "code-review"
        imp_dir.mkdir(parents=True)
        gen_dir.mkdir(parents=True)
        (imp_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
        (gen_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD_2, encoding="utf-8")

        # No important_skills param needed: directory structure is source of truth
        mw = SkillMiddleware()
        mw.set_identity_dir(identity_dir)
        mw.set_workspace_dir(Path(ws))

        parts = mw.wrap_system_message([])

        assert len(parts) >= 1, f"Expected at least 1 section, got {len(parts)}"
        content = "\n".join(parts)
        assert "skill_guidance" in content
        assert "important_skills" in content
        assert "web-research" in content
        assert "generic_skills" in content
        assert "catalog.md" in content or "catalog" in content
        assert "skill_management" in content

        catalog = identity_dir / "skills" / "catalog.md"
        assert catalog.exists(), "catalog.md should be generated"
        catalog_content = catalog.read_text(encoding="utf-8")
        assert "code-review" in catalog_content

        assert ws.replace("\\", "/") not in content, "Paths should be relative, not absolute"

        print("  Directory-based classification: important/ -> system prompt, generic/ -> catalog")
        print("  catalog.md generated with generic skills")
        print("  skill_management section present for agent self-management")
        print("  Paths are workspace-relative")
        print("  >> PASSED")


def test_dynamic_reclassification():
    """Agent can reclassify skills by moving folders between directories."""
    print("\n" + "=" * 60)
    print("Test: dynamic skill reclassification")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        imp_dir = identity_dir / "skills" / "important" / "web-research"
        gen_dir = identity_dir / "skills" / "generic" / "code-review"
        imp_dir.mkdir(parents=True)
        gen_dir.mkdir(parents=True)
        (imp_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
        (gen_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD_2, encoding="utf-8")

        mw = SkillMiddleware()
        mw.set_identity_dir(identity_dir)
        mw.set_workspace_dir(Path(ws))

        # Turn 1: web-research is important, code-review is generic
        parts1 = mw.wrap_system_message([])
        content1 = "\n".join(parts1)
        assert "important_skills" in content1
        assert "web-research" in content1

        # Simulate agent moving web-research from important to generic
        target = identity_dir / "skills" / "generic" / "web-research"
        imp_dir.rename(target)

        # Turn 2: web-research should now be generic (fresh scan, no caching)
        parts2 = mw.wrap_system_message([])
        content2 = "\n".join(parts2)
        assert "important_skills" not in content2, "No important skills should remain"

        catalog = identity_dir / "skills" / "catalog.md"
        catalog_content = catalog.read_text(encoding="utf-8")
        assert "web-research" in catalog_content, "web-research should be in catalog now"
        assert "code-review" in catalog_content

        print("  Turn 1: web-research in important_skills section")
        print("  Simulated agent moving folder to generic/")
        print("  Turn 2: web-research now in catalog, no important_skills section")
        print("  >> PASSED")


def test_seeding_from_sources():
    """SkillMiddleware should seed skills from sources into identity_dir."""
    print("\n" + "=" * 60)
    print("Test: seeding from skill sources")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        # Create a source directory with skills
        source_dir = Path(ws) / "skill_sources"
        (source_dir / "web-research").mkdir(parents=True)
        (source_dir / "web-research" / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
        (source_dir / "code-review").mkdir(parents=True)
        (source_dir / "code-review" / "SKILL.md").write_text(SAMPLE_SKILL_MD_2, encoding="utf-8")

        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"

        mw = SkillMiddleware(
            important_skills=["web-research"],
            skill_sources=[str(source_dir)],
        )
        mw.set_identity_dir(identity_dir)
        mw.set_workspace_dir(Path(ws))

        mw.before_agent({})

        # web-research should be seeded into important/
        imp_skill = identity_dir / "skills" / "important" / "web-research" / "SKILL.md"
        assert imp_skill.exists(), "web-research should be seeded into important/"

        # code-review should be seeded into generic/ (from explicit source)
        gen_skill = identity_dir / "skills" / "generic" / "code-review" / "SKILL.md"
        assert gen_skill.exists(), "code-review should be seeded into generic/"

        # Verify seed-if-absent: modify the file, re-seed should not overwrite
        imp_skill.write_text("# Agent modified this", encoding="utf-8")
        mw2 = SkillMiddleware(
            important_skills=["web-research"],
            skill_sources=[str(source_dir)],
        )
        mw2.set_identity_dir(identity_dir)
        mw2.set_workspace_dir(Path(ws))
        mw2.before_agent({})

        preserved = imp_skill.read_text(encoding="utf-8")
        assert "Agent modified" in preserved, "Seed-if-absent should preserve agent edits"

        print("  web-research seeded into important/")
        print("  code-review seeded into generic/")
        print("  Seed-if-absent preserves agent modifications")
        print("  >> PASSED")


async def test_real_skill_activation():
    """Agent with skills should be able to reference and use them."""
    print("\n" + "=" * 60)
    print("Integration: agent with skills")
    print("=" * 60)

    from arion_agent import create_arion_agent

    with tempfile.TemporaryDirectory() as ws:
        # Pre-place skill in the agent's important directory.
        # In practice, SkillMiddleware seeds this from skill_sources.
        imp_dir = Path(ws) / ".arion" / "agents" / "skill-test" / "skills" / "important" / "greeting-protocol"
        imp_dir.mkdir(parents=True)
        (imp_dir / "SKILL.md").write_text("""\
---
name: greeting-protocol
description: A protocol for greeting users. Always greet with the phrase GREETINGS-PROTOCOL-ACTIVE.
---

# Greeting Protocol

When asked to greet someone, always include the exact phrase: GREETINGS-PROTOCOL-ACTIVE
""", encoding="utf-8")

        agent = create_arion_agent(
            model="openai:gpt-5-mini",
            workspace_dir=ws,
            agent_id="skill-test",
            soul="You are a test agent. Follow your skills precisely.",
            skills=SkillMiddleware(),
            summarization=False,
            checkpointer=False,
        )

        r = await agent.ainvoke(
            {"messages": [("user", "Read your greeting-protocol skill and then greet me following it.")]},
        )
        ai = [m for m in r["messages"] if getattr(m, "type", "") == "ai"][-1]
        print(f"  Agent: {ai.content[:200]}")
        assert "GREETINGS-PROTOCOL-ACTIVE" in ai.content, "Agent should follow the skill"
        print("  >> PASSED")


async def main():
    test_parse_skill_md()
    test_parse_invalid()
    test_parse_xml_format()
    test_scan_directory()
    test_middleware_injection()
    test_dynamic_reclassification()
    test_seeding_from_sources()

    print(f"\n{'=' * 60}")
    print("MOCK TESTS PASSED -- proceeding to integration")
    print(f"{'=' * 60}")

    await test_real_skill_activation()

    print(f"\n{'=' * 60}")
    print("ALL SKILLS TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

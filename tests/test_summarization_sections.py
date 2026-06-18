"""Tests for summarization optional sections."""

from __future__ import annotations

import tempfile
from pathlib import Path

from arion_agent.summarization.sections import (
    build_supplemental_sections,
    format_configured_skills,
)


def test_format_configured_skills_includes_skill_and_soul():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        identity = root / "identity"
        skill_dir = identity / "skills" / "important" / "demo-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: Demo skill for tests\n---\n",
            encoding="utf-8",
        )
        (identity / "SOUL.md").write_text("# Demo agent", encoding="utf-8")

        text = format_configured_skills(identity_dir=identity, workspace_dir=root)

        assert "demo-skill" in text
        assert "SOUL.md" in text


def test_build_supplemental_sections():
    text = build_supplemental_sections()
    assert "EVIDENCE AND SOURCE TRACING" in text
    assert "DISCOVERIES" in text

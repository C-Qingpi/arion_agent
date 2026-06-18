"""Tests for compaction / inheritable-context prompt templates."""

from __future__ import annotations

from arion_agent.summarization.prompts import (
    PERPETUAL_SUMMARY_PROMPT,
    PERPETUAL_WRAPPER,
    TASK_SUMMARY_PROMPT,
    TASK_WRAPPER,
)


def _required_sections(prompt: str) -> list[str]:
    return [
        "## SESSION CONTEXT",
        "## RECENT USER REQUESTS (VERBATIM)",
        "## ACTIVE WORK & STATUS",
        "## PROGRESS & DECISIONS",
        "## FULL HISTORY TRAJECTORIES",
        "## SKILLS & GUIDELINES",
        "## ARTIFACTS",
        "### Project record & index files",
        "### Other artifacts",
        "## OPEN ITEMS",
        "## NEXT STEPS",
    ]


def test_task_prompt_is_inheritable_context_not_summary():
    assert "Inheritable Context Author" in TASK_SUMMARY_PROMPT
    assert "If you do not say it here" in TASK_SUMMARY_PROMPT
    assert "not writing a loose summary" in TASK_SUMMARY_PROMPT
    for section in _required_sections(TASK_SUMMARY_PROMPT):
        assert section in TASK_SUMMARY_PROMPT


def test_task_prompt_preserves_verbatim_user_requests_rules():
    assert "latest three substantive user messages" in TASK_SUMMARY_PROMPT
    assert "continue" in TASK_SUMMARY_PROMPT
    assert "verbatim" in TASK_SUMMARY_PROMPT.lower()


def test_task_prompt_artifact_categories():
    assert "README.md, PROJECT.md" in TASK_SUMMARY_PROMPT
    assert "Project record & index files" in TASK_SUMMARY_PROMPT
    assert "Other artifacts" in TASK_SUMMARY_PROMPT


def test_task_prompt_skills_and_history_sections():
    assert "{configured_skills}" in TASK_SUMMARY_PROMPT
    assert "Carry history forward" in TASK_SUMMARY_PROMPT
    assert "Configured skills at compaction time" in TASK_SUMMARY_PROMPT


def test_perpetual_prompt_matches_task_structure():
    assert "Inheritable Context Author" in PERPETUAL_SUMMARY_PROMPT
    for section in _required_sections(PERPETUAL_SUMMARY_PROMPT):
        assert section in PERPETUAL_SUMMARY_PROMPT


def test_wrappers_describe_inheritable_context():
    assert "inheritable context" in TASK_WRAPPER.lower()
    assert "inheritable context" in PERPETUAL_WRAPPER.lower()
    assert "{file_path}" in TASK_WRAPPER
    assert "{summary}" in TASK_WRAPPER


def test_prompt_placeholders():
    for prompt in (TASK_SUMMARY_PROMPT, PERPETUAL_SUMMARY_PROMPT):
        assert "{messages}" in prompt
        assert "{budget}" in prompt
        assert "{configured_skills}" in prompt
        assert "{optional_sections}" in prompt

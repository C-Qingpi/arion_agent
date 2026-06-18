"""Build summarization prompt sections from agent configuration."""

from __future__ import annotations

from pathlib import Path

from arion_agent.summarization.config import OPTIONAL_SECTIONS
from arion_agent.skills.config import scan_skills_directory


def format_configured_skills(
    identity_dir: Path | None = None,
    workspace_dir: Path | None = None,
) -> str:
    lines: list[str] = []
    if identity_dir is not None:
        important_dir = identity_dir / "skills" / "important"
        if important_dir.is_dir():
            for meta in scan_skills_directory(important_dir):
                lines.append(f"- {meta.name}: {meta.description}")

    if workspace_dir is not None:
        ws_skills = workspace_dir / ".arion" / "skills"
        if ws_skills.is_dir():
            seen = {line.split(":", 1)[0].removeprefix("- ").strip() for line in lines}
            for meta in scan_skills_directory(ws_skills):
                if meta.name not in seen:
                    lines.append(f"- {meta.name} (workspace): {meta.description}")

    if identity_dir is not None:
        soul = identity_dir / "SOUL.md"
        if soul.is_file():
            lines.append(f"- Agent identity: follow SOUL.md at {soul}")

    if not lines:
        return "None configured at compaction time."
    return "\n".join(lines)


def build_supplemental_sections() -> str:
    """Extra sections appended after the core inheritable-context structure."""
    return "".join(OPTIONAL_SECTIONS.values())


def build_optional_sections(
    identity_dir: Path | None = None,
    workspace_dir: Path | None = None,
) -> str:
    """Backward-compatible helper returning supplemental sections only."""
    return build_supplemental_sections()

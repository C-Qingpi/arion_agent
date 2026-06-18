"""Skill middleware: discovers, catalogs, and injects skills into the system prompt.

Company practice: define agent classes by skill configuration.
A "web researcher" agent has different important skills than a "project
commander" agent. Core environments (file, shell) stay the same; skills
are the specialization layer. Add or remove environment middleware to
further customize (e.g. browser env for web researcher, no shell for a
read-only analyst).

Skills follow the Agent Skills open standard:
https://agentskills.io/specification

Skills are dynamic and per-agent, following the same pattern as identity
(SOUL.md). Directory structure is the runtime source of truth:
  identity_dir/skills/important/ -> injected into system prompt every turn
  identity_dir/skills/generic/   -> listed in catalog, read on demand

The agent can reclassify skills at runtime by moving folders between
important/ and generic/. create_arion_agent only seeds the initial layout;
the agent evolves it from there.

Environments can also provide skills (e.g. browser environment seeds
web-navigation). These are discovered during seeding and placed into
the agent's identity directory based on the initial classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arion_agent.middleware.base import ArionMiddleware
from arion_agent.skills.config import SkillMetadata, scan_skills_directory
from arion_agent.skills.prompts import DEFAULT_SKILL_INSTRUCTIONS, SKILL_MANAGEMENT_INSTRUCTIONS
from arion_agent.util.persistence import seed_file, ensure_directory, workspace_relative_path, file_exists, read_file_text, write_file as persistence_write_file

logger = logging.getLogger(__name__)


class SkillMiddleware(ArionMiddleware):
    """Discovers, catalogs, and injects skills into the system prompt.

    Two tiers (determined by directory structure, not constructor params):
      important (identity_dir/skills/important/): in system prompt every turn
      generic (identity_dir/skills/generic/): catalog pointer, read on demand

    Dynamic: the agent can reclassify skills by moving folders between
    important/ and generic/, create new skills, or edit existing ones.
    Changes are picked up on the next turn (fresh scan, no caching).
    """

    def __init__(
        self,
        *,
        important_skills: list[str] | None = None,
        skill_sources: list[str | Path] | None = None,
        instructions: str = DEFAULT_SKILL_INSTRUCTIONS,
    ) -> None:
        # Seed-time only: determines initial directory placement
        self._seed_important_names = set(important_skills or [])
        self._seed_sources = [Path(s) for s in (skill_sources or [])]
        self._instructions = instructions

        self._identity_dir: Path | None = None
        self._workspace_dir: Path | None = None
        self._seeded = False

    def set_identity_dir(self, identity_dir: Path) -> None:
        """Set by assembly.py during agent construction."""
        self._identity_dir = identity_dir

    def set_workspace_dir(self, workspace_dir: Path) -> None:
        """Set by assembly.py during agent construction."""
        self._workspace_dir = workspace_dir

    # -- Seeding (once, at first run) ------------------------------------

    def _seed_skills(self) -> None:
        """Seed skills from sources into identity_dir on first run.

        Discovers skills from skill_sources and workspace-level, then
        copies them into identity_dir/skills/important/ or generic/
        based on the initial important_skills classification.

        Uses seed-if-absent contract: existing SKILL.md files are never
        overwritten, preserving agent modifications across restarts.
        """
        if self._seeded or self._identity_dir is None:
            return
        self._seeded = True

        important_dir = self._identity_dir / "skills" / "important"
        generic_dir = self._identity_dir / "skills" / "generic"
        ensure_directory(important_dir)
        ensure_directory(generic_dir)

        source_skills: dict[str, SkillMetadata] = {}
        source_origin: dict[str, str] = {}

        for source in self._seed_sources:
            for meta in scan_skills_directory(source):
                source_skills[meta.name] = meta
                source_origin[meta.name] = "explicit"

        if self._workspace_dir is not None:
            ws_skills = self._workspace_dir / ".arion" / "skills"
            for meta in scan_skills_directory(ws_skills):
                if meta.name not in source_skills:
                    source_skills[meta.name] = meta
                source_origin.setdefault(meta.name, "workspace")

        for meta in source_skills.values():
            if meta.name in self._seed_important_names:
                target_dir = important_dir / meta.name
            elif source_origin.get(meta.name) == "explicit":
                target_dir = generic_dir / meta.name
            else:
                # Workspace-level skills not in important_skills stay as
                # shared fallback; don't copy into per-agent directory.
                continue

            ensure_directory(target_dir)
            source_path = Path(meta.path)
            if file_exists(source_path):
                try:
                    content = read_file_text(source_path)
                    seed_file(target_dir / "SKILL.md", content)
                except Exception:
                    logger.warning("Cannot seed skill from %s", source_path)

    # -- Runtime scanning (every turn, fresh from disk) ------------------

    def _scan_skills(self) -> tuple[list[SkillMetadata], list[SkillMetadata]]:
        """Scan skill directories and return (important, generic).

        Directory structure is the source of truth for classification.
        Fresh scan every call (no caching), like identity reads SOUL.md.
        """
        important: list[SkillMetadata] = []
        generic: list[SkillMetadata] = []

        if self._identity_dir is not None:
            for meta in scan_skills_directory(self._identity_dir / "skills" / "important"):
                important.append(meta)
            for meta in scan_skills_directory(self._identity_dir / "skills" / "generic"):
                generic.append(meta)

        # Workspace-level skills as shared fallback (always generic)
        seen = {m.name for m in important} | {m.name for m in generic}
        if self._workspace_dir is not None:
            ws_skills = self._workspace_dir / ".arion" / "skills"
            for meta in scan_skills_directory(ws_skills):
                if meta.name not in seen:
                    generic.append(meta)

        return important, generic

    def _write_catalog(self, generic: list[SkillMetadata]) -> None:
        """Regenerate catalog.md from current generic skills.

        The catalog is a derived artifact (auto-generated index), not
        agent-authored content. It is always regenerated to reflect the
        current skill landscape.
        """
        if self._identity_dir is None:
            return

        catalog_path = self._identity_dir / "skills" / "catalog.md"
        lines = [
            "# Skill Catalog",
            "",
            "## How to use",
            "When no important skill matches your task, consult this catalog.",
            "Read the SKILL.md of the matching skill for full instructions.",
            "",
            "## Index",
        ]
        for s in generic:
            lines.append(f"- {s.name}: {s.description[:80]}")
        lines.append("")
        lines.append("---")
        lines.append("")
        for s in generic:
            rel = self._rel_path(s.path)
            lines.append(f"## {s.name}")
            lines.append(s.description)
            lines.append(f"Path: {rel}")
            lines.append("")

        persistence_write_file(catalog_path, "\n".join(lines))

    # -- Helpers ---------------------------------------------------------

    def _rel_path(self, path: str) -> str:
        if self._workspace_dir is not None:
            return workspace_relative_path(Path(path), self._workspace_dir)
        return path

    # -- Middleware hooks -------------------------------------------------

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Seed skill directories on first run."""
        self._seed_skills()
        return None

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        """Contribute skill guidance sections to the system message.

        Scans fresh from disk each turn so agent-driven changes
        (moving folders, creating skills) take effect immediately.
        """
        self._seed_skills()
        important, generic = self._scan_skills()

        if not important and not generic:
            return parts

        parts.append(f"<skill_guidance>\n{self._instructions}\n</skill_guidance>")

        if important:
            lines = ["<important_skills>"]
            for s in important:
                rel = self._rel_path(s.path)
                lines.append(f'<skill name="{s.name}">')
                lines.append(s.description)
                lines.append(f"Path: {rel}")
                lines.append("</skill>")
            lines.append("</important_skills>")
            parts.append("\n".join(lines))

        if generic:
            self._write_catalog(generic)
            if self._identity_dir is not None:
                catalog_rel = self._rel_path(
                    str(self._identity_dir / "skills" / "catalog.md")
                )
                parts.append(
                    "<generic_skills>\n"
                    f"Additional skills available in catalog: {catalog_rel}\n"
                    "Read it when no important skill matches your task.\n"
                    "</generic_skills>"
                )

        if self._identity_dir is not None:
            parts.append(
                f"<skill_management>\n{SKILL_MANAGEMENT_INSTRUCTIONS}\n</skill_management>"
            )

        return parts

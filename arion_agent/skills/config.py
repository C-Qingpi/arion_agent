"""Skill metadata parsing from SKILL.md files.

Follows the Agent Skills open standard: https://agentskills.io/specification
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


@dataclass(frozen=True)
class SkillMetadata:
    """Parsed metadata from a SKILL.md frontmatter."""

    name: str
    description: str
    path: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def _parse_yaml_frontmatter(content: str) -> dict[str, str] | None:
    """Extract data from YAML frontmatter (--- delimited)."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return None
    raw = match.group(1)
    try:
        import yaml
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    kv: dict[str, str] = {}
    for line in raw.strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            kv[key.strip()] = val.strip()
    return kv or None


def _parse_xml_frontmatter(content: str) -> dict[str, str] | None:
    """Extract name and description from XML-style frontmatter.

    Supports formats like:
      <skill name="web-research">
      <description>...</description>
      </skill>

    Or individual tags:
      <name>web-research</name>
      <description>...</description>
    """
    data: dict[str, str] = {}
    attr_match = re.search(r'<skill\s+name="([^"]+)"', content)
    if attr_match:
        data["name"] = attr_match.group(1)
    for tag in ("name", "description", "license", "compatibility"):
        tag_match = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
        if tag_match:
            data[tag] = tag_match.group(1).strip()
    return data if data else None


def parse_skill_md(content: str, skill_path: str, directory_name: str) -> SkillMetadata | None:
    """Parse frontmatter from SKILL.md content.

    Supports two formats:
    1. YAML frontmatter (Agent Skills standard): --- delimited block
    2. XML frontmatter: <name>, <description> tags

    Only name and description are required. All other fields are optional.

    Args:
        content: Raw SKILL.md file content.
        skill_path: Path to the SKILL.md (for error messages and metadata).
        directory_name: Parent directory name (should match skill name per spec).

    Returns:
        SkillMetadata if name and description are present, None otherwise.
    """
    data = _parse_yaml_frontmatter(content)
    if data is None:
        data = _parse_xml_frontmatter(content)
    if data is None:
        logger.warning("No frontmatter (YAML or XML) in %s", skill_path)
        return None

    name = str(data.get("name", "")).strip()
    description = str(data.get("description", "")).strip()

    if not name or not description:
        logger.warning("Missing required name or description in %s", skill_path)
        return None

    if len(description) > MAX_SKILL_DESCRIPTION_LENGTH:
        description = description[:MAX_SKILL_DESCRIPTION_LENGTH]

    raw_meta = data.get("metadata", {})
    meta = {str(k): str(v) for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}

    return SkillMetadata(
        name=name,
        description=description,
        path=skill_path,
        license=str(data.get("license", "")).strip() or None,
        compatibility=str(data.get("compatibility", "")).strip() or None,
        metadata=meta,
    )


def scan_skills_directory(skills_dir: Path) -> list[SkillMetadata]:
    """Scan a directory for skill folders containing SKILL.md.

    Args:
        skills_dir: Directory to scan (e.g. identity_dir/skills/important/).

    Returns:
        List of parsed SkillMetadata from valid SKILL.md files.
    """
    from arion_agent.util.persistence import is_directory, list_dir_entries, file_exists, read_file_text

    if not is_directory(skills_dir):
        return []

    results: list[SkillMetadata] = []
    for name, is_dir in list_dir_entries(skills_dir):
        if not is_dir:
            continue
        child = skills_dir / name
        skill_md = child / "SKILL.md"
        if not file_exists(skill_md):
            continue
        try:
            content = read_file_text(skill_md)
        except Exception:
            logger.warning("Cannot read %s", skill_md)
            continue

        meta = parse_skill_md(content, str(skill_md), name)
        if meta is not None:
            results.append(meta)

    return results

"""Skills system: progressive disclosure of specialized knowledge and workflows."""

from arion_agent.skills.config import SkillMetadata, parse_skill_md, scan_skills_directory
from arion_agent.skills.middleware import SkillMiddleware
from arion_agent.skills.prompts import DEFAULT_SKILL_INSTRUCTIONS, SKILL_MANAGEMENT_INSTRUCTIONS

__all__ = [
    "DEFAULT_SKILL_INSTRUCTIONS",
    "SKILL_MANAGEMENT_INSTRUCTIONS",
    "SkillMetadata",
    "SkillMiddleware",
    "parse_skill_md",
    "scan_skills_directory",
]

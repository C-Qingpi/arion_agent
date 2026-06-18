"""System prompt sections for skill-enabled agents."""

DEFAULT_SKILL_INSTRUCTIONS = (
    "You have access to skills that provide specialized knowledge and "
    "workflows. Skills follow progressive disclosure: you see names and "
    "descriptions here; read the full SKILL.md for detailed instructions "
    "when a skill matches your task. Skills may include step-by-step "
    "instructions, scripts to execute, API call patterns, or templates. "
    "After reading a skill, follow its instructions step by step."
)

SKILL_MANAGEMENT_INSTRUCTIONS = (
    "Your skills are stored in your identity directory under skills/. "
    "You can manage them at runtime; changes take effect next turn.\n"
    "\n"
    "Directory structure determines classification:\n"
    "  skills/important/{name}/SKILL.md - visible in your system prompt every turn\n"
    "  skills/generic/{name}/SKILL.md   - listed in catalog, read on demand\n"
    "\n"
    "Actions you can take:\n"
    "- Promote a skill: move its folder from generic/ to important/\n"
    "- Demote a skill: move its folder from important/ to generic/\n"
    "- Create a new skill: create a folder with a SKILL.md containing "
    "YAML frontmatter (name, description) and instructions\n"
    "- Edit a skill: modify its SKILL.md content directly"
)

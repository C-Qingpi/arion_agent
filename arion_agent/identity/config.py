"""Identity config types.

Each config bundles an initial template (seeded to file if not exists)
and system prompt instructions (injected every turn to guide the agent
on maintaining that file).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SoulConfig:
    """SOUL.md configuration: agent identity.

    initial_template: written to SOUL.md if file doesn't exist yet.
    instructions: injected into system prompt every turn (how to maintain SOUL).
    """
    initial_template: str
    instructions: str = ""


@dataclass(frozen=True)
class MemoryConfig:
    """DEEPMEMORY.md configuration: curated long-term memory.

    initial_template: written to DEEPMEMORY.md if file doesn't exist yet.
    instructions: injected into system prompt every turn.
    """
    initial_template: str
    instructions: str = ""


@dataclass(frozen=True)
class ShallowMemoryConfig:
    """SHALLOW_MEMORY.md configuration: storage guidance for memories/ folder.

    guidance: written to SHALLOW_MEMORY.md (folder structure rules, not injected as content).
    initial_folders: created on first run (e.g. ["memories/daily", "memories/secure"]).
    instructions: short pointer injected into system prompt every turn
                  (tells agent where to find storage guidance, not the guidance itself).
    """
    guidance: str
    initial_folders: list[str] = field(default_factory=list)
    instructions: str = ""

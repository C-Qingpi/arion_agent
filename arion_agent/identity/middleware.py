"""Identity middleware: manages SOUL, DEEPMEMORY, SHALLOW_MEMORY files and injects into context.

Depends on the file environment being available (workspace must exist).
Files are seeded on first run if they don't exist. Content is injected
into the system prompt every turn via wrap_system_message. The agent
can edit these files using normal file tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.identity.config import MemoryConfig, ShallowMemoryConfig, SoulConfig
from arion_agent.middleware.base import ArionMiddleware
from arion_agent.util.persistence import ensure_directory, seed_file

DEFAULT_SOUL_CONTENT = "You are a helpful assistant."
DEFAULT_DEEPMEMORY_CONTENT = "No memories recorded yet."

MAX_SOUL_CHARS = 10_000
MAX_DEEPMEMORY_CHARS = 5_000

BASE_ARION_PROMPT = """\
<role>
You are an ArionAgent. Use tools to accomplish tasks. Verify before assuming.
Do not guess file contents; read them. Do not simulate tool results; call them.
</role>"""


class IdentityMiddleware(ArionMiddleware):
    """Manages agent identity files and injects them into the system prompt.

    On first run, seeds SOUL.md, DEEPMEMORY.md, SHALLOW_MEMORY.md and
    initial folders if they don't exist. On every turn, reads the files
    and contributes their content (plus config instructions) as system
    message sections via wrap_system_message.
    """

    def __init__(
        self,
        identity_dir: Path,
        *,
        agent_id: str = "unknown",
        workspace_dir: Path | None = None,
        soul: SoulConfig | str | None = None,
        deep_memory: MemoryConfig | str | None = None,
        shallow_memory: ShallowMemoryConfig | None = None,
        pinned_instructions: str | None = None,
    ) -> None:
        self._identity_dir = identity_dir
        self._agent_id = agent_id
        self._workspace_dir = workspace_dir or identity_dir.parent.parent.parent
        self._pinned = pinned_instructions or ""

        if isinstance(soul, SoulConfig):
            self._soul_config = soul
        elif isinstance(soul, str):
            self._soul_config = SoulConfig(initial_template=soul)
        else:
            self._soul_config = SoulConfig(initial_template=DEFAULT_SOUL_CONTENT)

        if isinstance(deep_memory, MemoryConfig):
            self._memory_config = deep_memory
        elif isinstance(deep_memory, str):
            self._memory_config = MemoryConfig(initial_template=deep_memory)
        else:
            self._memory_config = MemoryConfig(initial_template=DEFAULT_DEEPMEMORY_CONTENT)

        self._shallow_config = shallow_memory

        self._seed_files()

    def _seed_files(self) -> None:
        """Create identity files and folders if they don't exist.

        Uses seed_file (seed-if-absent contract): existing files are
        never overwritten. This preserves agent-evolved identity across
        restarts when the same agent_id is reused.
        """
        ensure_directory(self._identity_dir)
        seed_file(self._identity_dir / "SOUL.md", self._soul_config.initial_template)
        seed_file(self._identity_dir / "DEEPMEMORY.md", self._memory_config.initial_template)
        if self._shallow_config:
            seed_file(self._identity_dir / "SHALLOW_MEMORY.md", self._shallow_config.guidance)
            for folder in self._shallow_config.initial_folders:
                ensure_directory(self._identity_dir / folder)

    def _read_file(self, name: str, max_chars: int) -> str:
        """Read an identity file, capped at max_chars."""
        from arion_agent.util.persistence import file_exists, read_file_text
        path = self._identity_dir / name
        if not file_exists(path):
            return ""
        try:
            content = read_file_text(path, max_chars=max_chars)
            return content
        except Exception:
            return ""

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        """Contribute identity sections to the system message."""
        from arion_agent.util.persistence import workspace_relative_path

        parts.append(BASE_ARION_PROMPT)

        rel_identity = workspace_relative_path(self._identity_dir, self._workspace_dir)
        parts.append(
            f"<agent_identity>\n"
            f"Agent ID: {self._agent_id}\n"
            f"Identity directory: {rel_identity}/\n"
            f"Files: SOUL.md, DEEPMEMORY.md, SHALLOW_MEMORY.md\n"
            f"</agent_identity>"
        )

        if self._pinned:
            parts.append(f"<pinned_instructions>\n{self._pinned}\n</pinned_instructions>")

        if self._soul_config.instructions:
            parts.append(f"<soul_guidance>\n{self._soul_config.instructions}\n</soul_guidance>")

        soul_content = self._read_file("SOUL.md", MAX_SOUL_CHARS)
        if soul_content:
            parts.append(f"<soul>\n{soul_content}\n</soul>")

        if self._memory_config.instructions:
            parts.append(f"<memory_guidance>\n{self._memory_config.instructions}\n</memory_guidance>")

        dm_content = self._read_file("DEEPMEMORY.md", MAX_DEEPMEMORY_CHARS)
        if dm_content and dm_content != DEFAULT_DEEPMEMORY_CONTENT:
            parts.append(f"<deep_memory>\n{dm_content}\n</deep_memory>")

        if self._shallow_config and self._shallow_config.instructions:
            parts.append(f"<shallow_memory>\n{self._shallow_config.instructions}\n</shallow_memory>")

        return parts

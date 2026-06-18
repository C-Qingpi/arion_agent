"""File environment middleware: contributes file tools and system prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.environments._sandbox.config import MOUNT_PREFIX, SandboxConfig
from arion_agent.environments.file.tools import create_file_tools
from arion_agent.middleware.base import ArionMiddleware

FILE_SYSTEM_PROMPT = """## File Environment

You have access to a workspace directory with file tools:
- read_file: Read files (text, images, PDF/docx/xlsx via extraction). Text reads include a Revision token. Use show_lines=True before editing.
- write_file: Create new files. Use mode='append' or 'prepend' for existing files.
- edit_file: Replace line ranges in text files. Always pass the latest Revision token from read_file as expected_revision. replacement_content must contain the full content for the selected range; any omitted line in that range is deleted. Successful edits do not return a fresh Revision token, so reread before another edit. If you get StaleRead, reread first because line numbers may have shifted.
- delete_file: Soft delete to .recycle_bin/ (recoverable).
- move_file: Move or rename files.
- undo_file_operation: Undo the latest undoable file operation.
- list_files: List directory contents. Use path="." to see the workspace root. Use depth=2 or depth=3 to explore subdirectories (recommended); depth=1 lists immediate children only.
- set_directory: Create, rename, delete, or move directories.

All paths are relative to the workspace root (shown in PathConfinement errors as the absolute workspace root). Use path="." for the workspace root. You cannot access files outside the workspace.
When starting a task, list the workspace root first (list_files with path=".") to understand what exists."""

MOUNT_SYSTEM_PROMPT_TEMPLATE = """
Connected directories are available under {prefix}/:
{mount_list}
These are real external locations. Changes you make here are reflected on the actual filesystem. Readonly mounts cannot be modified."""


class FileEnvironment(ArionMiddleware):
    """Middleware providing workspace-confined file operations."""

    def __init__(self, sandbox_config: SandboxConfig) -> None:
        self._config = sandbox_config
        mount_map = sandbox_config.mount_map if sandbox_config.mounts else None
        self._tools = create_file_tools(sandbox_config.workspace_dir, mounts=mount_map)

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        parts.append(FILE_SYSTEM_PROMPT)
        if self._config.mounts:
            mount_lines = []
            for m in self._config.mounts:
                label = f"- {MOUNT_PREFIX}/{m.name}/"
                if m.readonly:
                    label += " (readonly)"
                mount_lines.append(label)
            parts.append(MOUNT_SYSTEM_PROMPT_TEMPLATE.format(
                prefix=MOUNT_PREFIX,
                mount_list="\n".join(mount_lines),
            ))
        return parts

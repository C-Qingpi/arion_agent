"""File environment tools as LangChain @tool functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from langchain_core.tools import tool

from arion_agent.environments.file import ops

if TYPE_CHECKING:
    from arion_agent.environments._sandbox.config import MountSpec


def create_file_tools(workspace: Path, mounts: dict[str, MountSpec] | None = None) -> list:
    """Create all file environment tools bound to a workspace."""

    @tool
    def read_file(
        path: Annotated[str, "File path relative to workspace root."],
        start_line: Annotated[int, "First line to read (1-indexed). Default 1."] = 1,
        end_line: Annotated[int | None, "Last line to read (inclusive). Default: start_line + 200."] = None,
        show_lines: Annotated[bool, "Add line markers (L1|, L2|, ...) for navigation. Default False."] = False,
    ) -> str:
        """Read a file from the workspace. Returns raw content by default.
Text file reads include a Revision token; pass it to str_replace as expected_revision.
Handles text, images (multimodal), and binary formats (PDF/docx/xlsx via MarkItDown).
For files larger than 10 MB, use execute_python or execute_shell instead."""
        return ops.read_file(
            path, workspace,
            start_line=start_line, end_line=end_line, show_lines=show_lines,
            mounts=mounts,
        )

    @tool
    def write_file(
        path: Annotated[str, "File path relative to workspace root."],
        content: Annotated[str, "Content to write."],
        mode: Annotated[str, "Write mode: 'create' (default, fails if exists), 'overwrite' (replaces existing), 'append', or 'prepend'."] = "create",
    ) -> str:
        """Create a new file or overwrite/append/prepend to an existing one.
Creates parent directories automatically. Default mode refuses if file exists;
use 'overwrite' for full replacement or str_replace for surgical changes.
Successful overwrite operations return an Undo token for one-step rollback.
Write one file per turn. For multiple files, split across separate turns."""
        return ops.write_file(path, content, workspace, mode=mode, mounts=mounts)

    @tool
    def str_replace(
        path: Annotated[str, "File path relative to workspace root."],
        old_string: Annotated[str, "Exact literal text to find. Must match read_file output exactly, including whitespace."],
        new_string: Annotated[str, "Replacement text."],
        expected_revision: Annotated[str, "Revision token returned by read_file."],
        occurrence: Annotated[str, "Which matches to replace (1-based): default '1' = first only; '3' = third; '1-22' or '3-99' = range; '3-*' = third through last; '*' or 'all' = every match."] = "1",
    ) -> str:
        """Replace exact text in a text file.
Always read_file first and copy old_string literally from that output.
Use a longer old_string with surrounding context when the target text appears multiple times.
Default occurrence='1' replaces only the first match. Use '*' to replace all matches.
Returns a compact preview, fresh Revision token, and Undo token.
If str_replace returns StaleRead or NotFound, call read_file again before retrying."""
        return ops.str_replace(
            path,
            old_string,
            new_string,
            expected_revision,
            workspace,
            occurrence=occurrence,
            mounts=mounts,
        )

    @tool
    def delete_file(
        path: Annotated[str, "File path to delete."],
    ) -> str:
        """Delete a file by moving it to .recycle_bin/ for recovery.
Deleting a file already in .recycle_bin/ removes it permanently.
Successful deletes return an Undo token for one-step rollback.
Only works on files, not directories (use set_directory for that)."""
        return ops.delete_file(path, workspace, mounts=mounts)

    @tool
    def move_file(
        source_path: Annotated[str, "Current file path."],
        destination_path: Annotated[str, "New file path or directory to move into."],
    ) -> str:
        """Move or rename a file within the workspace.
Creates destination parent directories automatically.
Successful moves return an Undo token for one-step rollback.
Refuses if destination already exists. Files only, not directories."""
        return ops.move_file(source_path, destination_path, workspace, mounts=mounts)

    @tool
    def undo_file_operation(
        undo_token: Annotated[str, "Undo token returned by the latest undoable file operation."],
    ) -> str:
        """Undo the latest undoable file operation.
Only the most recent undoable file operation can be undone, and the token is consumed after a successful undo."""
        return ops.undo_file_operation(undo_token, workspace, mounts=mounts)

    @tool
    def list_files(
        path: Annotated[str, 'Directory path relative to workspace root. Use "." or omit for root directory.'] = "",
        depth: Annotated[int, "How many directory levels to include. 1 = immediate children only (default). Recommended: 2-3."] = 1,
        ignore: Annotated[str, "Comma-separated gitignore-style patterns to exclude (e.g. '*.pyc, __pycache__/, node_modules/'). Optional."] = "",
    ) -> str:
        """List files and directories at the given path.
Returns a flat list with [dir] and [file] markers.
To see the workspace root, call with path="." or omit the path argument.
Use depth=2 or depth=3 to explore subdirectories without listing the entire tree.
Use ignore to filter out files/directories matching gitignore-style patterns."""
        effective_path = path or "."
        return ops.list_files(effective_path, workspace, depth=depth, ignore=ignore, mounts=mounts)

    @tool
    def set_directory(
        action: Annotated[str, "One of: 'create', 'rename', 'delete', 'move'."],
        path: Annotated[str, "Directory path."],
        new_name: Annotated[str, "New name for rename action."] = "",
        destination: Annotated[str, "Destination path for move action."] = "",
    ) -> str:
        """Directory operations. create: make directory. rename: change name in place.
delete: move to .recycle_bin/. move: relocate directory. One action per call."""
        return ops.set_directory(action, path, workspace, new_name=new_name, destination=destination, mounts=mounts)

    return [read_file, write_file, str_replace, delete_file, move_file, undo_file_operation, list_files, set_directory]

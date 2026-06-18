"""Path confinement: resolve and validate all paths stay within workspace.

Mount-aware: paths under imported_directories/{name}/ are intercepted
before .resolve() follows symlinks, and confined against the mount
source rather than the workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arion_agent.environments._sandbox.config import MountSpec


class PathConfinementError(Exception):
    """Raised when a path escapes the workspace or a mount boundary."""


def workspace_path_context(
    workspace: Path,
    *,
    mounts: dict[str, MountSpec] | None = None,
) -> str:
    """Human-readable explanation of how file-tool paths are rooted."""
    from arion_agent.environments._sandbox.config import MOUNT_PREFIX

    root = workspace.resolve()
    lines = [
        f"Workspace root (absolute): {root}",
        f'All file-tool paths are relative to this root. "." means the workspace root ({root}).',
        f'Example: list_files with path="." lists {root}',
        "Do not use host absolute paths (e.g. /Users/... or C:\\...) unless they are expressed as workspace-relative paths.",
    ]
    if mounts:
        mount_names = ", ".join(f"{MOUNT_PREFIX}/{name}/" for name in mounts)
        lines.append(f"Mounted directories (also workspace-relative): {mount_names}")
    return "\n".join(lines)


def format_path_confinement_tool_error(
    tool_name: str,
    user_path: str,
    workspace: Path,
    exc: PathConfinementError,
    *,
    mounts: dict[str, MountSpec] | None = None,
) -> str:
    """Format a path confinement failure for tool output."""
    return (
        f"TOOL ERROR ({tool_name})\n"
        f"Type: PathConfinement\n"
        f"Message: {exc}\n"
        f"Path you provided: {user_path!r}\n"
        f"{workspace_path_context(workspace, mounts=mounts)}"
    )


def resolve_path(
    user_path: str,
    workspace: Path,
    *,
    mounts: dict[str, MountSpec] | None = None,
) -> Path:
    """Resolve a user-provided path to an absolute path within the workspace.

    Accepts:
      - Relative paths: "src/main.py" -> workspace/src/main.py
      - Absolute paths starting with /: "/src/main.py" -> workspace/src/main.py
      - Mount paths: "imported_directories/Desktop/file.txt" -> mount_source/file.txt

    Rejects (via validate_confinement):
      - ".." traversal escaping workspace or mount boundary
      - "~" expansion
    """
    from arion_agent.environments._sandbox.config import MOUNT_PREFIX

    clean = user_path.strip()
    if not clean:
        raise PathConfinementError(
            "Empty path. Use \".\" for the workspace root or a relative path such as \"src/main.py\"."
        )

    if clean.startswith("~"):
        raise PathConfinementError(
            f"Home directory expansion not allowed: {clean}. "
            f"Paths must be relative to the workspace root ({workspace.resolve()}), not the host home directory."
        )

    if clean.startswith("/") or clean.startswith("\\"):
        clean = clean.lstrip("/\\")

    if mounts and clean.startswith(MOUNT_PREFIX + "/"):
        rest = clean[len(MOUNT_PREFIX) + 1:]
        for mount_name, spec in mounts.items():
            if rest == mount_name or rest.startswith(mount_name + "/"):
                sub = rest[len(mount_name):].lstrip("/\\")
                resolved = (spec.source / sub).resolve() if sub else spec.source.resolve()
                _validate_mount_confinement(resolved, spec.source)
                return resolved

    resolved = (workspace / clean).resolve()
    validate_confinement(resolved, workspace)
    return resolved


def is_readonly_path(
    resolved: Path,
    mounts: dict[str, MountSpec] | None,
) -> bool:
    """Check if a resolved path falls within a readonly mount."""
    if not mounts:
        return False
    resolved_abs = resolved.resolve()
    for spec in mounts.values():
        source_abs = spec.source.resolve()
        try:
            resolved_abs.relative_to(source_abs)
        except ValueError:
            continue
        return spec.readonly
    return False


def validate_confinement(resolved: Path, workspace: Path) -> None:
    """Ensure resolved path is inside the workspace after symlink resolution."""
    workspace_resolved = workspace.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise PathConfinementError(
            f"Path escapes workspace: {resolved} is outside workspace root {workspace_resolved}. "
            f"Use paths relative to the workspace root; \".\" refers to {workspace_resolved}."
        ) from None


def _validate_mount_confinement(resolved: Path, mount_source: Path) -> None:
    """Ensure resolved path is inside the mount source directory."""
    source_resolved = mount_source.resolve()
    try:
        resolved.relative_to(source_resolved)
    except ValueError:
        raise PathConfinementError(
            f"Path escapes mount boundary: {resolved} is outside {source_resolved}"
        ) from None

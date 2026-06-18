"""Shell confinement via bubblewrap and mount link creation.

Bubblewrap (bwrap) provides Linux namespace-based isolation so shell
processes can only see the workspace and declared mounts. On other
platforms, confinement is not enforced at the kernel level.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arion_agent.environments._sandbox.config import SandboxConfig

logger = logging.getLogger(__name__)

_READONLY_BIND_PATHS = ["/usr", "/lib", "/lib64", "/bin", "/sbin"]


def create_mount_link(link_path: Path, target: Path) -> str:
    """Create an OS-level directory link (symlink or junction).

    Unix: symlink. Windows: directory junction (no admin required).
    Returns "symlink" or "junction".
    """
    if sys.platform == "win32":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise OSError(f"Failed to create junction {link_path} -> {target}: {result.stderr.strip()}")
        return "junction"
    else:
        link_path.symlink_to(target)
        return "symlink"


def build_bwrap_command(
    command: list[str],
    sandbox_cfg: SandboxConfig,
) -> list[str]:
    """Wrap a command with bubblewrap for namespace isolation.

    The wrapped process sees:
    - Standard system paths as read-only (/usr, /lib, /bin, etc.)
    - /proc, /dev, /tmp
    - The workspace directory as read-write at /workspace
    - Each mount from SandboxConfig bound into /workspace/imported_directories/
    - Optionally no network (--unshare-net)
    """
    from arion_agent.environments._sandbox.config import MOUNT_PREFIX

    args = ["bwrap"]

    for bind_path in _READONLY_BIND_PATHS:
        p = Path(bind_path)
        if p.exists():
            args += ["--ro-bind", bind_path, bind_path]

    # /etc is needed for DNS resolution, timezone, etc.
    if Path("/etc").exists():
        args += ["--ro-bind", "/etc", "/etc"]

    args += ["--proc", "/proc"]
    args += ["--dev", "/dev"]
    args += ["--tmpfs", "/tmp"]

    workspace = str(sandbox_cfg.workspace_dir)
    args += ["--bind", workspace, workspace]

    for mount in sandbox_cfg.mounts:
        mount_dest = str(sandbox_cfg.workspace_dir / MOUNT_PREFIX / mount.name)
        bind_flag = "--ro-bind" if mount.readonly else "--bind"
        args += [bind_flag, str(mount.source), mount_dest]

    if not sandbox_cfg.network_allowed:
        args += ["--unshare-net"]

    args += ["--chdir", workspace]
    args += ["--die-with-parent"]
    args += ["--"]
    args += command

    return args


def build_bwrap_shell(
    shell_path: str,
    sandbox_cfg: SandboxConfig,
) -> list[str]:
    """Build a bwrap-wrapped interactive shell command for terminals."""
    shell_cmd = [shell_path]
    if "bash" in shell_path:
        shell_cmd.append("-i")
    return build_bwrap_command(shell_cmd, sandbox_cfg)

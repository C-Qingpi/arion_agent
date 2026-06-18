"""Sandbox configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MOUNT_PREFIX = "imported_directories"

_SENSITIVE_ROOTS_UNIX = frozenset({"/", "/etc", "/var", "/usr", "/bin", "/sbin", "/root", "/boot", "/sys", "/proc"})
_SENSITIVE_ROOTS_WIN = frozenset({"C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)"})


@dataclass
class MountSpec:
    """A local directory bridged into the workspace namespace.

    Appears at workspace/imported_directories/{name}/ as a symlink
    (Unix) or directory junction (Windows).
    """

    name: str
    source: Path
    readonly: bool = False

    def __post_init__(self) -> None:
        self.source = Path(self.source).resolve()
        if not self.source.is_dir():
            raise ValueError(f"Mount source does not exist or is not a directory: {self.source}")
        if "/" in self.name or "\\" in self.name:
            raise ValueError(f"Mount name must be a simple name, not a path: {self.name}")
        self._validate_not_sensitive()

    def _validate_not_sensitive(self) -> None:
        source_str = str(self.source)
        import sys
        if sys.platform == "win32":
            for root in _SENSITIVE_ROOTS_WIN:
                if source_str.upper() == root.upper():
                    raise ValueError(f"Refusing to mount sensitive system directory: {self.source}")
        else:
            if source_str in _SENSITIVE_ROOTS_UNIX:
                raise ValueError(f"Refusing to mount sensitive system directory: {self.source}")


@dataclass
class SandboxConfig:
    """Configuration for the workspace sandbox."""

    workspace_dir: Path
    mounts: list[MountSpec] = field(default_factory=list)
    confinement: str = "auto"
    max_readable_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    default_shell_timeout: float = 120.0
    max_output_bytes: int = 200_000
    network_allowed: bool = False

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        recycle = self.workspace_dir / ".recycle_bin"
        recycle.mkdir(exist_ok=True)

        if self.confinement == "auto":
            self.confinement = _detect_confinement()

        if self.mounts:
            self._setup_mounts()

    def _setup_mounts(self) -> None:
        from arion_agent.environments._sandbox.confinement import create_mount_link

        mount_root = self.workspace_dir / MOUNT_PREFIX
        mount_root.mkdir(exist_ok=True)

        seen_names: set[str] = set()
        for mount in self.mounts:
            if mount.name in seen_names:
                raise ValueError(f"Duplicate mount name: {mount.name}")
            seen_names.add(mount.name)

            link_path = mount_root / mount.name
            if link_path.exists() or link_path.is_symlink():
                resolved_target = link_path.resolve()
                if resolved_target == mount.source:
                    continue
                logger.warning(
                    "Mount link %s points to %s, expected %s. Recreating.",
                    link_path, resolved_target, mount.source,
                )
                link_path.unlink()

            create_mount_link(link_path, mount.source)
            logger.info("Mounted %s -> %s", link_path, mount.source)

    @property
    def mount_map(self) -> dict[str, MountSpec]:
        return {m.name: m for m in self.mounts}


def _detect_confinement() -> str:
    import shutil
    import sys
    if sys.platform.startswith("linux"):
        from arion_agent.util.runtime import is_container
        if is_container():
            logger.info(
                "Running inside a container -- skipping bwrap "
                "(container provides isolation)."
            )
            return "none"
        if shutil.which("bwrap"):
            return "bwrap"
        logger.warning(
            "Shell sandboxing unavailable: bubblewrap (bwrap) not found. "
            "Install with: apt install bubblewrap"
        )
    elif sys.platform == "darwin":
        logger.warning("Shell sandboxing not available on macOS. Shell commands are unconfined.")
    elif sys.platform == "win32":
        logger.warning("Shell sandboxing not available on Windows. Shell commands are unconfined.")
    return "none"

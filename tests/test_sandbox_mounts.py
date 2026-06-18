"""Tests for shell sandboxing (confinement) and directory mounts.

No LLM needed. Tests cover:
- MountSpec validation
- SandboxConfig mount setup (symlinks/junctions)
- Mount-aware resolve_path
- Readonly enforcement at the ops layer
- is_readonly_path helper
- Confinement detection
- bwrap command building
- File ops through mounts (read, write, edit, delete, move, list)
- FileEnvironment system prompt with mounts
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arion_agent.environments._sandbox.config import (
    MOUNT_PREFIX,
    MountSpec,
    SandboxConfig,
    _detect_confinement,
)
from arion_agent.environments._sandbox.paths import (
    PathConfinementError,
    is_readonly_path,
    resolve_path,
)
from arion_agent.environments.file import ops
from arion_agent.serve import create_service_router
from arion_agent.util.remote_io import RemoteIOBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ws_with_mounts(
    *mount_defs: tuple[str, bool],
) -> tuple[tempfile.TemporaryDirectory, Path, list[MountSpec], dict[str, MountSpec]]:
    """Create a workspace with external directories and mounts.

    mount_defs: sequence of (name, readonly) tuples.
    Returns (td, workspace, mounts_list, mount_map).
    Each external directory gets a file 'seed.txt' for testing reads.
    """
    td = tempfile.TemporaryDirectory()
    base = Path(td.name)
    ws = base / "workspace"
    ws.mkdir()
    (ws / ".recycle_bin").mkdir()

    mounts = []
    for name, readonly in mount_defs:
        ext_dir = base / f"external_{name}"
        ext_dir.mkdir()
        (ext_dir / "seed.txt").write_text(f"content from {name}")
        mounts.append(MountSpec(name=name, source=ext_dir, readonly=readonly))

    cfg = SandboxConfig(workspace_dir=ws, mounts=mounts, confinement="none")
    mount_map = cfg.mount_map
    return td, cfg.workspace_dir, mounts, mount_map


def _simple_ws() -> tuple[tempfile.TemporaryDirectory, Path]:
    td = tempfile.TemporaryDirectory()
    ws = Path(td.name)
    (ws / ".recycle_bin").mkdir()
    return td, ws


def _extract_revision(read_result: str) -> str:
    for line in read_result.splitlines():
        if line.startswith("Revision: "):
            return line.split(": ", 1)[1]
    raise AssertionError(f"Revision token missing from read_file output:\n{read_result}")


class _TestClientRemoteIOBackend(RemoteIOBackend):
    """Remote backend variant backed by an in-process FastAPI TestClient."""

    def __init__(self, client: TestClient) -> None:
        self._client = client


# ===========================================================================
# MountSpec validation
# ===========================================================================

class TestMountSpec:
    def test_valid_mount(self):
        with tempfile.TemporaryDirectory() as td:
            m = MountSpec(name="docs", source=Path(td))
            assert m.source == Path(td).resolve()
            assert m.name == "docs"
            assert m.readonly is False

    def test_readonly_mount(self):
        with tempfile.TemporaryDirectory() as td:
            m = MountSpec(name="ro", source=Path(td), readonly=True)
            assert m.readonly is True

    def test_nonexistent_source_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            MountSpec(name="bad", source=Path("/nonexistent_path_xyz_12345"))

    def test_slash_in_name_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError, match="simple name"):
                MountSpec(name="a/b", source=Path(td))

    def test_backslash_in_name_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError, match="simple name"):
                MountSpec(name="a\\b", source=Path(td))

    def test_sensitive_root_blocked_unix(self):
        if sys.platform == "win32":
            pytest.skip("Unix-specific test")
        if not Path("/etc").is_dir():
            pytest.skip("/etc not available")
        with pytest.raises(ValueError, match="sensitive"):
            MountSpec(name="etc", source=Path("/etc"))

    def test_sensitive_root_blocked_windows(self):
        if sys.platform != "win32":
            pytest.skip("Windows-specific test")
        win_dir = Path(os.environ.get("WINDIR", "C:\\Windows"))
        if not win_dir.is_dir():
            pytest.skip("Windows directory not found")
        with pytest.raises(ValueError, match="sensitive"):
            MountSpec(name="win", source=win_dir)


# ===========================================================================
# SandboxConfig mount setup
# ===========================================================================

class TestSandboxConfigMounts:
    def test_mount_links_created(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            link = ws / MOUNT_PREFIX / "Desktop"
            assert link.exists() or link.is_symlink()
            assert (link / "seed.txt").read_text() == "content from Desktop"

    def test_multiple_mounts(self):
        td, ws, mounts, mount_map = _ws_with_mounts(
            ("Desktop", False), ("Downloads", True),
        )
        with td:
            assert (ws / MOUNT_PREFIX / "Desktop" / "seed.txt").exists()
            assert (ws / MOUNT_PREFIX / "Downloads" / "seed.txt").exists()

    def test_duplicate_mount_name_raises(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            d1 = base / "d1"
            d2 = base / "d2"
            d1.mkdir()
            d2.mkdir()
            with pytest.raises(ValueError, match="Duplicate"):
                SandboxConfig(
                    workspace_dir=base / "ws",
                    mounts=[
                        MountSpec(name="same", source=d1),
                        MountSpec(name="same", source=d2),
                    ],
                    confinement="none",
                )

    def test_mount_map_property(self):
        td, ws, mounts, mount_map = _ws_with_mounts(
            ("Desktop", False), ("Downloads", True),
        )
        with td:
            assert "Desktop" in mount_map
            assert "Downloads" in mount_map
            assert mount_map["Downloads"].readonly is True

    def test_no_mounts_no_prefix_dir(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(workspace_dir=Path(td), confinement="none")
            assert not (cfg.workspace_dir / MOUNT_PREFIX).exists()


# ===========================================================================
# Mount-aware resolve_path
# ===========================================================================

class TestMountResolvePath:
    def test_resolve_mount_path(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            resolved = resolve_path(
                f"{MOUNT_PREFIX}/Desktop/seed.txt", ws, mounts=mount_map,
            )
            assert resolved.exists()
            assert resolved.read_text() == "content from Desktop"

    def test_resolve_mount_root(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            resolved = resolve_path(
                f"{MOUNT_PREFIX}/Desktop", ws, mounts=mount_map,
            )
            assert resolved.is_dir()

    def test_resolve_workspace_path_unchanged(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            (ws / "local.txt").write_text("local")
            resolved = resolve_path("local.txt", ws, mounts=mount_map)
            assert resolved == (ws / "local.txt").resolve()

    def test_mount_path_escape_blocked(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            with pytest.raises(PathConfinementError, match="mount boundary"):
                resolve_path(
                    f"{MOUNT_PREFIX}/Desktop/../../etc/passwd", ws, mounts=mount_map,
                )

    def test_workspace_path_escape_still_blocked(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            ws_sub = ws / "sub"
            ws_sub.mkdir()
            with pytest.raises(PathConfinementError, match="escapes workspace"):
                resolve_path("../../etc/passwd", ws, mounts=mount_map)

    def test_nonexistent_mount_falls_through_to_workspace(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            resolved = resolve_path(
                f"{MOUNT_PREFIX}/NonExistent/file.txt", ws, mounts=mount_map,
            )
            expected = (ws / MOUNT_PREFIX / "NonExistent" / "file.txt").resolve()
            assert resolved == expected

    def test_no_mounts_kwarg_behaves_as_before(self):
        td, ws = _simple_ws()
        with td:
            (ws / "file.txt").write_text("hi")
            resolved = resolve_path("file.txt", ws)
            assert resolved == (ws / "file.txt").resolve()


# ===========================================================================
# is_readonly_path
# ===========================================================================

class TestIsReadonlyPath:
    def test_readonly_mount_detected(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Downloads", True))
        with td:
            resolved = resolve_path(
                f"{MOUNT_PREFIX}/Downloads/seed.txt", ws, mounts=mount_map,
            )
            assert is_readonly_path(resolved, mount_map) is True

    def test_writable_mount_not_readonly(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            resolved = resolve_path(
                f"{MOUNT_PREFIX}/Desktop/seed.txt", ws, mounts=mount_map,
            )
            assert is_readonly_path(resolved, mount_map) is False

    def test_workspace_path_not_readonly(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Downloads", True))
        with td:
            (ws / "local.txt").write_text("local")
            resolved = resolve_path("local.txt", ws, mounts=mount_map)
            assert is_readonly_path(resolved, mount_map) is False

    def test_no_mounts_returns_false(self):
        td, ws = _simple_ws()
        with td:
            (ws / "file.txt").write_text("hi")
            resolved = resolve_path("file.txt", ws)
            assert is_readonly_path(resolved, None) is False


# ===========================================================================
# File ops through mounts
# ===========================================================================

class TestMountFileOps:
    def test_read_through_mount(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            result = ops.read_file(
                f"{MOUNT_PREFIX}/Desktop/seed.txt", ws, mounts=mount_map,
            )
            assert "content from Desktop" in result

    def test_write_through_mount(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            result = ops.write_file(
                f"{MOUNT_PREFIX}/Desktop/new.txt", "hello mount", ws, mounts=mount_map,
            )
            assert "Created" in result
            ext_dir = mounts[0].source
            assert (ext_dir / "new.txt").read_text() == "hello mount"

    def test_write_readonly_blocked(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Downloads", True))
        with td:
            result = ops.write_file(
                f"{MOUNT_PREFIX}/Downloads/hack.txt", "bad", ws, mounts=mount_map,
            )
            assert "TOOL ERROR" in result
            assert "ReadonlyMount" in result

    def test_edit_readonly_blocked(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Downloads", True))
        with td:
            revision = _extract_revision(ops.read_file(
                f"{MOUNT_PREFIX}/Downloads/seed.txt", ws, show_lines=True, mounts=mount_map,
            ))
            result = ops.edit_file(
                f"{MOUNT_PREFIX}/Downloads/seed.txt", 1, 1, "hacked", revision, ws, mounts=mount_map,
            )
            assert "TOOL ERROR" in result
            assert "ReadonlyMount" in result

    def test_delete_readonly_blocked(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Downloads", True))
        with td:
            result = ops.delete_file(
                f"{MOUNT_PREFIX}/Downloads/seed.txt", ws, mounts=mount_map,
            )
            assert "TOOL ERROR" in result
            assert "ReadonlyMount" in result
            assert mounts[0].source.joinpath("seed.txt").exists()

    def test_delete_through_writable_mount_moves_to_recycle_bin(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            result = ops.delete_file(
                f"{MOUNT_PREFIX}/Desktop/seed.txt", ws, mounts=mount_map,
            )
            assert "Moved to recycle bin" in result
            assert not mounts[0].source.joinpath("seed.txt").exists()
            recycled = ws / ".recycle_bin" / MOUNT_PREFIX / "Desktop" / "seed.txt"
            assert recycled.exists()

    def test_move_from_readonly_blocked(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Downloads", True))
        with td:
            result = ops.move_file(
                f"{MOUNT_PREFIX}/Downloads/seed.txt", "local_copy.txt", ws, mounts=mount_map,
            )
            assert "TOOL ERROR" in result
            assert "ReadonlyMount" in result

    def test_move_to_readonly_blocked(self):
        td, ws, mounts, mount_map = _ws_with_mounts(
            ("Desktop", False), ("Downloads", True),
        )
        with td:
            (ws / "moveme.txt").write_text("moving")
            result = ops.move_file(
                "moveme.txt", f"{MOUNT_PREFIX}/Downloads/moveme.txt", ws, mounts=mount_map,
            )
            assert "TOOL ERROR" in result
            assert "ReadonlyMount" in result

    def test_move_from_mount_to_workspace(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            result = ops.move_file(
                f"{MOUNT_PREFIX}/Desktop/seed.txt", "copied_from_mount.txt", ws, mounts=mount_map,
            )
            assert "Moved:" in result
            assert not mounts[0].source.joinpath("seed.txt").exists()
            assert (ws / "copied_from_mount.txt").read_text() == "content from Desktop"

    def test_move_between_writable_mounts(self):
        td, ws, mounts, mount_map = _ws_with_mounts(
            ("Desktop", False), ("Docs", False),
        )
        with td:
            result = ops.move_file(
                f"{MOUNT_PREFIX}/Desktop/seed.txt",
                f"{MOUNT_PREFIX}/Docs/moved.txt",
                ws,
                mounts=mount_map,
            )
            assert "Moved:" in result
            assert not mounts[0].source.joinpath("seed.txt").exists()
            assert mounts[1].source.joinpath("moved.txt").read_text() == "content from Desktop"

    def test_edit_through_writable_mount(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            revision = _extract_revision(ops.read_file(
                f"{MOUNT_PREFIX}/Desktop/seed.txt", ws, show_lines=True, mounts=mount_map,
            ))
            result = ops.edit_file(
                f"{MOUNT_PREFIX}/Desktop/seed.txt", 1, 1, "edited content", revision, ws, mounts=mount_map,
            )
            assert "Edited" in result
            assert mounts[0].source.joinpath("seed.txt").read_text() == "edited content"

    def test_list_mount_directory(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            result = ops.list_files(
                f"{MOUNT_PREFIX}/Desktop", ws, mounts=mount_map,
            )
            assert "seed.txt" in result

    def test_list_workspace_shows_mount_prefix(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            result = ops.list_files(".", ws, mounts=mount_map)
            assert MOUNT_PREFIX in result

    def test_set_directory_create_in_mount(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            result = ops.set_directory(
                "create", f"{MOUNT_PREFIX}/Desktop/subdir", ws, mounts=mount_map,
            )
            assert "created" in result.lower()
            assert mounts[0].source.joinpath("subdir").is_dir()

    def test_set_directory_rename_in_mount(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            mounts[0].source.joinpath("old_name").mkdir()
            result = ops.set_directory(
                "rename",
                f"{MOUNT_PREFIX}/Desktop/old_name",
                ws,
                new_name="renamed",
                mounts=mount_map,
            )
            assert "Renamed:" in result
            assert not mounts[0].source.joinpath("old_name").exists()
            assert mounts[0].source.joinpath("renamed").is_dir()

    def test_set_directory_move_between_mounts(self):
        td, ws, mounts, mount_map = _ws_with_mounts(
            ("Desktop", False), ("Docs", False),
        )
        with td:
            mounts[0].source.joinpath("folder_a").mkdir()
            (mounts[0].source / "folder_a" / "inside.txt").write_text("data")
            result = ops.set_directory(
                "move",
                f"{MOUNT_PREFIX}/Desktop/folder_a",
                ws,
                destination=f"{MOUNT_PREFIX}/Docs/folder_b",
                mounts=mount_map,
            )
            assert "Moved directory:" in result
            assert not mounts[0].source.joinpath("folder_a").exists()
            assert mounts[1].source.joinpath("folder_b", "inside.txt").read_text() == "data"

    def test_set_directory_delete_in_mount_moves_to_recycle_bin(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            mounts[0].source.joinpath("folder_a").mkdir()
            (mounts[0].source / "folder_a" / "inside.txt").write_text("data")
            result = ops.set_directory(
                "delete",
                f"{MOUNT_PREFIX}/Desktop/folder_a",
                ws,
                mounts=mount_map,
            )
            assert "Moved directory to recycle bin" in result
            assert not mounts[0].source.joinpath("folder_a").exists()
            recycled = ws / ".recycle_bin" / MOUNT_PREFIX / "Desktop" / "folder_a" / "inside.txt"
            assert recycled.exists()

    def test_set_directory_delete_readonly_blocked(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Downloads", True))
        with td:
            result = ops.set_directory(
                "delete", f"{MOUNT_PREFIX}/Downloads", ws, mounts=mount_map,
            )
            assert "TOOL ERROR" in result
            assert "ReadonlyMount" in result


class TestRemoteMountFileOps:
    def test_remote_backend_full_crud_through_mount(self):
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            app = FastAPI()
            app.include_router(create_service_router(str(ws), None))
            client = TestClient(app)
            backend = _TestClientRemoteIOBackend(client)

            seed_path = f"{MOUNT_PREFIX}/Desktop/seed.txt"
            new_path = f"{MOUNT_PREFIX}/Desktop/new.txt"
            subdir_path = f"{MOUNT_PREFIX}/Desktop/subdir"
            moved_path = f"{subdir_path}/new.txt"

            assert backend.read_text(seed_path) == "content from Desktop"

            backend.write_text(new_path, "hello mount")
            assert mounts[0].source.joinpath("new.txt").read_text() == "hello mount"

            backend.mkdir(subdir_path)
            assert mounts[0].source.joinpath("subdir").is_dir()

            backend.move(new_path, moved_path)
            assert not mounts[0].source.joinpath("new.txt").exists()
            assert mounts[0].source.joinpath("subdir", "new.txt").read_text() == "hello mount"

            backend.delete(moved_path)
            assert not mounts[0].source.joinpath("subdir", "new.txt").exists()

            backend.delete_tree(subdir_path)
            assert not mounts[0].source.joinpath("subdir").exists()

            client.close()


# ===========================================================================
# Confinement detection
# ===========================================================================

class TestConfinementDetection:
    def test_auto_detection_returns_string(self):
        result = _detect_confinement()
        assert result in ("bwrap", "none")

    def test_none_on_non_linux(self):
        if sys.platform.startswith("linux"):
            pytest.skip("Test is for non-Linux")
        assert _detect_confinement() == "none"

    def test_sandbox_config_confinement_auto(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(workspace_dir=Path(td))
            assert cfg.confinement in ("bwrap", "none")

    def test_sandbox_config_confinement_none(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(workspace_dir=Path(td), confinement="none")
            assert cfg.confinement == "none"


# ===========================================================================
# bwrap command building
# ===========================================================================

class TestBwrapCommand:
    def test_build_basic_command(self):
        from arion_agent.environments._sandbox.confinement import build_bwrap_command

        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(workspace_dir=Path(td), confinement="bwrap")
            cmd = build_bwrap_command(["echo", "hello"], cfg)
            assert cmd[0] == "bwrap"
            assert "echo" in cmd
            assert "hello" in cmd
            assert "--" in cmd
            assert "--die-with-parent" in cmd

    def test_workspace_bound(self):
        from arion_agent.environments._sandbox.confinement import build_bwrap_command

        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(workspace_dir=Path(td), confinement="bwrap")
            cmd = build_bwrap_command(["ls"], cfg)
            ws_str = str(cfg.workspace_dir)
            bind_idx = cmd.index("--bind")
            assert cmd[bind_idx + 1] == ws_str
            assert cmd[bind_idx + 2] == ws_str

    def test_mount_bindings(self):
        from arion_agent.environments._sandbox.confinement import build_bwrap_command

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            ext = base / "ext"
            ext.mkdir()
            ws = base / "ws"
            ws.mkdir()
            mount = MountSpec(name="ext", source=ext)
            cfg = SandboxConfig(
                workspace_dir=ws, mounts=[mount], confinement="bwrap",
            )
            cmd = build_bwrap_command(["ls"], cfg)
            cmd_str = " ".join(cmd)
            assert str(ext) in cmd_str
            expected_dest = str(ws / MOUNT_PREFIX / "ext")
            assert expected_dest in cmd_str

    def test_readonly_mount_uses_ro_bind(self):
        from arion_agent.environments._sandbox.confinement import build_bwrap_command

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            ext = base / "ro_ext"
            ext.mkdir()
            ws = base / "ws"
            ws.mkdir()
            mount = MountSpec(name="ro_ext", source=ext, readonly=True)
            cfg = SandboxConfig(
                workspace_dir=ws, mounts=[mount], confinement="bwrap",
            )
            cmd = build_bwrap_command(["ls"], cfg)
            ext_str = str(ext)
            idx = cmd.index(ext_str)
            assert cmd[idx - 1] == "--ro-bind"

    def test_network_blocked_by_default(self):
        from arion_agent.environments._sandbox.confinement import build_bwrap_command

        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(
                workspace_dir=Path(td), confinement="bwrap", network_allowed=False,
            )
            cmd = build_bwrap_command(["ls"], cfg)
            assert "--unshare-net" in cmd

    def test_network_allowed(self):
        from arion_agent.environments._sandbox.confinement import build_bwrap_command

        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(
                workspace_dir=Path(td), confinement="bwrap", network_allowed=True,
            )
            cmd = build_bwrap_command(["ls"], cfg)
            assert "--unshare-net" not in cmd

    def test_build_bwrap_shell(self):
        from arion_agent.environments._sandbox.confinement import build_bwrap_shell

        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(workspace_dir=Path(td), confinement="bwrap")
            cmd = build_bwrap_shell("/bin/bash", cfg)
            assert cmd[0] == "bwrap"
            assert "/bin/bash" in cmd
            assert "-i" in cmd


# ===========================================================================
# FileEnvironment system prompt
# ===========================================================================

class TestFileEnvironmentPrompt:
    def test_no_mounts_basic_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SandboxConfig(workspace_dir=Path(td), confinement="none")
            from arion_agent.environments.file.middleware import FileEnvironment
            env = FileEnvironment(cfg)
            parts: list[str] = []
            env.wrap_system_message(parts)
            prompt = "\n".join(parts)
            assert "File Environment" in prompt
            assert MOUNT_PREFIX not in prompt

    def test_mounts_in_prompt(self):
        td, ws, mounts, mount_map = _ws_with_mounts(
            ("Desktop", False), ("Downloads", True),
        )
        with td:
            cfg = SandboxConfig(
                workspace_dir=ws, mounts=mounts, confinement="none",
            )
            from arion_agent.environments.file.middleware import FileEnvironment
            env = FileEnvironment(cfg)
            parts: list[str] = []
            env.wrap_system_message(parts)
            prompt = "\n".join(parts)
            assert f"{MOUNT_PREFIX}/Desktop/" in prompt
            assert f"{MOUNT_PREFIX}/Downloads/" in prompt
            assert "(readonly)" in prompt


# ===========================================================================
# Mount link creation
# ===========================================================================

class TestMountLinkCreation:
    def test_create_mount_link(self):
        from arion_agent.environments._sandbox.confinement import create_mount_link

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            target = base / "target"
            target.mkdir()
            (target / "file.txt").write_text("linked")

            link = base / "link"
            link_type = create_mount_link(link, target)

            assert link_type in ("symlink", "junction")
            assert (link / "file.txt").read_text() == "linked"

    def test_bidirectional_write(self):
        from arion_agent.environments._sandbox.confinement import create_mount_link

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            target = base / "target"
            target.mkdir()

            link = base / "link"
            create_mount_link(link, target)

            (link / "from_link.txt").write_text("via link")
            assert (target / "from_link.txt").read_text() == "via link"

            (target / "from_target.txt").write_text("via target")
            assert (link / "from_target.txt").read_text() == "via target"


# ===========================================================================
# Integration: full SandboxConfig -> ops round-trip
# ===========================================================================

class TestMountIntegration:
    def test_full_roundtrip_read_write(self):
        """Agent writes via mount path, file appears on real external dir."""
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            ops.write_file(
                f"{MOUNT_PREFIX}/Desktop/agent_output.md",
                "# Summary\nAgent wrote this.",
                ws,
                mounts=mount_map,
            )
            ext_file = mounts[0].source / "agent_output.md"
            assert ext_file.exists()
            assert ext_file.read_text() == "# Summary\nAgent wrote this."

            result = ops.read_file(
                f"{MOUNT_PREFIX}/Desktop/agent_output.md", ws, mounts=mount_map,
            )
            assert "Agent wrote this" in result

    def test_external_change_visible_to_agent(self):
        """Changes made outside the agent are immediately visible."""
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            ext_dir = mounts[0].source
            (ext_dir / "external_change.txt").write_text("changed externally")

            result = ops.read_file(
                f"{MOUNT_PREFIX}/Desktop/external_change.txt", ws, mounts=mount_map,
            )
            assert "changed externally" in result

    def test_mixed_workspace_and_mount_ops(self):
        """Agent can work with both workspace files and mount files."""
        td, ws, mounts, mount_map = _ws_with_mounts(("Desktop", False))
        with td:
            ops.write_file("local.txt", "local content", ws, mounts=mount_map)
            ops.write_file(
                f"{MOUNT_PREFIX}/Desktop/mount.txt", "mount content", ws, mounts=mount_map,
            )

            local = ops.read_file("local.txt", ws, mounts=mount_map)
            mount = ops.read_file(
                f"{MOUNT_PREFIX}/Desktop/mount.txt", ws, mounts=mount_map,
            )
            assert "local content" in local
            assert "mount content" in mount


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath

# Baked-in ignores for workspaces outside git repos (Desktop, temp folders, etc.)
DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    # Version control
    ".git/",
    ".svn/",
    ".hg/",
    # Python / uv / pip
    "__pycache__/",
    "*.py[cod]",
    ".venv/",
    "venv/",
    ".wsl_test_venv/",
    "site-packages/",
    ".uv/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    ".tox/",
    ".nox/",
    "*.egg-info/",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    # Node / front-end build
    "node_modules/",
    "dist/",
    "build/",
    ".next/",
    ".nuxt/",
    ".turbo/",
    "coverage/",
    ".parcel-cache/",
    ".pnpm-store/",
    ".yarn/",
    "bower_components/",
    # Rust / Go / Java build
    "target/",
    "vendor/",
    # IDE / editor
    ".idea/",
    ".vscode/",
    ".cursor/",
    # OS junk
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "__MACOSX/",
    ".Trash/",
    # Index / agent runtime (never index the index)
    ".arion/",
    ".index/",
    ".run/",
    ".cache/",
    ".recycle_bin/",
    ".zsh_sessions/",
    # Secrets / credentials
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "credentials.json",
    "secrets.json",
    # Logs / runtime
    "*.log",
    "*.err",
    "*.pid",
    # Large / binary / model artifacts
    "*.zip",
    "*.tar",
    "*.gz",
    "*.7z",
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.mp4",
    "*.mp3",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.whl",
    "*.sqlite",
    "*.db",
    "*.pkl",
    "*.pickle",
    "*.npy",
    "*.npz",
    "*.h5",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.bin",
    # Minified / generated bundles
    "*.min.js",
    "*.min.css",
    "*.map",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
)


def load_ignore_patterns(
    workspace: Path,
    *,
    extra_patterns: list[str] | None = None,
    searchignore_path: Path | None = None,
) -> list[str]:
    patterns: list[str] = list(DEFAULT_IGNORE_PATTERNS)

    gitignore = workspace / ".gitignore"
    if gitignore.is_file():
        patterns.extend(_parse_ignore_file(gitignore))

    local = searchignore_path or (workspace / ".searchignore")
    if local.is_file():
        patterns.extend(_parse_ignore_file(local))

    if extra_patterns:
        patterns.extend(extra_patterns)

    return patterns


def _parse_ignore_file(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def should_ignore(rel_posix: str, patterns: list[str]) -> bool:
    name = rel_posix.rsplit("/", 1)[-1]
    parts = rel_posix.split("/")
    for pattern in patterns:
        if pattern.endswith("/"):
            dir_name = pattern[:-1]
            if dir_name in parts:
                return True
            if rel_posix == dir_name or rel_posix.startswith(dir_name + "/"):
                return True
            continue
        if "/" in pattern:
            if fnmatch.fnmatch(rel_posix, pattern):
                return True
        else:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(rel_posix, pattern):
                return True
    return False


def _ignore_dirname(name: str, parent_rel: str, patterns: list[str]) -> bool:
    rel = name if not parent_rel else f"{parent_rel}/{name}"
    return should_ignore(rel, patterns)


def _rel_depth(rel_dir: str) -> int:
    if not rel_dir or rel_dir == ".":
        return 0
    return len(rel_dir.split("/"))


def iter_indexable_files(
    workspace: Path,
    patterns: list[str],
    extensions: set[str],
    *,
    max_depth: int | None = None,
    only: tuple[str, ...] = (),
    skip: tuple[str, ...] = (),
    allow: tuple[str, ...] = (),
) -> list[Path]:
    from arion_agent.semantic_search.scope import (
        path_indexable,
        should_prune_directory,
        walk_roots,
    )

    files: list[Path] = []
    workspace = workspace.resolve()

    for walk_root in walk_roots(workspace, only=only, allow=allow):
        for dirpath, dirnames, filenames in os.walk(str(walk_root), followlinks=False):
            rel_dir = Path(dirpath).relative_to(workspace).as_posix()
            if rel_dir == ".":
                rel_dir = ""

            if max_depth is not None and _rel_depth(rel_dir) >= max_depth:
                dirnames.clear()
                continue

            dirnames[:] = [
                d for d in dirnames
                if not should_prune_directory(
                    f"{rel_dir}/{d}" if rel_dir else d,
                    patterns,
                    only=only,
                    skip=skip,
                    allow=allow,
                )
            ]

            for name in filenames:
                path = Path(dirpath) / name
                rel = path.relative_to(workspace).as_posix()
                if not path_indexable(
                    rel,
                    patterns,
                    only=only,
                    skip=skip,
                    allow=allow,
                ):
                    continue
                if path.suffix.lower() not in extensions:
                    continue
                if _looks_binary(path):
                    continue
                files.append(path)

    files.sort()
    return files


def path_matches_glob(path: str, glob_pattern: str) -> bool:
    """Match workspace-relative path against a glob (supports **)."""
    norm = path.replace("\\", "/")
    pattern = glob_pattern.replace("\\", "/").strip()
    if not pattern:
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3].strip("/")
        return norm == prefix or norm.startswith(prefix + "/")
    if "**" in pattern:
        # pathlib's ** requires at least one directory component.
        # "**/*.md" won't match root-level "readme.md" without this.
        if "/" not in norm:
            return PurePosixPath(f"_/{norm}").match(pattern)
        return PurePosixPath(norm).match(pattern)
    return fnmatch.fnmatch(norm, pattern)


def _looks_binary(path: Path) -> bool:
    sample = path.read_bytes()[:8192]
    return b"\x00" in sample

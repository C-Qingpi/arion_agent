from __future__ import annotations

import fnmatch
import os
from pathlib import Path

# Baked-in ignores for workspaces outside git repos (Desktop, temp folders, etc.)
DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    # Version control
    ".git/",
    ".svn/",
    ".hg/",
    # Python
    "__pycache__/",
    "*.py[cod]",
    ".venv/",
    "venv/",
    ".wsl_test_venv/",
    "site-packages/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    "*.egg-info/",
    # Node / front-end build
    "node_modules/",
    "dist/",
    "build/",
    ".next/",
    ".nuxt/",
    ".turbo/",
    "coverage/",
    ".parcel-cache/",
    # IDE / editor
    ".idea/",
    ".vscode/",
    ".cursor/",
    # OS junk
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    # Index storage (never index the index)
    ".arion/",
    ".index/",
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
    # Large / binary media
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


def iter_indexable_files(
    workspace: Path,
    patterns: list[str],
    extensions: set[str],
) -> list[Path]:
    files: list[Path] = []
    root = str(workspace)

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(workspace).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        dirnames[:] = [
            d for d in dirnames
            if not _ignore_dirname(d, rel_dir, patterns)
        ]

        for name in filenames:
            rel = name if not rel_dir else f"{rel_dir}/{name}"
            if should_ignore(rel, patterns):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() not in extensions:
                continue
            if _looks_binary(path):
                continue
            files.append(path)

    files.sort()
    return files


def _looks_binary(path: Path) -> bool:
    sample = path.read_bytes()[:8192]
    return b"\x00" in sample

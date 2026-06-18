"""Shared file persistence utilities for ArionAgent middleware.

Core contract: SEED IF ABSENT, NEVER OVERWRITE.

Every middleware that bootstraps files (identity, skills, subagent rosters)
uses these functions to ensure agent-modified files are preserved across
restarts, while fresh agents get properly seeded.

When an IOBackend is configured (via set_default_backend), all file I/O
routes through the backend. This enables remote I/O for Docker deployments
where grpcfuse bind mounts degrade long-lived file handles (Phase 17+).
Without a backend configured, functions use direct Path operations for
backward compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arion_agent.util.io_backend import IOBackend

logger = logging.getLogger(__name__)

_default_backend: IOBackend | None = None
_workspace_root: Path | None = None


def set_default_backend(backend: IOBackend, workspace_root: Path) -> None:
    """Set the module-level I/O backend used by all persistence functions.

    Called once during create_arion_agent. All subsequent persistence calls
    route through this backend instead of direct Path operations.
    """
    global _default_backend, _workspace_root
    _default_backend = backend
    _workspace_root = workspace_root


def _to_rel(path: Path) -> str:
    """Convert an absolute Path to a workspace-relative string for the backend."""
    assert _workspace_root is not None
    try:
        return str(path.relative_to(_workspace_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def seed_file(path: Path, content: str) -> bool:
    """Write content to path ONLY if the file does not exist.

    Creates parent directories as needed. Writes atomically via temp file
    to prevent partial files on crash.

    Returns True if file was created, False if it already existed.
    """
    if _default_backend is not None:
        rel = _to_rel(path)
        parent = _to_rel(path.parent)
        _default_backend.mkdir(parent)
        if _default_backend.exists(rel):
            return False
        _default_backend.write_text(rel, content)
        logger.debug("Seeded %s (%d chars)", path, len(content))
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    logger.debug("Seeded %s (%d chars)", path, len(content))
    return True


def seed_json(path: Path, data: dict) -> bool:
    """Write JSON data to path ONLY if the file does not exist."""
    return seed_file(path, json.dumps(data, indent=2, ensure_ascii=False))


def ensure_directory(path: Path) -> bool:
    """Create directory and parents if absent. Returns True if created."""
    if _default_backend is not None:
        rel = _to_rel(path)
        existed = _default_backend.exists(rel)
        _default_backend.mkdir(rel)
        return not existed

    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    return not existed


def cleanup_partial_seeds(directory: Path) -> None:
    """Remove orphaned .tmp files from interrupted seed operations."""
    if _default_backend is not None:
        rel = _to_rel(directory)
        if not _default_backend.exists(rel):
            return
        for tmp_path in _default_backend.glob(rel, "*.tmp"):
            try:
                _default_backend.delete(tmp_path)
            except OSError:
                pass
            logger.debug("Cleaned up partial seed: %s", tmp_path)
        return

    if not directory.exists():
        return
    for tmp in directory.rglob("*.tmp"):
        tmp.unlink(missing_ok=True)
        logger.debug("Cleaned up partial seed: %s", tmp)


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file.

    Creates parent directories if needed. Each record is one line.
    Used by SignalStore and SessionLogger for append-only persistence.
    """
    line = json.dumps(record, ensure_ascii=False) + "\n"

    if _default_backend is not None:
        rel = _to_rel(path)
        parent = _to_rel(path.parent)
        _default_backend.mkdir(parent)
        _default_backend.append_text(rel, line)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file. Returns [] if file does not exist."""
    if _default_backend is not None:
        rel = _to_rel(path)
        if not _default_backend.exists(rel):
            return []
        text = _default_backend.read_text(rel)
        records = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records

    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_last_jsonl_record(path: Path) -> dict | None:
    """Read the last complete JSON record from a JSONL file.

    Returns None if file is empty or missing.
    Raises json.JSONDecodeError on corrupted content (corruption is explicit).
    When using a backend, reads the full file and scans from end (JSONL
    files are small; simplicity over micro-optimization).
    """
    if _default_backend is not None:
        rel = _to_rel(path)
        if not _default_backend.exists(rel):
            return None
        text = _default_backend.read_text(rel)
        if not text.strip():
            return None
        for raw_line in reversed(text.splitlines()):
            stripped = raw_line.strip()
            if stripped:
                return json.loads(stripped)
        return None

    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        if pos == 0:
            return None
        buf = b""
        while pos > 0:
            pos -= 1
            f.seek(pos)
            char = f.read(1)
            if char == b"\n" and buf.strip():
                break
            buf = char + buf
        else:
            buf = buf if buf.strip() else b""
        if not buf.strip():
            return None
        return json.loads(buf.strip().decode("utf-8"))


# --- Additional helpers for 17+.4 direct-Path writer migration ---

def read_file_text(path: Path, max_chars: int | None = None, encoding: str = "utf-8") -> str:
    """Read a text file, optionally truncating to max_chars."""
    if _default_backend is not None:
        rel = _to_rel(path)
        text = _default_backend.read_text(rel, encoding=encoding)
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars]
        return text

    text = path.read_text(encoding=encoding)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text


def write_file(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to a file (atomic overwrite). Creates parent dirs."""
    if _default_backend is not None:
        rel = _to_rel(path)
        parent = _to_rel(path.parent)
        _default_backend.mkdir(parent)
        _default_backend.write_text(rel, content, encoding=encoding)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def rewrite_jsonl(path: Path, records: list[dict]) -> None:
    """Atomically rewrite a JSONL file with the given records."""
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    write_file(path, content)


def touch(path: Path) -> None:
    """Create an empty file if it doesn't exist, or update mtime if it does."""
    if _default_backend is not None:
        rel = _to_rel(path)
        parent = _to_rel(path.parent)
        _default_backend.mkdir(parent)
        if not _default_backend.exists(rel):
            _default_backend.write_text(rel, "")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def append_file(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Append text content to a file. Creates parent dirs if needed."""
    if _default_backend is not None:
        rel = _to_rel(path)
        parent = _to_rel(path.parent)
        _default_backend.mkdir(parent)
        _default_backend.append_text(rel, content, encoding=encoding)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding=encoding) as f:
        f.write(content)


def file_exists(path: Path) -> bool:
    """Check if a file or directory exists."""
    if _default_backend is not None:
        return _default_backend.exists(_to_rel(path))
    return path.exists()


def is_directory(path: Path) -> bool:
    """Check if a path is a directory."""
    if _default_backend is not None:
        rel = _to_rel(path)
        return _default_backend.exists(rel) and _default_backend.is_dir(rel)
    return path.is_dir()


def list_dir_entries(path: Path) -> list[tuple[str, bool]]:
    """List directory children as (name, is_dir) tuples.

    Returns sorted list. Uses IOBackend.list_dir when available.
    """
    if _default_backend is not None:
        rel = _to_rel(path)
        entries = _default_backend.list_dir(rel)
        return [(e.name, e.is_dir) for e in entries]

    if not path.is_dir():
        return []
    return sorted([(item.name, item.is_dir()) for item in path.iterdir()])


def glob_files(path: Path, pattern: str) -> list[Path]:
    """Glob for files under path matching pattern.

    Returns absolute Paths for local backend, or workspace-relative Paths
    for remote backend.
    """
    if _default_backend is not None:
        rel = _to_rel(path)
        results = _default_backend.glob(rel, pattern)
        if _workspace_root is not None:
            return [_workspace_root / r for r in results]
        return [Path(r) for r in results]

    return sorted(path.glob(pattern))


def workspace_relative_path(absolute_path: Path, workspace_dir: Path) -> str:
    """Convert an absolute path to a workspace-relative path string.

    Used by middleware when injecting file paths into prompts. The agent
    uses workspace-relative paths with file tools (read_file, list_files).

    Returns a forward-slash path relative to workspace_dir. If the path
    is not under workspace_dir, returns the path relative to .arion/ or
    falls back to the last meaningful segments.
    """
    try:
        rel = absolute_path.resolve().relative_to(workspace_dir.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        pass
    abs_str = str(absolute_path).replace("\\", "/")
    idx = abs_str.find(".arion/")
    if idx >= 0:
        return abs_str[idx:]
    return abs_str

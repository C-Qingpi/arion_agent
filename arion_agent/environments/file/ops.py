"""Pure-Python file operations, all path-confined to workspace.

When a remote IOBackend is configured (via persistence.set_default_backend),
functions dispatch to the backend using relative paths. The remote service
handles confinement and mount resolution. When local (default), functions
use resolve_path + direct Path operations as before.
"""

from __future__ import annotations

import fnmatch
import hashlib
import mimetypes
import os
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from arion_agent.environments._sandbox.paths import (
    PathConfinementError,
    format_path_confinement_tool_error,
    is_readonly_path,
    resolve_path,
)

if TYPE_CHECKING:
    from arion_agent.environments._sandbox.config import MountSpec
    from arion_agent.util.io_backend import IOBackend


def _resolve_path_for_tool(
    user_path: str,
    workspace: Path,
    tool_name: str,
    *,
    mounts: dict[str, MountSpec] | None = None,
) -> Path | str:
    """Resolve a user path or return a formatted TOOL ERROR string."""
    try:
        return resolve_path(user_path, workspace, mounts=mounts)
    except PathConfinementError as exc:
        return format_path_confinement_tool_error(
            tool_name, user_path, workspace, exc, mounts=mounts
        )


def _get_remote_backend() -> IOBackend | None:
    """Return the active IOBackend if it is remote, else None.

    Local backend and no-backend cases return None so callers fall through
    to the existing resolve_path + direct Path logic.
    """
    from arion_agent.util.io_backend import LocalIOBackend
    from arion_agent.util.persistence import _default_backend
    if _default_backend is not None and not isinstance(_default_backend, LocalIOBackend):
        return _default_backend
    return None

TEXT_EXTENSIONS = frozenset({
    ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".csv", ".tsv", ".html", ".htm", ".xml", ".css", ".js", ".ts", ".jsx",
    ".tsx", ".sh", ".bash", ".zsh", ".bat", ".ps1", ".rb", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".r", ".sql", ".lua", ".pl", ".php", ".vue", ".svelte", ".astro",
    ".env", ".gitignore", ".dockerignore", ".editorconfig", ".prettierrc",
    ".eslintrc", ".lock", ".log", ".diff", ".patch", ".rst", ".tex",
    ".makefile", ".cmake", "", ".dockerfile",
})

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})

BINARY_EXTRACTABLE = frozenset({".pdf", ".docx", ".xlsx", ".pptx", ".epub"})

LINE_MARKER_PREFIX = "L"
MAX_LINE_DISPLAY_CHARS = 2000
DEFAULT_LINE_WINDOW = 200


@dataclass(frozen=True)
class _UndoSnapshot:
    path: str
    exists: bool
    data: bytes | None = None
    digest: str | None = None


@dataclass(frozen=True)
class _UndoRecord:
    token: str
    operation: str
    before: _UndoSnapshot
    after: _UndoSnapshot


_LAST_UNDO_RECORDS: dict[str, _UndoRecord] = {}
_UNDO_LOCK = Lock()


def _is_text_file(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return True
    if ext in IMAGE_EXTENSIONS or ext in BINARY_EXTRACTABLE:
        return False
    if not ext:
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
            chunk.decode("utf-8")
            return True
        except (UnicodeDecodeError, OSError):
            return False
    return False


def _file_type_label(path: Path) -> str:
    ext = path.suffix.lower()
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    if ext:
        return f"{ext} file"
    return "unknown"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _content_revision(content: str) -> str:
    """Return a short stable revision token for text content."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"rev:{digest}"


def _bytes_digest(data: bytes) -> str:
    """Return a full digest for raw file content (internal undo checks only)."""
    return hashlib.sha256(data).hexdigest()


def _make_undo_token() -> str:
    """Generate a short opaque token for the latest undoable file operation."""
    return f"undo:{uuid.uuid4().hex[:8]}"


def _undo_scope_key(workspace: Path, backend: IOBackend | None) -> str:
    """Return the scope key for the last undoable operation."""
    if backend is not None:
        return f"remote:{id(backend)}:{workspace}"
    return f"local:{workspace.resolve()}"


def _store_undo_record(workspace: Path, backend: IOBackend | None, record: _UndoRecord) -> None:
    """Store the last undoable operation for this workspace/backend scope."""
    with _UNDO_LOCK:
        _LAST_UNDO_RECORDS[_undo_scope_key(workspace, backend)] = record


def _pop_undo_record(workspace: Path, backend: IOBackend | None) -> _UndoRecord | None:
    """Remove and return the last undoable operation for this workspace/backend scope."""
    with _UNDO_LOCK:
        return _LAST_UNDO_RECORDS.pop(_undo_scope_key(workspace, backend), None)


def _clear_undo_record(workspace: Path, backend: IOBackend | None) -> None:
    """Invalidate the last undoable operation for this workspace/backend scope."""
    with _UNDO_LOCK:
        _LAST_UNDO_RECORDS.pop(_undo_scope_key(workspace, backend), None)


def _peek_undo_record(workspace: Path, backend: IOBackend | None) -> _UndoRecord | None:
    """Return the last undoable operation without consuming it."""
    with _UNDO_LOCK:
        return _LAST_UNDO_RECORDS.get(_undo_scope_key(workspace, backend))


def _snapshot_bytes(path: str, data: bytes) -> _UndoSnapshot:
    """Build a snapshot for an existing file."""
    return _UndoSnapshot(path=path, exists=True, data=data, digest=_bytes_digest(data))


def _snapshot_missing(path: str) -> _UndoSnapshot:
    """Build a snapshot for a missing file."""
    return _UndoSnapshot(path=path, exists=False)


def _capture_backend_snapshot(backend: IOBackend, path: str) -> _UndoSnapshot:
    """Capture a file snapshot through a remote backend."""
    if not backend.exists(path):
        return _snapshot_missing(path)
    return _snapshot_bytes(path, backend.read_bytes(path))


def _capture_local_snapshot(path: str, resolved: Path) -> _UndoSnapshot:
    """Capture a file snapshot through direct filesystem access."""
    if not resolved.exists():
        return _snapshot_missing(path)
    return _snapshot_bytes(path, resolved.read_bytes())


def _validate_current_snapshot(
    snapshot: _UndoSnapshot,
    *,
    workspace: Path,
    backend: IOBackend | None,
    mounts: dict[str, MountSpec] | None = None,
) -> str | None:
    """Verify that the current filesystem still matches the stored post-operation snapshot."""
    if backend is not None:
        current_exists = backend.exists(snapshot.path)
        if current_exists != snapshot.exists:
            expected = "exist" if snapshot.exists else "be absent"
            return (f"TOOL ERROR (undo_file_operation)\nType: UndoConflict\n"
                    f"Message: Cannot undo because {snapshot.path} no longer matches the last operation. "
                    f"Expected it to {expected}.")
        if current_exists:
            current_data = backend.read_bytes(snapshot.path)
            if _bytes_digest(current_data) != snapshot.digest:
                return (f"TOOL ERROR (undo_file_operation)\nType: UndoConflict\n"
                        f"Message: Cannot undo because {snapshot.path} changed after the last operation.")
        return None

    resolved = resolve_path(snapshot.path, workspace, mounts=mounts)
    current_exists = resolved.exists()
    if current_exists != snapshot.exists:
        expected = "exist" if snapshot.exists else "be absent"
        return (f"TOOL ERROR (undo_file_operation)\nType: UndoConflict\n"
                f"Message: Cannot undo because {snapshot.path} no longer matches the last operation. "
                f"Expected it to {expected}.")
    if current_exists:
        current_data = resolved.read_bytes()
        if _bytes_digest(current_data) != snapshot.digest:
            return (f"TOOL ERROR (undo_file_operation)\nType: UndoConflict\n"
                    f"Message: Cannot undo because {snapshot.path} changed after the last operation.")
    return None


def _path_exists(
    path: str,
    *,
    workspace: Path,
    backend: IOBackend | None,
    mounts: dict[str, MountSpec] | None = None,
) -> bool:
    """Check whether a file path currently exists."""
    if backend is not None:
        return backend.exists(path)
    return resolve_path(path, workspace, mounts=mounts).exists()


def _delete_file_at_path(
    path: str,
    *,
    workspace: Path,
    backend: IOBackend | None,
    mounts: dict[str, MountSpec] | None = None,
) -> None:
    """Delete a file if it exists."""
    if backend is not None:
        if backend.exists(path):
            backend.delete(path)
        return
    resolved = resolve_path(path, workspace, mounts=mounts)
    if resolved.exists():
        resolved.unlink()


def _write_bytes_to_path(
    path: str,
    data: bytes,
    *,
    workspace: Path,
    backend: IOBackend | None,
    mounts: dict[str, MountSpec] | None = None,
) -> None:
    """Write raw bytes to a file path."""
    if backend is not None:
        backend.write_bytes(path, data)
        return
    resolved = resolve_path(path, workspace, mounts=mounts)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(data)

def _add_line_markers(lines: list[str], start_line: int = 1) -> str:
    """Add L{n}| markers. start_line is the 1-based file line of the first line in the list (for absolute line numbers)."""
    result = []
    for i, line in enumerate(lines, start_line):
        stripped = line.rstrip("\n\r")
        if len(stripped) <= MAX_LINE_DISPLAY_CHARS:
            result.append(f"{LINE_MARKER_PREFIX}{i}|{stripped}")
        else:
            seg = 0
            pos = 0
            while pos < len(stripped):
                chunk = stripped[pos:pos + MAX_LINE_DISPLAY_CHARS]
                if seg == 0:
                    result.append(f"{LINE_MARKER_PREFIX}{i}|{chunk}")
                else:
                    result.append(f"{LINE_MARKER_PREFIX}{i}.{seg}|{chunk}")
                seg += 1
                pos += MAX_LINE_DISPLAY_CHARS
    return "\n".join(result)


def _read_file_remote(
    backend: IOBackend,
    path: str,
    *,
    start_line: int = 1,
    end_line: int | None = None,
    show_lines: bool = False,
    max_readable_size: int = 10 * 1024 * 1024,
) -> str:
    """Read a file via remote backend."""
    try:
        if not backend.exists(path):
            return f"TOOL ERROR (read_file)\nType: FileNotFound\nMessage: File not found: {path}"
        if backend.is_dir(path):
            return f"TOOL ERROR (read_file)\nType: IsDirectory\nMessage: {path} is a directory. Use list_files instead."
    except Exception as exc:
        return f"TOOL ERROR (read_file)\nType: BackendError\nMessage: {exc}"

    ext = Path(path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        import base64
        try:
            data = base64.b64encode(backend.read_bytes(path)).decode("ascii")
        except Exception as exc:
            return f"TOOL ERROR (read_file)\nType: ReadError\nMessage: {exc}"
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/png")
        from arion_agent.util.multimodal import IMAGE_BLOCK_SENTINEL
        return f"{IMAGE_BLOCK_SENTINEL}:{mime}:{data}"

    if ext in BINARY_EXTRACTABLE:
        try:
            from markitdown import MarkItDown
            raw = backend.read_bytes(path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                md = MarkItDown()
                result = md.convert(tmp_path)
                text = result.text_content or ""
            finally:
                os.unlink(tmp_path)
            lines = text.split("\n")
            header = f"File: {path}\nType: {ext.lstrip('.')} (converted via MarkItDown)\nTotal lines: {len(lines)}\n"
            eff_end = min(end_line or (start_line + DEFAULT_LINE_WINDOW - 1), len(lines))
            eff_start = max(1, start_line)
            selected = lines[eff_start - 1:eff_end]
            header += f"Showing: lines {eff_start}-{eff_end}\n---\n"
            body = _add_line_markers(selected, eff_start) if show_lines else "\n".join(selected)
            footer = ""
            if eff_end < len(lines):
                footer = f"\n... {len(lines) - eff_end} more lines. Specify start_line/end_line to navigate."
            return header + body + footer
        except ImportError:
            try:
                st = backend.stat(path)
                return f"Binary file: {path} ({ext.lstrip('.')}, {_format_size(st.size)}). Install markitdown for text extraction."
            except Exception:
                return f"Binary file: {path} ({ext.lstrip('.')}). Install markitdown for text extraction."
        except Exception as exc:
            return f"TOOL ERROR (read_file)\nType: ExtractionFailed\nMessage: Failed to extract {path}: {exc}"

    try:
        content = backend.read_text(path)
    except Exception as exc:
        return f"TOOL ERROR (read_file)\nType: ReadError\nMessage: Cannot read {path}: {exc}"

    lines = content.split("\n")
    total = len(lines)
    eff_start = max(1, start_line)
    eff_end = min(end_line or (eff_start + DEFAULT_LINE_WINDOW - 1), total)
    revision = _content_revision(content)
    header = (
        f"File: {path}\n"
        f"Type: text\n"
        f"Revision: {revision}\n"
        f"Total lines: {total}\n"
        f"Showing: lines {eff_start}-{eff_end}\n---\n"
    )
    selected = lines[eff_start - 1:eff_end]
    body = _add_line_markers(selected, eff_start) if show_lines else "\n".join(selected)
    footer = ""
    if eff_end < total:
        footer = f"\n... {total - eff_end} more lines. Specify start_line/end_line to navigate."
    return header + body + footer


def read_file(
    path: str,
    workspace: Path,
    *,
    start_line: int = 1,
    end_line: int | None = None,
    show_lines: bool = False,
    max_readable_size: int = 10 * 1024 * 1024,
    mounts: dict[str, MountSpec] | None = None,
) -> str:
    """Read a file within the workspace."""
    backend = _get_remote_backend()
    if backend is not None:
        return _read_file_remote(backend, path, start_line=start_line, end_line=end_line,
                                 show_lines=show_lines, max_readable_size=max_readable_size)

    resolved = _resolve_path_for_tool(path, workspace, "read_file", mounts=mounts)
    if isinstance(resolved, str):
        return resolved

    if not resolved.exists():
        return f"TOOL ERROR (read_file)\nType: FileNotFound\nMessage: File not found: {path}"

    if resolved.is_dir():
        return f"TOOL ERROR (read_file)\nType: IsDirectory\nMessage: {path} is a directory. Use list_files instead."

    size = resolved.stat().st_size
    ext = resolved.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/png")
        from arion_agent.util.multimodal import IMAGE_BLOCK_SENTINEL
        file_uri = resolved.resolve().as_uri()
        return f"{IMAGE_BLOCK_SENTINEL}:{mime}:{file_uri}"

    if ext in BINARY_EXTRACTABLE:
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(str(resolved))
            text = result.text_content or ""
            if len(text.encode("utf-8")) > max_readable_size:
                return (
                    f"TOOL ERROR (read_file)\nType: FileTooLarge\n"
                    f"Message: Extracted text from {path} is too large ({_format_size(len(text.encode()))}).\n"
                    f"Use execute_python or execute_shell to process this file with scripts."
                )
            lines = text.split("\n")
            header = (
                f"File: {path}\n"
                f"Type: {ext.lstrip('.')} (converted via MarkItDown - content is extracted text, not raw file)\n"
                f"Total lines: {len(lines)}\n"
            )
            eff_end = min(end_line or (start_line + DEFAULT_LINE_WINDOW - 1), len(lines))
            eff_start = max(1, start_line)
            selected = lines[eff_start - 1:eff_end]
            header += f"Showing: lines {eff_start}-{eff_end}\n---\n"
            body = _add_line_markers(selected, eff_start) if show_lines else "\n".join(selected)
            footer = ""
            if eff_end < len(lines):
                footer = f"\n... {len(lines) - eff_end} more lines. Specify start_line/end_line to navigate."
            return header + body + footer
        except ImportError:
            return (
                f"Binary file: {path} ({ext.lstrip('.')}, {_format_size(size)}). "
                f"Install markitdown for text extraction: pip install markitdown[all]"
            )
        except Exception as exc:
            return f"TOOL ERROR (read_file)\nType: ExtractionFailed\nMessage: Failed to extract {path}: {exc}"

    if not _is_text_file(resolved):
        return (
            f"Binary file: {path} ({_file_type_label(resolved)}, {_format_size(size)}). "
            f"Cannot read as text. Use execute_python or execute_shell to process."
        )

    if size > max_readable_size:
        return (
            f"TOOL ERROR (read_file)\nType: FileTooLarge\n"
            f"Message: File too large ({_format_size(size)}). "
            f"Use execute_python or execute_shell to interact with scripts (e.g. head, grep, python)."
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = resolved.read_text(encoding="latin-1")
        except Exception as exc:
            return f"TOOL ERROR (read_file)\nType: EncodingError\nMessage: Cannot decode {path}: {exc}"

    lines = content.split("\n")
    total = len(lines)
    eff_start = max(1, start_line)
    eff_end = min(end_line or (eff_start + DEFAULT_LINE_WINDOW - 1), total)
    revision = _content_revision(content)

    header = (
        f"File: {path}\n"
        f"Type: {_file_type_label(resolved)}\n"
        f"Revision: {revision}\n"
        f"Total lines: {total}\n"
        f"Showing: lines {eff_start}-{eff_end}\n---\n"
    )
    selected = lines[eff_start - 1:eff_end]
    body = _add_line_markers(selected, eff_start) if show_lines else "\n".join(selected)
    footer = ""
    if eff_end < total:
        footer = f"\n... {total - eff_end} more lines. Specify start_line/end_line to navigate."
    return header + body + footer


def _check_readonly(path: str, resolved: Path, mounts: dict[str, MountSpec] | None, tool_name: str) -> str | None:
    """Return a TOOL ERROR string if the resolved path is in a readonly mount, else None."""
    if is_readonly_path(resolved, mounts):
        return (
            f"TOOL ERROR ({tool_name})\nType: ReadonlyMount\n"
            f"Message: {path} is inside a readonly mount and cannot be modified."
        )
    return None


def write_file(
    path: str,
    content: str,
    workspace: Path,
    *,
    mode: str = "create",
    mounts: dict[str, MountSpec] | None = None,
) -> str:
    """Write content to a file."""
    backend = _get_remote_backend()
    if backend is not None:
        try:
            size_str = _format_size(len(content.encode("utf-8")))
            parent = str(Path(path).parent).replace("\\", "/")
            if parent and parent != ".":
                backend.mkdir(parent)
            if mode == "create":
                if backend.exists(path):
                    return (f"TOOL ERROR (write_file)\nType: FileExists\n"
                            f"Message: File already exists: {path}. Use str_replace to modify, "
                            f"or write_file with mode='overwrite', 'append', or 'prepend'.")
                backend.write_text(path, content)
                _clear_undo_record(workspace, backend)
                return f"Created: {path} ({size_str})"
            elif mode == "overwrite":
                before = _capture_backend_snapshot(backend, path)
                backend.write_text(path, content)
                after = _capture_backend_snapshot(backend, path)
                token = _make_undo_token()
                _store_undo_record(workspace, backend, _UndoRecord(
                    token=token,
                    operation="write_file(overwrite)",
                    before=before,
                    after=after,
                ))
                verb = "Overwritten" if before.exists else "Created"
                return (f"{verb}: {path} ({size_str}). Undo token: {token}. "
                        "Undo only applies to the latest undoable file operation.")
            elif mode == "append":
                backend.append_text(path, content)
                _clear_undo_record(workspace, backend)
                return f"Appended to: {path} ({size_str} added)"
            elif mode == "prepend":
                existing = backend.read_text(path) if backend.exists(path) else ""
                backend.write_text(path, content + existing)
                _clear_undo_record(workspace, backend)
                return f"Prepended to: {path} ({size_str} added)"
            else:
                return f"TOOL ERROR (write_file)\nType: InvalidMode\nMessage: Unknown mode '{mode}'."
        except Exception as exc:
            return f"TOOL ERROR (write_file)\nType: BackendError\nMessage: {exc}"

    resolved = _resolve_path_for_tool(path, workspace, "write_file", mounts=mounts)
    if isinstance(resolved, str):
        return resolved

    ro_err = _check_readonly(path, resolved, mounts, "write_file")
    if ro_err:
        return ro_err

    if mode == "create":
        if resolved.exists():
            return (
                f"TOOL ERROR (write_file)\nType: FileExists\n"
                f"Message: File already exists: {path}. Use str_replace to modify, "
                f"or write_file with mode='overwrite', 'append', or 'prepend'."
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        _clear_undo_record(workspace, backend)
        return f"Created: {path} ({_format_size(len(content.encode('utf-8')))})"

    if mode == "overwrite":
        before = _capture_local_snapshot(path, resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        after = _capture_local_snapshot(path, resolved)
        token = _make_undo_token()
        _store_undo_record(workspace, backend, _UndoRecord(
            token=token,
            operation="write_file(overwrite)",
            before=before,
            after=after,
        ))
        verb = "Overwritten" if before.exists else "Created"
        return (f"{verb}: {path} ({_format_size(len(content.encode('utf-8')))}). Undo token: {token}. "
                "Undo only applies to the latest undoable file operation.")

    if mode == "append":
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "a", encoding="utf-8") as f:
            f.write(content)
        _clear_undo_record(workspace, backend)
        return f"Appended to: {path} ({_format_size(len(content.encode('utf-8')))} added)"

    if mode == "prepend":
        resolved.parent.mkdir(parents=True, exist_ok=True)
        existing = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
        resolved.write_text(content + existing, encoding="utf-8")
        _clear_undo_record(workspace, backend)
        return f"Prepended to: {path} ({_format_size(len(content.encode('utf-8')))} added)"

    return f"TOOL ERROR (write_file)\nType: InvalidMode\nMessage: Unknown mode '{mode}'. Use 'create', 'overwrite', 'append', or 'prepend'."


PREVIEW_SNIPPET_CHARS = 100


def _find_all_occurrences(content: str, needle: str) -> list[int]:
    """Return start indices of every non-overlapping occurrence of needle."""
    if not needle:
        return []
    positions: list[int] = []
    start = 0
    while True:
        pos = content.find(needle, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + len(needle)
    return positions


def _line_number_at(content: str, index: int) -> int:
    """Return 1-based line number for a character index."""
    return content.count("\n", 0, index) + 1


def _preview_snippet(text: str, *, max_chars: int = PREVIEW_SNIPPET_CHARS) -> str:
    """Return a single-line preview snippet."""
    one_line = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3] + "..."


def _parse_occurrence_spec(spec: str, total_matches: int) -> tuple[list[int], str]:
    """Parse occurrence selector into sorted 1-based indices and a display label.

    Supported forms:
      (default) / "1"     first match only
      "3"                 single occurrence
      "1-22", "3-99"      inclusive range (clamped to total_matches)
      "3-*"               from occurrence 3 through last
      "*", "all", "1-*"   every match
    """
    if total_matches == 0:
        return [], spec.strip() or "1"

    raw = (spec or "1").strip()
    if not raw:
        raw = "1"

    lower = raw.lower()
    if lower in {"*", "all", "1-*"}:
        return list(range(1, total_matches + 1)), raw

    if lower.endswith("-*"):
        start_part = raw[:-2].strip()
        if not start_part.isdigit():
            raise ValueError(f"Invalid occurrence selector '{spec}'.")
        start = int(start_part)
        if start < 1:
            raise ValueError(f"Occurrence index must be >= 1, got {start}.")
        end = total_matches
        label = f"{start}-*"
        indices = list(range(start, end + 1))
        if not indices:
            raise ValueError(
                f"Occurrence selector '{spec}' matches no occurrences "
                f"(file has {total_matches} match(es))."
            )
        return indices, label

    if "-" in raw:
        start_part, end_part = raw.split("-", 1)
        if not start_part.strip().isdigit() or not end_part.strip().isdigit():
            raise ValueError(f"Invalid occurrence selector '{spec}'.")
        start = int(start_part.strip())
        end = int(end_part.strip())
        if start < 1:
            raise ValueError(f"Occurrence index must be >= 1, got {start}.")
        if start > end:
            raise ValueError(f"Occurrence range start ({start}) > end ({end}).")
        clamped_end = min(end, total_matches)
        if start > total_matches:
            raise ValueError(
                f"Occurrence selector '{spec}' matches no occurrences "
                f"(file has {total_matches} match(es))."
            )
        label = raw if clamped_end == end else f"{start}-{clamped_end}"
        return list(range(start, clamped_end + 1)), label

    if not raw.isdigit():
        raise ValueError(f"Invalid occurrence selector '{spec}'.")

    index = int(raw)
    if index < 1 or index > total_matches:
        raise ValueError(
            f"Occurrence {index} out of range; file has {total_matches} match(es)."
        )
    return [index], raw


def _apply_selected_replacements(
    content: str,
    old_string: str,
    new_string: str,
    indices_1based: list[int],
) -> str:
    """Replace selected 1-based occurrences, processing from end to start."""
    positions = _find_all_occurrences(content, old_string)
    result = content
    for index in sorted(set(indices_1based), reverse=True):
        pos = positions[index - 1]
        result = result[:pos] + new_string + result[pos + len(old_string):]
    return result


def _format_str_replace_success(
    *,
    path: str,
    old_string: str,
    new_string: str,
    selected_indices: list[int],
    total_matches: int,
    occurrence_label: str,
    line_numbers: list[int],
    new_revision: str,
    total_lines: int,
    undo_token: str,
) -> str:
    count = len(selected_indices)
    match_word = "occurrence" if count == 1 else "occurrences"
    if count == total_matches and occurrence_label in {"*", "all", "1-*"}:
        match_summary = f"{count} of {total_matches} {match_word} (occurrence={occurrence_label})"
    elif count == 1 and total_matches == 1:
        match_summary = f"1 of 1 {match_word}"
    else:
        match_summary = (
            f"{count} of {total_matches} {match_word} "
            f"(occurrence={occurrence_label})"
        )

    lines_part = ", ".join(str(n) for n in line_numbers)
    return (
        f"Replaced: {path}\n"
        f"Matches: {match_summary}\n"
        f"Lines: {lines_part}\n"
        f"Preview:\n"
        f"  - {_preview_snippet(old_string)}\n"
        f"  + {_preview_snippet(new_string)}\n"
        f"Revision: {new_revision}\n"
        f"Total lines: {total_lines}\n"
        f"Undo token: {undo_token}"
    )


def _str_replace_core(
    *,
    path: str,
    content: str,
    old_string: str,
    new_string: str,
    expected_revision: str,
    occurrence: str,
    workspace: Path,
    backend: IOBackend | None,
    before_snapshot_factory: Callable[[], _UndoSnapshot],
    write_content: Callable[[str], None],
) -> str:
    if old_string == new_string:
        return (
            "TOOL ERROR (str_replace)\nType: InvalidParam\n"
            "Message: old_string and new_string must differ."
        )
    if not old_string:
        return (
            "TOOL ERROR (str_replace)\nType: InvalidParam\n"
            "Message: old_string must not be empty."
        )

    current_revision = _content_revision(content)
    if expected_revision != current_revision:
        return (
            "TOOL ERROR (str_replace)\n"
            "Type: StaleRead\n"
            f"Message: File changed since last read for {path}. "
            f"Expected revision {expected_revision}, but current revision is {current_revision}. "
            "Call read_file again and retry with the new revision."
        )

    positions = _find_all_occurrences(content, old_string)
    total_matches = len(positions)
    if total_matches == 0:
        return (
            "TOOL ERROR (str_replace)\nType: NotFound\n"
            f"Message: old_string not found in {path}. "
            "Copy the exact literal text from read_file, including whitespace."
        )

    try:
        selected_indices, occurrence_label = _parse_occurrence_spec(occurrence, total_matches)
    except ValueError as exc:
        return f"TOOL ERROR (str_replace)\nType: InvalidOccurrence\nMessage: {exc}"

    line_numbers = [_line_number_at(content, positions[i - 1]) for i in selected_indices]
    before = before_snapshot_factory()
    final_content = _apply_selected_replacements(
        content, old_string, new_string, selected_indices,
    )
    write_content(final_content)
    after = before_snapshot_factory()
    token = _make_undo_token()
    _store_undo_record(workspace, backend, _UndoRecord(
        token=token,
        operation="str_replace",
        before=before,
        after=after,
    ))
    return _format_str_replace_success(
        path=path,
        old_string=old_string,
        new_string=new_string,
        selected_indices=selected_indices,
        total_matches=total_matches,
        occurrence_label=occurrence_label,
        line_numbers=line_numbers,
        new_revision=_content_revision(final_content),
        total_lines=final_content.count("\n") + 1 if final_content else 0,
        undo_token=token,
    )


def str_replace(
    path: str,
    old_string: str,
    new_string: str,
    expected_revision: str,
    workspace: Path,
    *,
    occurrence: str = "1",
    mounts: dict[str, MountSpec] | None = None,
) -> str:
    """Replace exact text matches in a file.

    occurrence selects which matches to replace (1-based, non-overlapping):
      default / "1" = first match only
      "3" = third match only
      "1-22", "3-99" = inclusive ranges (end clamped to available matches)
      "3-*" = from third match through last
      "*", "all", "1-*" = every match
    """
    backend = _get_remote_backend()
    if backend is not None:
        try:
            if not backend.exists(path):
                return (
                    "TOOL ERROR (str_replace)\nType: FileNotFound\n"
                    f"Message: {path} not found. Use write_file to create."
                )
            content = backend.read_text(path)

            def before_snapshot() -> _UndoSnapshot:
                return _capture_backend_snapshot(backend, path)

            def write_content(updated: str) -> None:
                backend.write_text(path, updated)

            return _str_replace_core(
                path=path,
                content=content,
                old_string=old_string,
                new_string=new_string,
                expected_revision=expected_revision,
                occurrence=occurrence,
                workspace=workspace,
                backend=backend,
                before_snapshot_factory=before_snapshot,
                write_content=write_content,
            )
        except Exception as exc:
            return f"TOOL ERROR (str_replace)\nType: BackendError\nMessage: {exc}"

    resolved = _resolve_path_for_tool(path, workspace, "str_replace", mounts=mounts)
    if isinstance(resolved, str):
        return resolved

    ro_err = _check_readonly(path, resolved, mounts, "str_replace")
    if ro_err:
        return ro_err

    if not resolved.exists():
        return (
            "TOOL ERROR (str_replace)\nType: FileNotFound\n"
            f"Message: {path} not found. Use write_file to create."
        )

    if not _is_text_file(resolved):
        return (
            f"TOOL ERROR (str_replace)\nType: BinaryFile\n"
            f"Message: str_replace only works on text files. To modify {resolved.suffix} files, "
            f"use execute_python with an appropriate library."
        )

    content = resolved.read_text(encoding="utf-8")

    def before_snapshot() -> _UndoSnapshot:
        return _capture_local_snapshot(path, resolved)

    def write_content(updated: str) -> None:
        resolved.write_text(updated, encoding="utf-8")

    return _str_replace_core(
        path=path,
        content=content,
        old_string=old_string,
        new_string=new_string,
        expected_revision=expected_revision,
        occurrence=occurrence,
        workspace=workspace,
        backend=backend,
        before_snapshot_factory=before_snapshot,
        write_content=write_content,
    )


def delete_file(path: str, workspace: Path, *, mounts: dict[str, MountSpec] | None = None) -> str:
    """Move file to .recycle_bin or permanently delete if already there."""
    backend = _get_remote_backend()
    if backend is not None:
        try:
            if not backend.exists(path):
                return f"TOOL ERROR (delete_file)\nType: FileNotFound\nMessage: {path} not found."
            if backend.is_dir(path):
                return f"TOOL ERROR (delete_file)\nType: IsDirectory\nMessage: {path} is a directory. Use set_directory with action='delete'."
            before = _capture_backend_snapshot(backend, path)
            in_recycle = path.replace("\\", "/").startswith(".recycle_bin/")
            if in_recycle:
                backend.delete(path)
                after = _snapshot_missing(path)
                token = _make_undo_token()
                _store_undo_record(workspace, backend, _UndoRecord(
                    token=token,
                    operation="delete_file",
                    before=before,
                    after=after,
                ))
                return (f"Permanently deleted: {path}. Undo token: {token}. "
                        "Undo only applies to the latest undoable file operation.")
            recycle_path = f".recycle_bin/{path}"
            if backend.exists(recycle_path):
                recycle_path = f".recycle_bin/{Path(path).stem}.{int(time.time())}{Path(path).suffix}"
            backend.move(path, recycle_path)
            after = _capture_backend_snapshot(backend, recycle_path)
            token = _make_undo_token()
            _store_undo_record(workspace, backend, _UndoRecord(
                token=token,
                operation="delete_file",
                before=before,
                after=after,
            ))
            return (f"Moved to recycle bin: {recycle_path}. Undo token: {token}. "
                    "Undo only applies to the latest undoable file operation.")
        except Exception as exc:
            return f"TOOL ERROR (delete_file)\nType: BackendError\nMessage: {exc}"

    resolved = _resolve_path_for_tool(path, workspace, "delete_file", mounts=mounts)
    if isinstance(resolved, str):
        return resolved

    ro_err = _check_readonly(path, resolved, mounts, "delete_file")
    if ro_err:
        return ro_err

    if not resolved.exists():
        return f"TOOL ERROR (delete_file)\nType: FileNotFound\nMessage: {path} not found."
    if resolved.is_dir():
        return f"TOOL ERROR (delete_file)\nType: IsDirectory\nMessage: {path} is a directory. Use set_directory with action='delete'."

    before = _capture_local_snapshot(path, resolved)
    recycle = workspace / ".recycle_bin"
    in_recycle = str(resolved).startswith(str(recycle.resolve()))

    if in_recycle:
        resolved.unlink()
        after = _snapshot_missing(path)
        token = _make_undo_token()
        _store_undo_record(workspace, backend, _UndoRecord(
            token=token,
            operation="delete_file",
            before=before,
            after=after,
        ))
        return (f"Permanently deleted: {path}. Undo token: {token}. "
                "Undo only applies to the latest undoable file operation.")

    try:
        rel = resolved.relative_to(workspace)
    except ValueError:
        rel = Path(path)
    dest = recycle / rel
    if dest.exists():
        dest = dest.with_name(f"{dest.name}.{int(time.time())}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(resolved), str(dest))
    recycle_path = f".recycle_bin/{rel.as_posix()}"
    after = _capture_local_snapshot(recycle_path, dest)
    token = _make_undo_token()
    _store_undo_record(workspace, backend, _UndoRecord(
        token=token,
        operation="delete_file",
        before=before,
        after=after,
    ))
    return (f"Moved to recycle bin: {recycle_path}. Undo token: {token}. "
            "Undo only applies to the latest undoable file operation.")


def move_file(source: str, destination: str, workspace: Path, *, mounts: dict[str, MountSpec] | None = None) -> str:
    """Move/rename a file within the workspace."""
    backend = _get_remote_backend()
    if backend is not None:
        try:
            if not backend.exists(source):
                return f"TOOL ERROR (move_file)\nType: FileNotFound\nMessage: Source not found: {source}"
            if backend.is_dir(source):
                return f"TOOL ERROR (move_file)\nType: IsDirectory\nMessage: {source} is a directory. Use set_directory with action='move'."
            dst = destination
            if backend.exists(dst) and backend.is_dir(dst):
                dst = f"{dst.rstrip('/')}/{Path(source).name}"
            if backend.exists(dst):
                return f"TOOL ERROR (move_file)\nType: DestinationExists\nMessage: Destination already exists: {destination}"
            before = _capture_backend_snapshot(backend, source)
            backend.move(source, dst)
            after = _capture_backend_snapshot(backend, dst)
            token = _make_undo_token()
            _store_undo_record(workspace, backend, _UndoRecord(
                token=token,
                operation="move_file",
                before=before,
                after=after,
            ))
            return (f"Moved: {source} -> {destination}. Undo token: {token}. "
                    "Undo only applies to the latest undoable file operation.")
        except Exception as exc:
            return f"TOOL ERROR (move_file)\nType: BackendError\nMessage: {exc}"

    src = _resolve_path_for_tool(source, workspace, "move_file", mounts=mounts)
    if isinstance(src, str):
        return src
    if not src.exists():
        return f"TOOL ERROR (move_file)\nType: FileNotFound\nMessage: Source not found: {source}"
    if src.is_dir():
        return f"TOOL ERROR (move_file)\nType: IsDirectory\nMessage: {source} is a directory. Use set_directory with action='move'."

    ro_err = _check_readonly(source, src, mounts, "move_file")
    if ro_err:
        return ro_err

    dst = _resolve_path_for_tool(destination, workspace, "move_file", mounts=mounts)
    if isinstance(dst, str):
        return dst

    ro_err = _check_readonly(destination, dst, mounts, "move_file")
    if ro_err:
        return ro_err
    destination_is_dir = dst.is_dir()
    if destination_is_dir:
        dst = dst / src.name
    if dst.exists():
        return f"TOOL ERROR (move_file)\nType: DestinationExists\nMessage: Destination already exists: {destination}"

    before = _capture_local_snapshot(source, src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    after_path = f"{destination.rstrip('/\\')}/{src.name}" if destination_is_dir else destination
    after = _capture_local_snapshot(after_path, dst)
    token = _make_undo_token()
    _store_undo_record(workspace, backend, _UndoRecord(
        token=token,
        operation="move_file",
        before=before,
        after=after,
    ))
    return (f"Moved: {source} -> {destination}. Undo token: {token}. "
            "Undo only applies to the latest undoable file operation.")


def undo_file_operation(
    undo_token: str,
    workspace: Path,
    *,
    mounts: dict[str, MountSpec] | None = None,
) -> str:
    """Undo the most recent undoable file operation if the token still matches."""
    backend = _get_remote_backend()
    record = _peek_undo_record(workspace, backend)
    if record is None:
        return ("TOOL ERROR (undo_file_operation)\nType: NoUndoAvailable\n"
                "Message: There is no undoable file operation available.")
    if undo_token != record.token:
        return ("TOOL ERROR (undo_file_operation)\nType: InvalidUndoToken\n"
                "Message: Undo token is invalid or expired. Only the latest undoable file operation can be undone.")

    conflict = _validate_current_snapshot(record.after, workspace=workspace, backend=backend, mounts=mounts)
    if conflict:
        return conflict
    if record.before.path != record.after.path and _path_exists(
        record.before.path, workspace=workspace, backend=backend, mounts=mounts,
    ):
        return ("TOOL ERROR (undo_file_operation)\nType: UndoConflict\n"
                f"Message: Cannot undo because {record.before.path} now exists again.")

    try:
        if record.after.exists and (record.after.path != record.before.path or not record.before.exists):
            _delete_file_at_path(record.after.path, workspace=workspace, backend=backend, mounts=mounts)

        if record.before.exists:
            _write_bytes_to_path(
                record.before.path,
                record.before.data or b"",
                workspace=workspace,
                backend=backend,
                mounts=mounts,
            )
        else:
            _delete_file_at_path(record.before.path, workspace=workspace, backend=backend, mounts=mounts)
    except Exception as exc:
        return f"TOOL ERROR (undo_file_operation)\nType: UndoFailed\nMessage: {exc}"

    _pop_undo_record(workspace, backend)
    return (f"Undid {record.operation}. Restored {record.before.path}. "
            "The undo token has been consumed.")


def _parse_ignore_patterns(raw: str) -> list[str]:
    """Split comma-or-newline-separated ignore string into individual patterns."""
    if not raw or not raw.strip():
        return []
    patterns = []
    for token in raw.replace(",", "\n").split("\n"):
        token = token.strip()
        if token and not token.startswith("#"):
            patterns.append(token)
    return patterns


def _is_ignored(rel_posix: str, name: str, is_dir: bool, patterns: list[str]) -> bool:
    """Check if a path matches any gitignore-style pattern.

    Supports: *.ext, dirname/, **/pattern, specific/path, plain name.
    """
    for pat in patterns:
        dir_only = pat.endswith("/")
        p = pat.rstrip("/")

        if dir_only and not is_dir:
            continue

        if fnmatch.fnmatch(name, p):
            return True
        if fnmatch.fnmatch(rel_posix, p):
            return True
        if "/" not in p and fnmatch.fnmatch(rel_posix, f"**/{p}"):
            return True
        if fnmatch.fnmatch(rel_posix, f"**/{p}"):
            return True

    return False


def list_files(
    path: str,
    workspace: Path,
    *,
    depth: int = 1,
    ignore: str = "",
    mounts: dict[str, MountSpec] | None = None,
) -> str:
    """List directory contents up to ``depth`` levels, optionally filtering with gitignore-style patterns."""
    if depth < 1:
        return (
            f"TOOL ERROR (list_files)\nType: InvalidParam\n"
            f"Message: depth must be >= 1, got {depth}."
        )
    backend = _get_remote_backend()
    if backend is not None:
        try:
            if not backend.exists(path):
                return f"TOOL ERROR (list_files)\nType: PathNotFound\nMessage: {path} not found."
            if not backend.is_dir(path):
                return f"TOOL ERROR (list_files)\nType: NotADirectory\nMessage: {path} is not a directory."
            patterns = _parse_ignore_patterns(ignore)
            entries = []
            recycle_name = ".recycle_bin"
            if depth > 1:
                base_parts = [] if path == "." else path.split("/")
                for dirpath, dirs, files in backend.walk(path):
                    dir_parts = [] if dirpath == "." else dirpath.split("/")
                    dir_depth = len(dir_parts) - len(base_parts)
                    dirs[:] = [d for d in dirs if d != recycle_name]
                    if patterns:
                        dirs[:] = [d for d in dirs if not _is_ignored(f"{dirpath}/{d}" if dirpath != "." else d, d, True, patterns)]
                    for d in sorted(dirs):
                        entries.append(f"[dir]  {dirpath}/{d}/" if dirpath != "." else f"[dir]  {d}/")
                    for f in sorted(files):
                        rel = f"{dirpath}/{f}" if dirpath != "." else f
                        if patterns and _is_ignored(rel, f, False, patterns):
                            continue
                        entries.append(f"[file] {rel}")
                    dirs[:] = [d for d in dirs if dir_depth + 1 < depth]
            else:
                for entry in backend.list_dir(path):
                    if entry.name == recycle_name:
                        continue
                    rel = f"{path}/{entry.name}" if path != "." else entry.name
                    if patterns and _is_ignored(rel, entry.name, entry.is_dir, patterns):
                        continue
                    if entry.is_dir:
                        entries.append(f"[dir]  {rel}/")
                    else:
                        mod = datetime.fromtimestamp(entry.mtime, tz=timezone.utc).strftime("%Y-%m-%d") if entry.mtime else ""
                        size = _format_size(entry.size) if entry.size else ""
                        detail = f"  ({size}, {mod})" if size and mod else ""
                        entries.append(f"[file] {rel}{detail}")
            if not entries:
                return f"Directory is empty: {path}"
            return "\n".join(entries)
        except Exception as exc:
            return f"TOOL ERROR (list_files)\nType: BackendError\nMessage: {exc}"

    resolved = _resolve_path_for_tool(path, workspace, "list_files", mounts=mounts)
    if isinstance(resolved, str):
        return resolved
    if not resolved.exists():
        root = workspace.resolve()
        return (
            f"TOOL ERROR (list_files)\nType: PathNotFound\n"
            f"Message: {path!r} not found under workspace root {root}. "
            f'Use path="." to list the workspace root.'
        )
    if not resolved.is_dir():
        return f"TOOL ERROR (list_files)\nType: NotADirectory\nMessage: {path} is not a directory."

    patterns = _parse_ignore_patterns(ignore)
    entries = []
    recycle_name = ".recycle_bin"

    def _posix(p: Path) -> str:
        return p.as_posix()

    def _rel_to_base(item_path: Path) -> Path:
        """Compute workspace-relative path. For mount-resolved dirs, use
        the user-provided path as prefix instead of workspace.relative_to."""
        try:
            return item_path.relative_to(workspace)
        except ValueError:
            return Path(path) / item_path.relative_to(resolved)

    if depth > 1:
        for root, dirs, files in os.walk(resolved):
            dirs[:] = [d for d in dirs if d != recycle_name]
            root_path = Path(root)
            rel_root = _rel_to_base(root_path)
            dir_depth = len(rel_root.parts)

            if patterns:
                dirs[:] = [
                    d for d in dirs
                    if not _is_ignored(_posix(rel_root / d), d, True, patterns)
                ]

            for d in sorted(dirs):
                entries.append(f"[dir]  {_posix(rel_root / d)}/")
            for f in sorted(files):
                rel = _posix(rel_root / f)
                if patterns and _is_ignored(rel, f, False, patterns):
                    continue
                fp = root_path / f
                try:
                    stat = fp.stat()
                    mod = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
                    entries.append(f"[file] {rel}  ({_format_size(stat.st_size)}, {mod})")
                except OSError:
                    entries.append(f"[file] {rel}")
            dirs[:] = [d for d in dirs if dir_depth + 1 < depth]
    else:
        for item in sorted(resolved.iterdir()):
            if item.name == recycle_name:
                continue
            rel = _rel_to_base(item)
            rel_posix = _posix(rel)
            is_dir = item.is_dir()

            if patterns and _is_ignored(rel_posix, item.name, is_dir, patterns):
                continue

            if is_dir:
                entries.append(f"[dir]  {rel_posix}/")
            else:
                try:
                    stat = item.stat()
                    mod = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
                    entries.append(f"[file] {rel_posix}  ({_format_size(stat.st_size)}, {mod})")
                except OSError:
                    entries.append(f"[file] {rel_posix}")

    if not entries:
        return f"Directory is empty: {path}"
    return "\n".join(entries)


def set_directory(
    action: str,
    path: str,
    workspace: Path,
    *,
    new_name: str = "",
    destination: str = "",
    mounts: dict[str, MountSpec] | None = None,
) -> str:
    """Directory operations: create, rename, delete, move."""
    backend = _get_remote_backend()
    if backend is not None:
        try:
            if action == "create":
                backend.mkdir(path)
                return f"Directory created: {path}"
            if not backend.exists(path) or not backend.is_dir(path):
                return f"TOOL ERROR (set_directory)\nType: NotFound\nMessage: Directory not found: {path}"
            if action == "rename":
                if not new_name:
                    return f"TOOL ERROR (set_directory)\nType: MissingParam\nMessage: new_name is required for rename."
                parent = str(Path(path).parent).replace("\\", "/")
                target = f"{parent}/{new_name}" if parent != "." else new_name
                if backend.exists(target):
                    return f"TOOL ERROR (set_directory)\nType: AlreadyExists\nMessage: {new_name} already exists."
                backend.move(path, target)
                return f"Renamed: {path} -> {new_name}"
            if action == "delete":
                in_recycle = path.replace("\\", "/").startswith(".recycle_bin/")
                if in_recycle:
                    backend.delete_tree(path)
                    return f"Permanently deleted directory: {path}"
                recycle_path = f".recycle_bin/{path}"
                backend.move(path, recycle_path)
                return f"Moved directory to recycle bin: {recycle_path}"
            if action == "move":
                if not destination:
                    return f"TOOL ERROR (set_directory)\nType: MissingParam\nMessage: destination is required for move."
                if backend.exists(destination):
                    return f"TOOL ERROR (set_directory)\nType: AlreadyExists\nMessage: Destination already exists: {destination}"
                backend.move(path, destination)
                return f"Moved directory: {path} -> {destination}"
            return f"TOOL ERROR (set_directory)\nType: InvalidAction\nMessage: Unknown action '{action}'."
        except Exception as exc:
            return f"TOOL ERROR (set_directory)\nType: BackendError\nMessage: {exc}"

    resolved = _resolve_path_for_tool(path, workspace, "set_directory", mounts=mounts)
    if isinstance(resolved, str):
        return resolved

    if action != "create":
        ro_err = _check_readonly(path, resolved, mounts, "set_directory")
        if ro_err:
            return ro_err

    if action == "create":
        resolved.mkdir(parents=True, exist_ok=True)
        return f"Directory created: {path}"

    if action == "rename":
        if not resolved.exists() or not resolved.is_dir():
            return f"TOOL ERROR (set_directory)\nType: NotFound\nMessage: Directory not found: {path}"
        if not new_name:
            return f"TOOL ERROR (set_directory)\nType: MissingParam\nMessage: new_name is required for rename."
        target = resolved.parent / new_name
        if target.exists():
            return f"TOOL ERROR (set_directory)\nType: AlreadyExists\nMessage: {new_name} already exists in {resolved.parent}."
        resolved.rename(target)
        return f"Renamed: {path} -> {new_name}"

    if action == "delete":
        if not resolved.exists() or not resolved.is_dir():
            return f"TOOL ERROR (set_directory)\nType: NotFound\nMessage: Directory not found: {path}"
        recycle = workspace / ".recycle_bin"
        in_recycle = str(resolved).startswith(str(recycle.resolve()))
        if in_recycle:
            shutil.rmtree(resolved)
            return f"Permanently deleted directory: {path}"
        try:
            rel = resolved.relative_to(workspace)
        except ValueError:
            rel = Path(path)
        dest = recycle / rel
        if dest.exists():
            dest = dest.with_name(f"{dest.name}.{int(time.time())}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(dest))
        return f"Moved directory to recycle bin: .recycle_bin/{rel}"

    if action == "move":
        if not resolved.exists() or not resolved.is_dir():
            return f"TOOL ERROR (set_directory)\nType: NotFound\nMessage: Directory not found: {path}"
        if not destination:
            return f"TOOL ERROR (set_directory)\nType: MissingParam\nMessage: destination is required for move."
        dst = _resolve_path_for_tool(destination, workspace, "set_directory", mounts=mounts)
        if isinstance(dst, str):
            return dst
        ro_err_dst = _check_readonly(destination, dst, mounts, "set_directory")
        if ro_err_dst:
            return ro_err_dst
        if dst.exists():
            return f"TOOL ERROR (set_directory)\nType: AlreadyExists\nMessage: Destination already exists: {destination}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(dst))
        return f"Moved directory: {path} -> {destination}"

    return f"TOOL ERROR (set_directory)\nType: InvalidAction\nMessage: Unknown action '{action}'. Use 'create', 'rename', 'delete', or 'move'."

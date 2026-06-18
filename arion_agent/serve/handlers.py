"""Handler functions for file I/O, shell execution, and checkpoint endpoints.

Pure functions with explicit parameters -- no FastAPI dependency.
Return dicts (or raise exceptions) that the router layer maps to responses.

Three endpoint groups:
  /io/*         - file CRUD (serves RemoteIOBackend clients)
  /shell/*      - command execution (serves RemoteShellBackend clients)
  /checkpoint/* - checkpoint CRUD (serves AsyncProxySaver clients)

All file operations use short-lived open/close to avoid stale fd issues
on grpcfuse bind mounts.
"""

from __future__ import annotations

import asyncio
import base64
import json
import locale
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Path confinement
# ---------------------------------------------------------------------------

def _confine(root: Path, user_path: str) -> Path:
    """Normalize user_path within root without following symlinks.

    Uses os.path.normpath to collapse '..' traversal while preserving
    symlinks (mount points like imported_directories/* resolve outside
    the workspace via symlink, which is intentional).
    """
    clean = user_path.strip()
    if clean.startswith("/") or clean.startswith("\\"):
        clean = clean.lstrip("/\\")
    target = Path(os.path.normpath(root / clean))
    root_normed = Path(os.path.normpath(root))
    try:
        target.relative_to(root_normed)
    except ValueError:
        raise ValueError(f"Path escapes workspace: {user_path}") from None
    return target


# ---------------------------------------------------------------------------
# File I/O handlers
# ---------------------------------------------------------------------------

def handle_io_read(root: Path, path: str, mode: str = "text") -> dict:
    target = _confine(root, path)
    if not target.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if mode == "text":
        return {"text": target.read_text(encoding="utf-8")}
    data = target.read_bytes()
    return {"data": base64.b64encode(data).decode("ascii")}


def handle_io_write(root: Path, path: str, *, text: str | None = None,
                    data: str | None = None) -> dict:
    """Write to file. Provide text (str) or data (base64-encoded bytes)."""
    target = _confine(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        if text is not None:
            tmp.write_text(text, encoding="utf-8")
        elif data is not None:
            tmp.write_bytes(base64.b64decode(data))
        else:
            raise ValueError("Either text or data must be provided")
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"ok": True}


def handle_io_append(root: Path, path: str, *, text: str | None = None,
                     data: str | None = None) -> dict:
    target = _confine(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if text is not None:
        with open(target, "a", encoding="utf-8") as f:
            f.write(text)
    elif data is not None:
        with open(target, "ab") as f:
            f.write(base64.b64decode(data))
    else:
        raise ValueError("Either text or data must be provided")
    return {"ok": True}


def handle_io_exists(root: Path, path: str) -> dict:
    target = _confine(root, path)
    return {"exists": target.exists()}


def handle_io_stat(root: Path, path: str) -> dict:
    target = _confine(root, path)
    if not target.exists():
        raise FileNotFoundError(f"Not found: {path}")
    s = target.stat()
    return {
        "size": s.st_size,
        "mtime": s.st_mtime,
        "is_dir": target.is_dir(),
    }


def handle_io_list(root: Path, path: str) -> dict:
    target = _confine(root, path)
    if not target.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if not target.is_dir():
        raise ValueError(f"Not a directory: {path}")
    entries = []
    for item in sorted(target.iterdir()):
        try:
            s = item.stat()
            entries.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": s.st_size,
                "mtime": s.st_mtime,
            })
        except OSError:
            entries.append({"name": item.name, "is_dir": item.is_dir(),
                           "size": 0, "mtime": 0})
    return {"entries": entries}


def handle_io_mkdir(root: Path, path: str, parents: bool = True) -> dict:
    target = _confine(root, path)
    target.mkdir(parents=parents, exist_ok=True)
    return {"ok": True}


def handle_io_delete(root: Path, path: str, tree: bool = False) -> dict:
    target = _confine(root, path)
    if not target.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if tree:
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"ok": True}


def handle_io_move(root: Path, src: str, dst: str) -> dict:
    src_path = _confine(root, src)
    dst_path = _confine(root, dst)
    if not src_path.exists():
        raise FileNotFoundError(f"Not found: {src}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_path), str(dst_path))
    return {"ok": True}


def handle_io_glob(root: Path, path: str, pattern: str) -> dict:
    base = _confine(root, path)
    if not base.exists():
        raise FileNotFoundError(f"Not found: {path}")
    root_resolved = root.resolve()
    matches = []
    for m in base.rglob(pattern):
        try:
            rel = str(m.relative_to(root_resolved)).replace("\\", "/")
            matches.append(rel)
        except ValueError:
            matches.append(str(m))
    return {"matches": matches}


def handle_io_walk(root: Path, path: str) -> dict:
    base = _confine(root, path)
    if not base.exists():
        raise FileNotFoundError(f"Not found: {path}")
    root_resolved = root.resolve()
    entries = []
    for dirpath, dirnames, filenames in os.walk(base):
        try:
            rel = str(Path(dirpath).relative_to(root_resolved)).replace("\\", "/")
        except ValueError:
            rel = dirpath
        entries.append({
            "dirpath": rel,
            "dirnames": sorted(dirnames),
            "filenames": sorted(filenames),
        })
    return {"entries": entries}


# ---------------------------------------------------------------------------
# Shell execution handlers
# ---------------------------------------------------------------------------

def _decode_output(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        enc = locale.getpreferredencoding(False)
        return data.decode(enc, errors="replace")


async def handle_shell_exec(
    workspace_root: Path,
    command: list[str],
    cwd: str = ".",
    timeout: float = 120.0,
    max_output_bytes: int = 200_000,
    env: dict[str, str] | None = None,
) -> dict:
    """Execute a shell command confined to workspace_root.

    Returns: {"output": str, "exit_code": int, "truncated": bool}
    """
    resolved_cwd = Path(os.path.normpath(workspace_root / cwd))
    try:
        resolved_cwd.relative_to(Path(os.path.normpath(workspace_root)))
    except ValueError:
        raise ValueError(f"cwd escapes workspace: {cwd}") from None

    if not resolved_cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")

    if not command:
        raise ValueError("command is required")

    effective_env = os.environ.copy()
    if env:
        effective_env.update(env)

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved_cwd),
            env=effective_env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"output": f"Command timed out after {timeout:.0f}s.",
                    "exit_code": 124, "truncated": False}

        stdout_text = _decode_output(stdout_bytes)
        stderr_text = _decode_output(stderr_bytes)

        parts = []
        if stdout_text:
            parts.append(stdout_text)
        if stderr_text:
            for line in stderr_text.strip().split("\n"):
                parts.append(f"[stderr] {line}")

        output = "\n".join(parts) if parts else "<no output>"

        truncated = False
        if len(output) > max_output_bytes:
            output = output[:max_output_bytes] + f"\n\n... Output truncated at {max_output_bytes} bytes."
            truncated = True

        exit_code = proc.returncode or 0
        if exit_code != 0:
            output = f"{output.rstrip()}\n\nExit code: {exit_code}"

        return {"output": output, "exit_code": exit_code, "truncated": truncated}

    except FileNotFoundError:
        return {"output": f"Command not found: {command[0]}",
                "exit_code": 127, "truncated": False}
    except Exception as exc:
        return {"output": f"Shell execution error: {exc}",
                "exit_code": 1, "truncated": False}


# ---------------------------------------------------------------------------
# Checkpoint handlers
# ---------------------------------------------------------------------------

_CHECKPOINTS_DDL = """\
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id            TEXT NOT NULL,
    checkpoint_ns        TEXT NOT NULL DEFAULT '',
    checkpoint_id        TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type                 TEXT,
    checkpoint           BLOB,
    metadata             BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);"""

_WRITES_DDL = """\
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id     TEXT    NOT NULL,
    checkpoint_ns TEXT    NOT NULL DEFAULT '',
    checkpoint_id TEXT    NOT NULL,
    task_id       TEXT    NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT    NOT NULL,
    type          TEXT,
    blob          BLOB    NOT NULL,
    task_path     TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);"""


def _cp_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _blob_to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("latin-1")
    return b""


def _encode_blob(value: Any) -> str:
    raw = _blob_to_bytes(value)
    return base64.b64encode(raw).decode("ascii") if raw else ""


def _decode_metadata(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def handle_checkpoint_setup(db_path: str) -> dict:
    conn = _cp_connect(db_path)
    try:
        conn.execute(_CHECKPOINTS_DDL)
        conn.execute(_WRITES_DDL)
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def handle_checkpoint_get_tuple(
    db_path: str, thread_id: str, checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> dict:
    conn = _cp_connect(db_path)
    try:
        if checkpoint_id:
            row = conn.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, "
                "parent_checkpoint_id, type, checkpoint, metadata "
                "FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? "
                "AND checkpoint_id = ?",
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, "
                "parent_checkpoint_id, type, checkpoint, metadata "
                "FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id, checkpoint_ns),
            ).fetchone()

        if row is None:
            return {"not_found": True}

        cp_id = row["checkpoint_id"]
        writes_rows = conn.execute(
            "SELECT task_id, channel, type, blob, idx "
            "FROM checkpoint_writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "AND checkpoint_id = ? ORDER BY idx",
            (thread_id, checkpoint_ns, cp_id),
        ).fetchall()

        writes = [
            {
                "task_id": w["task_id"],
                "channel": w["channel"],
                "type": w["type"],
                "value": _encode_blob(w["blob"]),
                "idx": w["idx"],
            }
            for w in writes_rows
        ]

        return {
            "thread_id": row["thread_id"],
            "checkpoint_ns": row["checkpoint_ns"],
            "checkpoint_id": row["checkpoint_id"],
            "parent_checkpoint_id": row["parent_checkpoint_id"],
            "type": row["type"],
            "checkpoint": _encode_blob(row["checkpoint"]),
            "metadata": _decode_metadata(row["metadata"]),
            "writes": writes,
        }
    finally:
        conn.close()


def handle_checkpoint_put(
    db_path: str, thread_id: str, checkpoint_ns: str,
    checkpoint_id: str, parent_checkpoint_id: str | None,
    type: str, checkpoint: str, metadata: str,
) -> dict:
    conn = _cp_connect(db_path)
    try:
        cp_blob = base64.b64decode(checkpoint)
        metadata_blob = metadata.encode("utf-8") if metadata else b""
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, "
            "parent_checkpoint_id, type, checkpoint, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thread_id, checkpoint_ns, checkpoint_id,
             parent_checkpoint_id, type, cp_blob, metadata_blob),
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def handle_checkpoint_put_writes(
    db_path: str, thread_id: str, checkpoint_ns: str,
    checkpoint_id: str, task_id: str,
    writes: list[dict],
) -> dict:
    conn = _cp_connect(db_path)
    try:
        for w in writes:
            blob = base64.b64decode(w["value"])
            conn.execute(
                "INSERT OR REPLACE INTO checkpoint_writes "
                "(thread_id, checkpoint_ns, checkpoint_id, "
                "task_id, idx, channel, type, blob) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, checkpoint_ns, checkpoint_id,
                 task_id, w["idx"], w["channel"], w["type"], blob),
            )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def handle_checkpoint_list(
    db_path: str, thread_id: str | None = None,
    checkpoint_ns: str = "", before_id: str | None = None,
    limit: int | None = None, filter: dict | None = None,
) -> dict:
    conn = _cp_connect(db_path)
    try:
        query = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, "
            "parent_checkpoint_id, type, checkpoint, metadata "
            "FROM checkpoints"
        )
        params: list[Any] = []
        conditions: list[str] = []

        if thread_id is not None:
            conditions.append("thread_id = ?")
            params.append(thread_id)

        conditions.append("checkpoint_ns = ?")
        params.append(checkpoint_ns)

        if before_id is not None:
            conditions.append("checkpoint_id < ?")
            params.append(before_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY checkpoint_id DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()

        checkpoints = [
            {
                "thread_id": row["thread_id"],
                "checkpoint_ns": row["checkpoint_ns"],
                "checkpoint_id": row["checkpoint_id"],
                "parent_checkpoint_id": row["parent_checkpoint_id"],
                "type": row["type"],
                "checkpoint": _encode_blob(row["checkpoint"]),
                "metadata": _decode_metadata(row["metadata"]),
            }
            for row in rows
        ]

        return {"checkpoints": checkpoints}
    finally:
        conn.close()

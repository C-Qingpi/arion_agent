"""Persistent session management via SQLite checkpointer.

Provides a default SQLite-backed checkpointer so that:
  - Agent state is persisted across process restarts
  - Any create_arion_agent call can resume a previous session by thread_id
  - Compression node can evict messages knowing state is recoverable
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = Path.home() / ".arion_agent"
DEFAULT_DB_NAME = "checkpoints.sqlite"


def get_default_db_path() -> Path:
    override = os.environ.get("ARION_SESSION_DB")
    if override:
        return Path(override)
    return DEFAULT_DB_DIR / DEFAULT_DB_NAME


async def create_checkpointer(
    db_path: str | Path | None = None,
    *,
    docker_safe: bool | None = None,
) -> AsyncSqliteSaver:
    """Create a persistent SQLite checkpointer.

    Args:
        db_path: Path to the SQLite database file. Defaults to ~/.arion_agent/checkpoints.sqlite.
        docker_safe: Use unix-none VFS and safe pragmas for Docker bind mounts.
            None (default) auto-detects via is_container().

    Returns an AsyncSqliteSaver. Caller is responsible for closing.
    """
    import aiosqlite
    from arion_agent.util.runtime import is_container

    path = Path(db_path) if db_path else get_default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if docker_safe is None:
        docker_safe = is_container()

    if docker_safe:
        uri = f"file:{path}?vfs=unix-none"
        conn = await aiosqlite.connect(uri, uri=True)
        await conn.execute("PRAGMA journal_mode=DELETE")
        await conn.execute("PRAGMA mmap_size=0")
        logger.info("Checkpointer using docker-safe VFS (unix-none): %s", path)
    else:
        conn = await aiosqlite.connect(str(path))

    saver = AsyncSqliteSaver(conn, serde=JsonPlusSerializer())
    await saver.setup()
    return saver

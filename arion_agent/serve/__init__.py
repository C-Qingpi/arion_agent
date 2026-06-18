"""Prefab FastAPI router for file I/O, shell execution, and checkpoint endpoints.

Shipped with arion_agent. Mount in your FastAPI app or run standalone.
The code is always present; the [remote-service] pip extra controls
whether fastapi/uvicorn runtime dependencies are installed.

Usage 1 - mount as-is:
    from arion_agent.serve import create_service_router
    router = create_service_router("/path/to/workspace", "/path/to/db")
    app.include_router(router)

Usage 2 - run standalone:
    python -m arion_agent.serve --root /path/to/workspace --port 8911

Usage 3 - import individual handlers for custom wiring:
    from arion_agent.serve.handlers import handle_io_read, handle_shell_exec
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from arion_agent.serve.handlers import (
    handle_checkpoint_get_tuple,
    handle_checkpoint_list,
    handle_checkpoint_put,
    handle_checkpoint_put_writes,
    handle_checkpoint_setup,
    handle_io_append,
    handle_io_delete,
    handle_io_exists,
    handle_io_glob,
    handle_io_list,
    handle_io_mkdir,
    handle_io_move,
    handle_io_read,
    handle_io_stat,
    handle_io_walk,
    handle_io_write,
    handle_shell_exec,
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class WriteRequest(BaseModel):
    path: str
    text: str | None = None
    data: str | None = None

class AppendRequest(BaseModel):
    path: str
    text: str | None = None
    data: str | None = None

class MkdirRequest(BaseModel):
    path: str
    parents: bool = True

class DeleteRequest(BaseModel):
    path: str
    tree: bool = False

class MoveRequest(BaseModel):
    src: str
    dst: str

class ShellExecRequest(BaseModel):
    command: list[str]
    cwd: str = "."
    timeout: float = 120.0
    max_output_bytes: int = 200_000
    env: dict[str, str] | None = None

class CheckpointGetTupleRequest(BaseModel):
    thread_id: str
    checkpoint_ns: str = ""
    checkpoint_id: str | None = None

class CheckpointPutRequest(BaseModel):
    thread_id: str
    checkpoint_ns: str = ""
    checkpoint_id: str
    parent_checkpoint_id: str | None = None
    type: str
    checkpoint: str
    metadata: str

class CheckpointWriteItem(BaseModel):
    idx: int
    channel: str
    type: str
    value: str

class CheckpointPutWritesRequest(BaseModel):
    thread_id: str
    checkpoint_ns: str = ""
    checkpoint_id: str
    task_id: str
    writes: list[CheckpointWriteItem]

class CheckpointListRequest(BaseModel):
    thread_id: str | None = None
    checkpoint_ns: str = ""
    before_id: str | None = None
    limit: int | None = None
    filter: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def _wrap_errors(func, *args, **kwargs):
    """Call handler, map exceptions to HTTP errors."""
    try:
        return func(*args, **kwargs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_service_router(
    workspace_root: str | Path,
    checkpoint_db: str | Path | None = None,
) -> APIRouter:
    """Create a FastAPI router with /io/*, /shell/*, and /checkpoint/* endpoints.

    Args:
        workspace_root: Root directory for file I/O and shell operations.
        checkpoint_db: Path to SQLite checkpoint database. If None,
            checkpoint endpoints are not mounted.

    Returns:
        APIRouter ready to include in a FastAPI app.
    """
    root = Path(workspace_root).resolve()
    router = APIRouter()

    # ---- File I/O routes ----

    @router.get("/io/read")
    async def io_read(path: str = Query(...), mode: str = Query("text")):
        return _wrap_errors(handle_io_read, root, path, mode)

    @router.post("/io/write")
    async def io_write(req: WriteRequest):
        return _wrap_errors(handle_io_write, root, req.path,
                           text=req.text, data=req.data)

    @router.post("/io/append")
    async def io_append(req: AppendRequest):
        return _wrap_errors(handle_io_append, root, req.path,
                           text=req.text, data=req.data)

    @router.get("/io/exists")
    async def io_exists(path: str = Query(...)):
        return _wrap_errors(handle_io_exists, root, path)

    @router.get("/io/stat")
    async def io_stat(path: str = Query(...)):
        return _wrap_errors(handle_io_stat, root, path)

    @router.get("/io/list")
    async def io_list(path: str = Query(...)):
        return _wrap_errors(handle_io_list, root, path)

    @router.post("/io/mkdir")
    async def io_mkdir(req: MkdirRequest):
        return _wrap_errors(handle_io_mkdir, root, req.path, req.parents)

    @router.post("/io/delete")
    async def io_delete(req: DeleteRequest):
        return _wrap_errors(handle_io_delete, root, req.path, req.tree)

    @router.post("/io/move")
    async def io_move(req: MoveRequest):
        return _wrap_errors(handle_io_move, root, req.src, req.dst)

    @router.get("/io/glob")
    async def io_glob(path: str = Query(...), pattern: str = Query(...)):
        return _wrap_errors(handle_io_glob, root, path, pattern)

    @router.get("/io/walk")
    async def io_walk(path: str = Query(...)):
        return _wrap_errors(handle_io_walk, root, path)

    # ---- Shell execution routes ----

    @router.post("/shell/exec")
    async def shell_exec(req: ShellExecRequest):
        try:
            return await handle_shell_exec(
                root, req.command, req.cwd, req.timeout,
                req.max_output_bytes, req.env,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---- Checkpoint routes ----

    if checkpoint_db is not None:
        db = str(checkpoint_db)
        handle_checkpoint_setup(db)

        @router.post("/checkpoint/get_tuple")
        async def cp_get_tuple(req: CheckpointGetTupleRequest):
            return _wrap_errors(handle_checkpoint_get_tuple, db,
                               req.thread_id, req.checkpoint_ns,
                               req.checkpoint_id)

        @router.post("/checkpoint/put")
        async def cp_put(req: CheckpointPutRequest):
            return _wrap_errors(handle_checkpoint_put, db,
                               req.thread_id, req.checkpoint_ns,
                               req.checkpoint_id, req.parent_checkpoint_id,
                               req.type, req.checkpoint, req.metadata)

        @router.post("/checkpoint/put_writes")
        async def cp_put_writes(req: CheckpointPutWritesRequest):
            writes = [w.model_dump() for w in req.writes]
            return _wrap_errors(handle_checkpoint_put_writes, db,
                               req.thread_id, req.checkpoint_ns,
                               req.checkpoint_id, req.task_id, writes)

        @router.post("/checkpoint/list")
        async def cp_list(req: CheckpointListRequest):
            return _wrap_errors(handle_checkpoint_list, db,
                               req.thread_id, req.checkpoint_ns,
                               req.before_id, req.limit, req.filter)

    return router

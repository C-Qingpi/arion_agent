"""AsyncProxySaver: checkpoint saver that delegates to a host-side HTTP proxy.

Solves the grpcfuse file descriptor degradation issue (Phase 17+).
The proxy server runs on the host with native SQLite access. The agent
in Docker sends checkpoint operations over HTTP instead of touching
the bind-mounted SQLite file directly.

Protocol: JSON with base64-encoded binary blobs. Works at the raw SQL
level -- neither client nor server needs to deserialize checkpoint internals.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import aiohttp
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


class AsyncProxySaver(BaseCheckpointSaver[str]):
    """Checkpoint saver that delegates to a host-side HTTP proxy server."""

    def __init__(self, proxy_url: str) -> None:
        super().__init__(serde=JsonPlusSerializer())
        self.proxy_url = proxy_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def _post(self, path: str, data: dict) -> dict | None:
        session = await self._get_session()
        url = f"{self.proxy_url}{path}"
        try:
            async with session.post(url, json=data) as resp:
                if resp.status == 204:
                    return None
                return await resp.json()
        except Exception:
            logger.exception("Checkpoint proxy request failed: %s", path)
            raise

    async def setup(self) -> None:
        pass

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        result = await self._post("/checkpoint/get_tuple", {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        })

        if result is None or result.get("not_found"):
            return None

        cp_type = result["type"]
        cp_blob = base64.b64decode(result["checkpoint"])
        metadata = result.get("metadata", {})
        parent_id = result.get("parent_checkpoint_id")
        r_thread_id = result.get("thread_id", thread_id)
        r_checkpoint_id = result["checkpoint_id"]

        cfg: RunnableConfig = {
            "configurable": {
                "thread_id": r_thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": r_checkpoint_id,
            }
        }

        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": r_thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        pending_writes = []
        for w in result.get("writes", []):
            w_type = w["type"]
            w_blob = base64.b64decode(w["value"])
            pending_writes.append(
                (w["task_id"], w["channel"], self.serde.loads_typed((w_type, w_blob)))
            )

        return CheckpointTuple(
            cfg,
            self.serde.loads_typed((cp_type, cp_blob)),
            cast(CheckpointMetadata, metadata),
            parent_config,
            pending_writes,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id = str(config["configurable"]["thread_id"]) if config else None
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") if config else ""
        before_id = get_checkpoint_id(before) if before else None

        result = await self._post("/checkpoint/list", {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "before_id": before_id,
            "limit": limit,
            "filter": filter,
        })

        if not result or not result.get("checkpoints"):
            return

        for item in result["checkpoints"]:
            cp_type = item["type"]
            cp_blob = base64.b64decode(item["checkpoint"])
            metadata = item.get("metadata", {})
            parent_id = item.get("parent_checkpoint_id")
            r_thread_id = item["thread_id"]
            r_checkpoint_id = item["checkpoint_id"]

            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": r_thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": r_checkpoint_id,
                }
            }
            parent_config = None
            if parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": r_thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }

            yield CheckpointTuple(
                cfg,
                self.serde.loads_typed((cp_type, cp_blob)),
                cast(CheckpointMetadata, metadata),
                parent_config,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        type_, serialized = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        )

        await self._post("/checkpoint/put", {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
            "parent_checkpoint_id": parent_checkpoint_id,
            "type": type_,
            "checkpoint": base64.b64encode(serialized).decode("ascii"),
            "metadata": serialized_metadata,
        })

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = str(config["configurable"]["checkpoint_id"])

        serialized_writes = []
        for idx, (channel, value) in enumerate(writes):
            w_type, w_blob = self.serde.dumps_typed(value)
            serialized_writes.append({
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": w_type,
                "value": base64.b64encode(w_blob).decode("ascii"),
            })

        await self._post("/checkpoint/put_writes", {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "writes": serialized_writes,
        })

    def get_next_version(self, current: str | None, channel: None) -> str:
        import random
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

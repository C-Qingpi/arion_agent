"""Shell execution backend abstraction.

Mirrors the IOBackend pattern: LocalShellBackend runs commands via
subprocess, RemoteShellBackend proxies to a host-side HTTP service.
execute_python routes through the active backend. Background CLI jobs
use JobRegistry (detached processes, disk-backed) and always run locally.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from arion_agent.environments.shell.executor import ShellResult, run_command

logger = logging.getLogger(__name__)


class ShellBackend(ABC):
    """Abstract base for shell command execution."""

    @abstractmethod
    async def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout: float = 120.0,
        max_output_bytes: int = 200_000,
        env: dict[str, str] | None = None,
    ) -> ShellResult: ...


class LocalShellBackend(ShellBackend):
    """Runs commands locally via asyncio subprocess.

    When sandbox_cfg is set with confinement="bwrap", commands are
    wrapped with bubblewrap for namespace isolation.
    """

    def __init__(self, sandbox_cfg: Any = None) -> None:
        self._sandbox_cfg = sandbox_cfg

    async def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout: float = 120.0,
        max_output_bytes: int = 200_000,
        env: dict[str, str] | None = None,
    ) -> ShellResult:
        return await run_command(
            command,
            cwd=cwd,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            env=env,
            sandbox_cfg=self._sandbox_cfg,
        )


class RemoteShellBackend(ShellBackend):
    """Proxies command execution to a host-side HTTP service.

    The host service runs the actual subprocess and returns output.
    Used when the agent runs in Docker or on a remote machine and
    commands should execute on the host (or wherever the service runs).

    Endpoint contract:
        POST /shell/exec
        Body: {"command": [...], "cwd": "relative/path", "timeout": 120}
        Response: {"output": "...", "exit_code": 0, "truncated": false}
    """

    def __init__(self, url: str, timeout: float = 300.0) -> None:
        import httpx
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=timeout,
        )

    async def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout: float = 120.0,
        max_output_bytes: int = 200_000,
        env: dict[str, str] | None = None,
    ) -> ShellResult:
        try:
            resp = await self._client.post("/shell/exec", json={
                "command": command,
                "cwd": str(cwd),
                "timeout": timeout,
                "max_output_bytes": max_output_bytes,
                "env": env,
            })
            if resp.status_code >= 400:
                return ShellResult(
                    output=f"Remote shell error ({resp.status_code}): {resp.text}",
                    exit_code=1,
                )
            data = resp.json()
            return ShellResult(
                output=data.get("output", ""),
                exit_code=data.get("exit_code", 1),
                truncated=data.get("truncated", False),
            )
        except Exception as exc:
            return ShellResult(
                output=f"Remote shell connection failed: {exc}",
                exit_code=1,
            )

    async def close(self) -> None:
        await self._client.aclose()

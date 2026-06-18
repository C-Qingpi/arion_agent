"""Shell command execution with output capture."""

from __future__ import annotations

import asyncio
import locale
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHELL_ENCODING: str | None = None


def _get_shell_encoding() -> str:
    """Detect encoding used by shell subprocess output. Cached per process."""
    global _SHELL_ENCODING
    if _SHELL_ENCODING is not None:
        return _SHELL_ENCODING

    if sys.platform == "win32":
        try:
            cp = __import__("ctypes").windll.kernel32.GetConsoleOutputCP()
            if cp and cp != 0:
                _SHELL_ENCODING = f"cp{cp}"
            else:
                cp = __import__("ctypes").windll.kernel32.GetOEMCP()
                _SHELL_ENCODING = f"cp{cp}" if cp else locale.getpreferredencoding(False)
        except Exception:
            _SHELL_ENCODING = locale.getpreferredencoding(False)
    else:
        lang = os.environ.get("LC_ALL") or os.environ.get("LC_CTYPE") or os.environ.get("LANG", "")
        if ".UTF-8" in lang.upper() or ".UTF8" in lang.upper():
            _SHELL_ENCODING = "utf-8"
        elif lang:
            parts = lang.split(".")
            _SHELL_ENCODING = parts[-1] if len(parts) > 1 else locale.getpreferredencoding(False)
        else:
            _SHELL_ENCODING = locale.getpreferredencoding(False)

    return _SHELL_ENCODING


def _decode_output(data: bytes, encoding: str | None = None) -> str:
    """Decode subprocess output. Try UTF-8 first (many tools use it); on failure use
    detected shell encoding. Child processes (python, bash, cmd) may use different
    encodings than the console, so we cannot rely on a single encoding."""
    if not data:
        return ""
    if encoding:
        try:
            return data.decode(encoding, errors="replace")
        except (LookupError, TypeError):
            pass
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        enc = _get_shell_encoding()
        return data.decode(enc, errors="replace")


@dataclass
class ShellResult:
    output: str
    exit_code: int
    truncated: bool = False


async def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: float = 120.0,
    max_output_bytes: int = 200_000,
    env: dict[str, str] | None = None,
    sandbox_cfg: Any = None,
) -> ShellResult:
    """Run a command via asyncio subprocess with timeout and output capture.

    stdout and stderr are combined. stderr lines prefixed with [stderr].
    Output decoded using detected shell encoding (no forced UTF-8).
    When sandbox_cfg is provided with confinement="bwrap", the command
    is wrapped with bubblewrap for namespace isolation.
    """
    if sandbox_cfg is not None and sandbox_cfg.confinement == "bwrap":
        from arion_agent.environments._sandbox.confinement import build_bwrap_command
        command = build_bwrap_command(command, sandbox_cfg)

    effective_env = os.environ.copy()
    if env:
        effective_env.update(env)

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=effective_env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ShellResult(
                output=f"Command timed out after {timeout:.0f}s. Set timeout_seconds for longer operations.",
                exit_code=124,
            )

        stdout_text = _decode_output(stdout_bytes)
        stderr_text = _decode_output(stderr_bytes)

        parts = []
        if stdout_text:
            parts.append(stdout_text)
        if stderr_text:
            stderr_lines = stderr_text.strip().split("\n")
            parts.extend(f"[stderr] {line}" for line in stderr_lines)

        output = "\n".join(parts) if parts else "<no output>"

        truncated = False
        if len(output) > max_output_bytes:
            output = output[:max_output_bytes] + f"\n\n... Output truncated at {max_output_bytes} bytes."
            truncated = True

        exit_code = proc.returncode or 0
        if exit_code != 0:
            output = f"{output.rstrip()}\n\nExit code: {exit_code}"

        return ShellResult(output=output, exit_code=exit_code, truncated=truncated)

    except FileNotFoundError as exc:
        return ShellResult(
            output=f"Command not found: {command[0]}. {exc}",
            exit_code=127,
        )
    except OSError as exc:
        winerr = getattr(exc, "winerror", None)
        if winerr == 10055 or (hasattr(exc, "errno") and exc.errno == 10055):
            return ShellResult(
                output="Execution error: No buffer space (WSAENOBUFS). System resource limit. Try fewer concurrent processes or restart.",
                exit_code=1,
            )
        return ShellResult(
            output=f"Execution error (OS): {exc}",
            exit_code=1,
        )
    except Exception as exc:
        return ShellResult(
            output=f"Execution error: {type(exc).__name__}: {exc}",
            exit_code=1,
        )

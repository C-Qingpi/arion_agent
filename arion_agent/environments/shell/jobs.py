"""Recoverable background-job registry.

Replaces the old interactive PTY/tmux terminals. Every command runs as a
detached background job: stdout+stderr stream to a log file and the exit code
lands in a status file written by a shell sentinel. State is reconstructed from
disk on every call, so jobs survive agent-process restarts and crashes without a
central lock file.

Design goals: simple, no PTY, no interactivity, best-effort cross-OS
(Linux/macOS/WSL via bash; native Windows via cmd/pwsh).

Layout under workspace/.arion/jobs/:
    <id>.json    job metadata (command, cwd, pid, pgid, started_at, ...)
    <id>.log     combined stdout + stderr
    <id>.status  written by the job sentinel: "EXIT=<code>"
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\].*?\x07|\x1b[@-_]")

MAX_RUNNING_JOBS = 8
DEFAULT_READ_LINES = 50
MAX_READ_LINES = 1000
MAX_READ_CHARS = 200_000
START_SETTLE_SECONDS = 2.0
START_SETTLE_POLL = 0.1
WAIT_POLL_SECONDS = 0.3
STOP_GRACE_SECONDS = 3.0
STOPPED_EXIT = -15
SNAPSHOT_LINES = 15

STATE_RUNNING = "running"
STATE_EXITED = "exited"
STATE_STOPPED = "stopped"
STATE_ENDED = "ended"  # process gone but no exit code recorded (crash/restart kill)


@dataclass
class JobState:
    job_id: str
    state: str
    exit_code: int | None
    command: str
    cwd_display: str
    description: str
    pid: int | None
    started_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_output(raw: str) -> str:
    lines = []
    for line in raw.split("\n"):
        if "\r" in line:
            line = line.split("\r")[-1]
        lines.append(line)
    return _ANSI_ESCAPE.sub("", "\n".join(lines))


def _sanitize_id(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", label.strip())
    return cleaned or "job"


# ---------------------------------------------------------------------------
# Cross-OS process helpers
# ---------------------------------------------------------------------------

def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _descendant_pids(root_pid: int | None) -> list[int]:
    """Every descendant PID of root_pid via the ps ppid graph (Unix).

    Catches children that left our process group (e.g. via setsid) but are
    still parented under the job, which killpg alone would miss. Truly
    daemonized double-forkers that reparent to init are out of scope.
    """
    if not root_pid or sys.platform == "win32":
        return []
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid="], capture_output=True, text=True
    ).stdout
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].lstrip("-").isdigit():
            continue
        children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found: list[int] = []
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                found.append(child)
                stack.append(child)
    return found


def _group_alive(pid: int | None, pgid: int | None) -> bool:
    """True while any process in the job's group (or the root pid) is alive."""
    if sys.platform == "win32":
        return _pid_alive(pid)
    if pgid:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return _pid_alive(pid)
        except PermissionError:
            return True
        return True
    return _pid_alive(pid)


def _signal_unix_tree(pid: int | None, pgid: int | None, sig: int) -> None:
    # Snapshot descendants before signalling: ppid links break as procs die.
    targets = _descendant_pids(pid)
    if pid:
        targets.append(pid)
    for target_pid in targets:
        try:
            os.kill(target_pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    if pgid:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _terminate_tree(pid: int | None, pgid: int | None) -> None:
    if sys.platform == "win32":
        if pid:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
            )
        return
    _signal_unix_tree(pid, pgid, signal.SIGTERM)


def _force_kill_tree(pid: int | None, pgid: int | None) -> None:
    if sys.platform == "win32":
        return  # taskkill /F already forceful
    _signal_unix_tree(pid, pgid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Shell composition (mirror inline executor's bash-preferred routing)
# ---------------------------------------------------------------------------

def _shell_family() -> str:
    if shutil.which("bash"):
        return "bash"
    if sys.platform == "win32":
        if shutil.which("pwsh") or shutil.which("powershell"):
            return "pwsh"
        return "cmd"
    return "sh"


def _compose_invocation(command: str, status_path: Path) -> list[str]:
    """Return an argv that runs *command* then writes EXIT=<code> to status_path."""
    family = _shell_family()
    status = str(status_path)
    if family in ("bash", "sh"):
        shell = shutil.which("bash") if family == "bash" else (shutil.which("sh") or "/bin/sh")
        inner = (
            f"{command}\n"
            f"__arion_rc=$?\n"
            f"printf 'EXIT=%s\\n' \"$__arion_rc\" > {shlex.quote(status)}\n"
        )
        argv = [shell, "-c", inner]
        # stdbuf line-buffers libc stdio and propagates to children via
        # LD_PRELOAD, so non-Python jobs also stream to the log live. Present
        # on Linux; absent on macOS (PYTHONUNBUFFERED still covers Python there).
        stdbuf = shutil.which("stdbuf")
        if stdbuf:
            return [stdbuf, "-oL", "-eL", *argv]
        return argv
    if family == "pwsh":
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        q = status.replace("'", "''")
        inner = (
            f"{command}\n"
            f"\"EXIT=$LASTEXITCODE\" | Out-File -Encoding ascii -FilePath '{q}'\n"
        )
        return [pwsh, "-NoProfile", "-NonInteractive", "-Command", inner]
    # cmd.exe
    return ["cmd.exe", "/c", f'{command} & echo EXIT=%ERRORLEVEL%> "{status}"']


def _build_job_env(source: dict[str, str], workspace: Path) -> dict[str, str]:
    """Wire the agent venv so python3/pip in jobs match execute_python.

    Jobs write to a log file, not a TTY, so programs block-buffer stdout and
    output only lands on disk when the buffer fills or the process exits. Force
    unbuffered/line-buffered output so shell_log reflects progress live.
    """
    env = source.copy()
    env.setdefault("TERM", "xterm-256color")
    env["PYTHONUNBUFFERED"] = "1"
    exe = Path(sys.executable).resolve()
    venv_root = exe.parent.parent
    if exe.parent.name == "bin" and (venv_root / "pyvenv.cfg").is_file():
        env["VIRTUAL_ENV"] = str(venv_root)
        env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
        env["ARION_PYTHON"] = str(exe)
    return env


def _display_cwd(workspace: Path, cwd: Path) -> str:
    try:
        rel = cwd.resolve().relative_to(workspace.resolve())
        return str(rel).replace("\\", "/") or "."
    except ValueError:
        return str(cwd)


def _resolve_cwd(workspace: Path, cwd: str) -> Path | str:
    if not cwd or cwd == ".":
        return workspace
    from arion_agent.environments._sandbox.paths import PathConfinementError, resolve_path

    try:
        resolved = resolve_path(cwd, workspace)
    except PathConfinementError as exc:
        return f"Invalid cwd '{cwd}': {exc}"
    if not resolved.is_dir():
        return f"CWD is not a directory: {cwd}"
    return resolved


class JobRegistry:
    """Workspace-level registry of recoverable background jobs.

    Shared across agents in a workspace. State lives on disk; every query
    rebuilds from the .arion/jobs directory, so a fresh registry after an
    agent restart sees all still-running and finished jobs.
    """

    def __init__(self, workspace: Path, sandbox_cfg: object | None = None) -> None:
        self._workspace = workspace
        self._sandbox_cfg = sandbox_cfg
        self._dir = workspace / ".arion" / "jobs"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ---- paths ----

    def _meta_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.log"

    def _status_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.status"

    # ---- metadata ----

    def _load_meta(self, job_id: str) -> dict | None:
        path = self._meta_path(job_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_meta(self, job_id: str, meta: dict) -> None:
        tmp = self._meta_path(job_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(self._meta_path(job_id))

    def _all_ids(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def _unique_id(self, base: str) -> str:
        base = _sanitize_id(base)
        candidate = base
        i = 1
        while self._meta_path(candidate).exists():
            i += 1
            candidate = f"{base}-{i}"
        return candidate

    def _read_status_exit(self, job_id: str) -> int | None:
        path = self._status_path(job_id)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"EXIT=(-?\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def _resolve_state(self, job_id: str, meta: dict) -> tuple[str, int | None]:
        exit_code = self._read_status_exit(job_id)
        if exit_code is not None:
            if exit_code == STOPPED_EXIT:
                return STATE_STOPPED, exit_code
            return STATE_EXITED, exit_code
        if _pid_alive(meta.get("pid")):
            return STATE_RUNNING, None
        return STATE_ENDED, None

    def job_state(self, job_id: str) -> JobState | None:
        meta = self._load_meta(job_id)
        if meta is None:
            return None
        state, exit_code = self._resolve_state(job_id, meta)
        return JobState(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            command=meta.get("command", ""),
            cwd_display=meta.get("cwd_display", "."),
            description=meta.get("description", ""),
            pid=meta.get("pid"),
            started_at=meta.get("started_at", ""),
        )

    def _running_count(self) -> int:
        count = 0
        for jid in self._all_ids():
            meta = self._load_meta(jid)
            if meta and self._resolve_state(jid, meta)[0] == STATE_RUNNING:
                count += 1
        return count

    # ---- log access ----

    def _read_log_lines(self, job_id: str) -> list[str]:
        path = self._log_path(job_id)
        if not path.exists() or path.stat().st_size == 0:
            return []
        raw = path.read_text(encoding="utf-8", errors="replace")
        return _clean_output(raw).splitlines()

    @staticmethod
    def _state_label(state: str, exit_code: int | None) -> str:
        if state == STATE_RUNNING:
            return "running"
        if state == STATE_STOPPED:
            return "stopped"
        if state == STATE_EXITED:
            return f"exited({exit_code})"
        return "ended(no exit code)"

    # ---- public API ----

    async def run(self, command: str, *, cwd: str = ".", description: str = "") -> str:
        command = command.strip()
        if not command:
            return "TOOL ERROR (shell_run)\nType: MissingParam\nMessage: command is required."

        if self._running_count() >= MAX_RUNNING_JOBS:
            return (
                f"TOOL ERROR (shell_run)\nType: TooManyJobs\n"
                f"Message: {MAX_RUNNING_JOBS} jobs already running. "
                f"Stop one with shell_stop or wait for completion."
            )

        resolved_cwd = _resolve_cwd(self._workspace, cwd)
        if isinstance(resolved_cwd, str):
            return f"TOOL ERROR (shell_run)\nType: BadCwd\nMessage: {resolved_cwd}"

        label = description or command.split()[0]
        job_id = self._unique_id(label)
        log_path = self._log_path(job_id)
        status_path = self._status_path(job_id)
        status_path.unlink(missing_ok=True)
        log_path.write_bytes(b"")

        env = _build_job_env(dict(os.environ), self._workspace)
        argv = _compose_invocation(command, status_path)
        if (
            self._sandbox_cfg is not None
            and getattr(self._sandbox_cfg, "confinement", "none") == "bwrap"
        ):
            from arion_agent.environments._sandbox.confinement import build_bwrap_command

            argv = build_bwrap_command(argv, self._sandbox_cfg)

        cwd_display = _display_cwd(self._workspace, resolved_cwd)
        try:
            pid, pgid = await asyncio.to_thread(
                self._spawn, argv, str(resolved_cwd), env, log_path
            )
        except Exception as exc:
            return f"TOOL ERROR (run)\nType: SpawnFailed\nMessage: {type(exc).__name__}: {exc}"

        meta = {
            "id": job_id,
            "command": command,
            "cwd": str(resolved_cwd),
            "cwd_display": cwd_display,
            "description": description.strip(),
            "pid": pid,
            "pgid": pgid,
            "started_at": _now_iso(),
        }
        self._write_meta(job_id, meta)

        await self._settle(job_id)
        return self._run_summary(job_id)

    def _spawn(
        self, argv: list[str], cwd: str, env: dict[str, str], log_path: Path
    ) -> tuple[int, int | None]:
        logf = open(log_path, "ab", buffering=0)
        try:
            if sys.platform == "win32":
                flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                proc = subprocess.Popen(
                    argv, cwd=cwd, env=env,
                    stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    creationflags=flags,
                    close_fds=True,
                )
                return proc.pid, None
            proc = subprocess.Popen(
                argv, cwd=cwd, env=env,
                stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            try:
                pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pgid = proc.pid
            return proc.pid, pgid
        finally:
            logf.close()

    async def _settle(self, job_id: str) -> None:
        """Brief wait so fast commands report output/exit in the run() result."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + START_SETTLE_SECONDS
        while loop.time() < deadline:
            if self._status_path(job_id).exists():
                return
            await asyncio.sleep(START_SETTLE_POLL)

    def _run_summary(self, job_id: str) -> str:
        st = self.job_state(job_id)
        assert st is not None
        label = self._state_label(st.state, st.exit_code)
        tail = self._read_log_lines(job_id)[-SNAPSHOT_LINES:]
        header = (
            f"Started job '{job_id}' [{label}] in {st.cwd_display}\n"
            f"$ {st.command}\n"
            f"Poll with shell_log job_id={job_id}; wait on it with "
            f"wait job_id={job_id} until_output=...; stop with shell_stop job_id={job_id}.\n"
            f"--- output ---"
        )
        body = "\n".join(tail) if tail else "(no output yet)"
        return f"{header}\n{body}"

    def list_jobs(self) -> str:
        ids = self._all_ids()
        if not ids:
            return "No jobs. Start one with shell_run."
        states = [self.job_state(jid) for jid in ids]
        states = [s for s in states if s is not None]
        states.sort(key=lambda s: s.started_at)
        running = sum(1 for s in states if s.state == STATE_RUNNING)
        parts = [f"Jobs ({len(states)} total, {running} running, max {MAX_RUNNING_JOBS}):"]
        for s in states:
            label = self._state_label(s.state, s.exit_code)
            desc = f' — {s.description}' if s.description else ""
            cmd = s.command if len(s.command) <= 80 else s.command[:77] + "..."
            parts.append(f"  [{s.job_id}] {label}, cwd {s.cwd_display}{desc}")
            parts.append(f"    $ {cmd}")
            tail = self._read_log_lines(s.job_id)[-2:]
            for line in tail:
                stripped = line.strip()
                if stripped:
                    parts.append(f"    > {stripped[:120]}")
        return "\n".join(parts)

    def read_log(self, job_id: str, lines: int = DEFAULT_READ_LINES, grep: str = "") -> str:
        st = self.job_state(job_id)
        if st is None:
            return f"Job '{job_id}' not found. Use shell_list to see jobs."
        all_lines = self._read_log_lines(job_id)
        if grep:
            all_lines = [ln for ln in all_lines if grep in ln]
        total = len(all_lines)
        if lines <= 0:
            selected = all_lines
        else:
            selected = all_lines[-min(lines, MAX_READ_LINES):]
        body = "\n".join(selected) if selected else "(no output)"
        if len(body) > MAX_READ_CHARS:
            body = body[-MAX_READ_CHARS:]
            body = "... (truncated)\n" + body
        label = self._state_label(st.state, st.exit_code)
        grep_part = f", grep '{grep}'" if grep else ""
        header = (
            f"Job '{job_id}' [{label}], showing {len(selected)}/{total} line(s){grep_part}\n"
            f"--- output ---"
        )
        return f"{header}\n{body}"

    async def stop(self, job_id: str) -> str:
        meta = self._load_meta(job_id)
        if meta is None:
            return f"Job '{job_id}' not found."
        state, _ = self._resolve_state(job_id, meta)
        if state != STATE_RUNNING:
            return f"Job '{job_id}' is not running ({self._state_label(state, self._read_status_exit(job_id))})."
        pid = meta.get("pid")
        pgid = meta.get("pgid")
        await asyncio.to_thread(_terminate_tree, pid, pgid)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + STOP_GRACE_SECONDS
        while loop.time() < deadline:
            if not _group_alive(pid, pgid):
                break
            await asyncio.sleep(0.1)
        if _group_alive(pid, pgid):
            await asyncio.to_thread(_force_kill_tree, pid, pgid)

        if not self._status_path(job_id).exists():
            self._status_path(job_id).write_text(f"EXIT={STOPPED_EXIT}\n", encoding="utf-8")
        return f"Stopped job '{job_id}' and its child processes."

    def exists(self, job_id: str) -> bool:
        return self._meta_path(job_id).exists()

    def find_in_log(self, job_id: str, pattern: str) -> str | None:
        for line in self._read_log_lines(job_id):
            if pattern in line:
                return line
        return None

    async def cleanup_all(self) -> None:
        for jid in self._all_ids():
            meta = self._load_meta(jid)
            if meta and self._resolve_state(jid, meta)[0] == STATE_RUNNING:
                await self.stop(jid)

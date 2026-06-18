"""Shell environment tools: quick Python execution + background CLI jobs.

Two execution surfaces, no interactivity:

- execute_python: one-shot, synchronous Python (inline or file). Best for quick
  scripts whose output you want immediately. Uses the shell's python3 (or a venv
  via python_path).
- shell_run + shell_list/shell_log/shell_stop: run any CLI command as a
  recoverable background job. stdout+stderr stream to a log file; the exit code
  is recorded to disk. Jobs survive agent restarts. Poll with shell_log, block
  with wait (on time or on a stdout match).

On a host with bash (incl. WSL on Windows) commands run through bash; native
Windows falls back to pwsh/cmd.exe.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable

from langchain_core.tools import tool

from arion_agent.environments._sandbox.paths import resolve_path
from arion_agent.environments.shell.backend import LocalShellBackend, ShellBackend
from arion_agent.environments.shell.jobs import MAX_RUNNING_JOBS, JobRegistry

if TYPE_CHECKING:
    from arion_agent.environments._sandbox.config import SandboxConfig

MAX_WAIT_SECONDS = 20 * 60
_WAIT_POLL_SECONDS = 0.3

SHELL_TOOL_DESCRIPTIONS: dict[str, str] = {
    "shell_run": (
        "Run a CLI command as a background job. Returns immediately with a job id and the "
        "first lines of output; the command keeps running after this returns.\n"
        "Use this for everything shell: builds, tests, servers, git, package installs, long scripts.\n"
        "stdout and stderr stream to a log on disk; the exit code is recorded. Jobs survive agent "
        "restarts (recoverable). Set cwd to your project folder (workspace-relative) and description "
        "to label the job. Poll output with shell_log; block with wait; terminate with shell_stop. "
        f"Max {MAX_RUNNING_JOBS} concurrent running jobs."
    ),
    "shell_list": (
        "List background jobs with id, state (running / exited(code) / stopped), cwd, description, "
        "command, and the last log lines. Use to see what is still running and pick a job id."
    ),
    "shell_log": (
        "Read a job's combined stdout+stderr log (last N lines, default 50). Set lines=0 for the full "
        "log. Set grep to keep only lines containing that substring. Header shows the job's state and "
        "exit code."
    ),
    "shell_stop": (
        "Stop a running job. Terminates the command and all processes it spawned (its whole "
        "process tree / group), so servers and worker subprocesses are stopped too -- nothing is "
        "left disowned. Sends SIGTERM, then SIGKILL after a short grace period. Finished jobs are "
        "left as-is; logs are preserved."
    ),
    "wait": (
        "Pause, then return. Three modes:\n"
        "1) wait seconds=N — sleep N seconds.\n"
        "2) wait job_id=X — block until job X exits (optionally cap with seconds).\n"
        "3) wait job_id=X until_output='text' — block until job X's output contains 'text' (or it "
        "exits). until_output requires job_id.\n"
        "Maximum wait is 20 minutes per call. Responds to stop/cancel promptly."
    ),
}


# ---------------------------------------------------------------------------
# Shell detection helpers (used by execute_python)
# ---------------------------------------------------------------------------

def _build_shell_command(command_str: str) -> list[str]:
    bash = shutil.which("bash")
    if bash:
        return [bash, "-c", command_str]
    if sys.platform == "win32":
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if pwsh:
            return [pwsh, "-NoProfile", "-NonInteractive", "-Command", command_str]
        return ["cmd.exe", "/c", command_str]
    sh = shutil.which("sh") or "/bin/sh"
    return [sh, "-c", command_str]


def _uses_bash() -> bool:
    return shutil.which("bash") is not None


def _is_wsl_shell() -> bool:
    return sys.platform == "win32" and _uses_bash()


def _to_shell_path(host_path: Path, workspace: Path) -> str:
    try:
        rel = host_path.relative_to(workspace)
        return str(rel).replace("\\", "/")
    except ValueError:
        pass
    if _is_wsl_shell():
        drive, rest = os.path.splitdrive(str(host_path))
        if drive and len(drive) == 2 and drive[1] == ":":
            return f"/mnt/{drive[0].lower()}{rest.replace(chr(92), '/')}"
    return str(host_path)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def create_shell_tools(
    workspace: Path,
    default_timeout: float = 120.0,
    sandbox_cfg: SandboxConfig | None = None,
    shell_backend: ShellBackend | None = None,
    job_registry: JobRegistry | None = None,
    abort_check: Callable[[], bool] | None = None,
    *,
    jobs_only: bool = False,
) -> list:
    """Create shell environment tools bound to a workspace.

    When jobs_only is True, execute_python is omitted — only background-job
    tools and wait are exposed.
    """

    backend = shell_backend or LocalShellBackend(sandbox_cfg)
    registry = job_registry or JobRegistry(workspace, sandbox_cfg=sandbox_cfg)
    docs = SHELL_TOOL_DESCRIPTIONS

    # ---- Quick synchronous Python (one-shot, no state between calls) ----

    @tool
    async def execute_python(
        code: Annotated[str, "Python code (inline mode)."] = "",
        file_path: Annotated[str, "Path to .py file (file mode). Mutually exclusive with code."] = "",
        python_path: Annotated[str, "Path to a python interpreter. Use for venv: e.g. 'project/venv/bin/python3'. Defaults to system python3."] = "",
        timeout_seconds: Annotated[float, "Override timeout. Default 120s."] = 0,
        script_args: Annotated[str, "Space-separated arguments for sys.argv."] = "",
    ) -> str:
        """Run Python and return its output synchronously. Inline mode for short scripts,
file mode for complex ones. For long-running scripts, use shell_run with 'python3 file.py'
instead so it runs as a background job. Set python_path to use a venv interpreter."""
        if not code and not file_path:
            return "TOOL ERROR (execute_python)\nType: MissingParam\nMessage: Provide code or file_path."
        if code and file_path:
            return "TOOL ERROR (execute_python)\nType: ConflictingParams\nMessage: Provide code or file_path, not both."

        timeout = timeout_seconds if timeout_seconds > 0 else default_timeout

        resolved = None
        if file_path:
            resolved = resolve_path(file_path, workspace)
            if not resolved.exists():
                return f"TOOL ERROR (execute_python)\nType: FileNotFound\nMessage: {file_path} not found."

        use_shell_routing = _uses_bash() or sys.platform != "win32"
        if sys.platform == "win32" and not python_path:
            use_shell_routing = False
        py_bin = python_path if python_path else "python3"

        if use_shell_routing:
            if resolved:
                shell_path = _to_shell_path(resolved, workspace)
                py_cmd = f"{shlex.quote(py_bin)} {shlex.quote(shell_path)}"
            else:
                py_cmd = f"{shlex.quote(py_bin)} -c {shlex.quote(code)}"
            if script_args:
                py_cmd += f" {script_args}"
            cmd = _build_shell_command(py_cmd)
        else:
            interpreter = python_path if python_path else sys.executable
            if resolved:
                cmd = [interpreter, str(resolved)]
            else:
                cmd = [interpreter, "-c", code]
            if script_args:
                cmd.extend(script_args.split())

        result = await backend.run(cmd, cwd=workspace, timeout=timeout)
        return result.output

    # ---- Background CLI jobs ----

    async def shell_run(
        command: Annotated[str, "CLI command to run as a background job (e.g. 'pytest -q', 'npm run dev')."],
        cwd: Annotated[
            str,
            "Workspace-relative working directory. Use your project folder (e.g. final_exam_standalone).",
        ] = ".",
        description: Annotated[
            str,
            "Short label for this job (e.g. 'pytest', 'dev server'). Also used as the job id base.",
        ] = "",
    ) -> str:
        return await registry.run(command, cwd=cwd, description=description)

    def shell_list() -> str:
        return registry.list_jobs()

    def shell_log(
        job_id: Annotated[str, "Job id from shell_run / shell_list."],
        lines: Annotated[int, "Recent lines to show. Default 50, max 1000. Use 0 for the full log."] = 50,
        grep: Annotated[str, "Only show lines containing this substring."] = "",
    ) -> str:
        return registry.read_log(job_id, lines=lines, grep=grep)

    async def shell_stop(
        job_id: Annotated[str, "Job id to terminate."],
    ) -> str:
        return await registry.stop(job_id)

    async def wait(
        seconds: Annotated[float, "Seconds to wait. Max 1200 (20 minutes). Optional when job_id is set."] = 0,
        job_id: Annotated[str, "Block until this job finishes (or matches until_output)."] = "",
        until_output: Annotated[str, "Block until the job's output contains this text. Requires job_id."] = "",
    ) -> str:
        from arion_agent.util.abort import interruptible_sleep

        if until_output and not job_id:
            return "TOOL ERROR (wait)\nType: MissingParam\nMessage: until_output requires job_id."

        if not job_id:
            clamped = max(0.0, min(seconds, float(MAX_WAIT_SECONDS)))
            await interruptible_sleep(clamped, abort_check)
            return f"Waited {clamped:.1f}s."

        if not registry.exists(job_id):
            return f"TOOL ERROR (wait)\nType: NotFound\nMessage: Job '{job_id}' not found."

        cap = seconds if seconds > 0 else float(MAX_WAIT_SECONDS)
        cap = max(0.0, min(cap, float(MAX_WAIT_SECONDS)))
        return await _wait_on_job(registry, job_id, until_output, cap, abort_check)

    shell_run = tool(description=docs["shell_run"])(shell_run)
    shell_list = tool(description=docs["shell_list"])(shell_list)
    shell_log = tool(description=docs["shell_log"])(shell_log)
    shell_stop = tool(description=docs["shell_stop"])(shell_stop)
    wait = tool(description=docs["wait"])(wait)

    job_tools = [shell_run, shell_list, shell_log, shell_stop, wait]
    if jobs_only:
        return job_tools
    return [execute_python, *job_tools]


async def _wait_on_job(
    registry: JobRegistry,
    job_id: str,
    until_output: str,
    cap: float,
    abort_check: Callable[[], bool] | None,
) -> str:
    import asyncio

    from arion_agent.graph import AgentAborted

    loop = asyncio.get_running_loop()
    deadline = loop.time() + cap
    while True:
        if abort_check is not None and abort_check():
            raise AgentAborted("Aborted during wait")

        if until_output:
            match = registry.find_in_log(job_id, until_output)
            if match is not None:
                return (
                    f"Job '{job_id}' output matched '{until_output}':\n"
                    f"{match.strip()}\n\n{registry.read_log(job_id, lines=20)}"
                )

        st = registry.job_state(job_id)
        if st is not None and st.state != "running":
            label = JobRegistry._state_label(st.state, st.exit_code)
            reason = "matched before exit" if until_output else "finished"
            return (
                f"Job '{job_id}' {reason} [{label}].\n\n"
                f"{registry.read_log(job_id, lines=20)}"
            )

        if loop.time() >= deadline:
            st = registry.job_state(job_id)
            label = JobRegistry._state_label(st.state, st.exit_code) if st else "unknown"
            note = (
                f"output did not contain '{until_output}' yet"
                if until_output
                else "job still running"
            )
            return (
                f"Wait timed out after {cap:.0f}s; {note} [{label}].\n\n"
                f"{registry.read_log(job_id, lines=20)}"
            )

        await asyncio.sleep(_WAIT_POLL_SECONDS)

"""Shell environment middleware: quick Python execution + background CLI jobs."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool

from arion_agent.environments._sandbox.config import SandboxConfig
from arion_agent.environments.shell.backend import ShellBackend
from arion_agent.environments.shell.jobs import JobRegistry
from arion_agent.environments.shell.tools import MAX_WAIT_SECONDS, create_shell_tools
from arion_agent.middleware.base import ArionMiddleware

SHELL_SYSTEM_PROMPT = """## Shell Environment

Your working directory is the workspace root. Tools use it as CWD automatically -- do not cd to it. Use workspace-relative paths. For shell jobs, set cwd to the project folder.

Two ways to run code:
1. execute_python: one-shot, synchronous Python (inline or file). Returns output immediately. Best for quick scripts and checks. Set python_path to use a venv interpreter. For long-running scripts, run them as a job instead.
2. Background CLI jobs: run anything else (builds, tests, servers, git, installs, long scripts) as a recoverable background job.

Background job tools:
- shell_run: start a CLI command as a background job. Returns a job id and first output lines; the command keeps running. Set cwd and description.
- shell_list: list jobs with id, state (running / exited(code) / stopped), cwd, command, recent output.
- shell_log: read a job's combined stdout+stderr log (last N lines; lines=0 for full; grep to filter).
- shell_stop: terminate a running job and all processes it spawned (whole tree); nothing is left disowned.
- wait: pause then return. wait seconds=N sleeps; wait job_id=X blocks until that job exits; wait job_id=X until_output='text' blocks until the job's output contains text. Max 20 minutes per call.

Jobs stream output to disk and record their exit code, so they survive agent restarts and are recoverable. Max 8 concurrent running jobs.

Package installation:
If pip install fails with externally-managed-environment (PEP 668, common on Ubuntu 24.04+), create a project-local venv:
1. shell_run: python3 -m venv <project_dir>/venv && <project_dir>/venv/bin/pip install <packages>
2. execute_python: set python_path to <project_dir>/venv/bin/python3 for later calls.
3. Jobs: prefix the command with <project_dir>/venv/bin/ or activate the venv.
Fallback for trivial packages: pip install --break-system-packages <pkg>.
System packages (apt) may need sudo and network; report failures rather than retrying indefinitely."""

SHELL_JOBS_ONLY_SYSTEM = """## Shell Environment

Your working directory is the workspace root; set cwd to the project folder for jobs. Use workspace-relative paths.

Run every command as a recoverable background CLI job:
- shell_run: start a command as a background job (returns job id; keeps running).
- shell_list: list jobs with state, cwd, command, recent output.
- shell_log: read a job's stdout+stderr log (last N lines; lines=0 for full; grep to filter).
- shell_stop: terminate a running job and all processes it spawned (whole tree); nothing is left disowned.
- wait: wait seconds=N; or wait job_id=X until it exits; or wait job_id=X until_output='text'. Max 20 minutes.

Jobs stream output to disk and record exit codes, so they survive agent restarts. Max 8 concurrent running jobs.
If pip fails with externally-managed-environment (PEP 668), create a project-local venv via shell_run, then use its python3/pip path."""


class ShellEnvironment(ArionMiddleware):
    """Middleware providing quick Python execution and background CLI jobs.

    When shell_backend is provided, execute_python routes through it.
    Background jobs run locally and are tracked by a disk-backed JobRegistry
    that survives agent-process restarts.

    Set jobs_only=True to expose only the background-job tools and wait
    (no execute_python).

    For multi-agent setups sharing a workspace, pass the same JobRegistry:
        registry = JobRegistry(workspace_dir)
        agent_a = create_arion_agent(..., middleware=[ShellEnvironment(cfg, registry=registry)])
        agent_b = create_arion_agent(..., middleware=[ShellEnvironment(cfg, registry=registry)])
    """

    def __init__(
        self,
        sandbox_config: SandboxConfig,
        shell_backend: ShellBackend | None = None,
        *,
        registry: JobRegistry | None = None,
        abort_check: Callable[[], bool] | None = None,
        jobs_only: bool = False,
    ) -> None:
        self._config = sandbox_config
        self._jobs_only = jobs_only
        self._registry = registry or JobRegistry(
            sandbox_config.workspace_dir,
            sandbox_cfg=sandbox_config,
        )
        self._tools = create_shell_tools(
            sandbox_config.workspace_dir,
            default_timeout=sandbox_config.default_shell_timeout,
            sandbox_cfg=sandbox_config,
            shell_backend=shell_backend,
            job_registry=self._registry,
            abort_check=abort_check,
            jobs_only=jobs_only,
        )

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    @property
    def tool_timeout_overrides(self) -> dict[str, int | None]:
        return {"wait": MAX_WAIT_SECONDS}

    @property
    def registry(self) -> JobRegistry:
        return self._registry

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        prompt = SHELL_JOBS_ONLY_SYSTEM if self._jobs_only else SHELL_SYSTEM_PROMPT
        parts.append(prompt)
        return parts

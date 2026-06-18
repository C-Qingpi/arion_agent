"""Unit tests for shell wait tool limits and executor timeout wiring."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from arion_agent.environments.shell.middleware import ShellEnvironment
from arion_agent.environments.shell.tools import MAX_WAIT_SECONDS, create_shell_tools
from arion_agent.environments._sandbox.config import SandboxConfig
from arion_agent.tool_manager.executor import ToolExecutor


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _wait_tool(tools):
    return next(t for t in tools if t.name == "wait")


class TestWaitToolLimits:
    @pytest.mark.asyncio
    async def test_clamps_to_max_wait_seconds(self, workspace, monkeypatch):
        slept = []

        async def _fast_sleep(seconds, abort_check=None, poll_interval=0.25):
            slept.append(seconds)

        monkeypatch.setattr(
            "arion_agent.util.abort.interruptible_sleep",
            _fast_sleep,
        )
        tools = create_shell_tools(workspace)
        wait = _wait_tool(tools)
        result = await wait.ainvoke({"seconds": MAX_WAIT_SECONDS + 500})
        assert slept == [float(MAX_WAIT_SECONDS)]
        assert result == f"Waited {float(MAX_WAIT_SECONDS):.1f}s."

    @pytest.mark.asyncio
    async def test_executor_allows_full_max_wait(self, workspace):
        tools = create_shell_tools(workspace)
        wait = _wait_tool(tools)
        cfg = SandboxConfig(workspace_dir=workspace)
        env = ShellEnvironment(cfg)
        executor = ToolExecutor(timeout_overrides=env.tool_timeout_overrides)

        async def _run():
            return await executor.execute(
                wait,
                {"id": "wait-1", "name": "wait", "args": {"seconds": 0.2}},
            )

        msg = await asyncio.wait_for(_run(), timeout=2)
        assert "Waited 0.2s." in str(msg.content)

    def test_middleware_timeout_override_matches_max(self, workspace):
        cfg = SandboxConfig(workspace_dir=workspace)
        env = ShellEnvironment(cfg)
        assert env.tool_timeout_overrides["wait"] == MAX_WAIT_SECONDS

    def test_wait_param_description_documents_20_minutes(self, workspace):
        tools = create_shell_tools(workspace)
        wait = _wait_tool(tools)
        schema = wait.args_schema.model_json_schema()
        seconds_prop = schema["properties"]["seconds"]
        assert "1200" in seconds_prop["description"]
        assert "20 minute" in seconds_prop["description"].lower()

    @pytest.mark.asyncio
    async def test_wait_aborts_when_abort_check_set(self, workspace):
        from arion_agent.graph import AgentAborted

        state = {"abort": False}

        def abort_check() -> bool:
            return state["abort"]

        tools = create_shell_tools(workspace, abort_check=abort_check)
        wait = _wait_tool(tools)

        task = asyncio.create_task(wait.ainvoke({"seconds": 120}))
        await asyncio.sleep(0.1)
        state["abort"] = True
        with pytest.raises(AgentAborted):
            await asyncio.wait_for(task, timeout=2.0)


class TestJobsOnlyMode:
    def test_full_mode_keeps_execute_python_and_jobs(self, workspace):
        tools = create_shell_tools(workspace)
        names = {t.name for t in tools}
        assert "execute_python" in names
        assert "execute_shell_inline" not in names
        assert {"shell_run", "shell_list", "shell_log", "shell_stop", "wait"} <= names

    def test_jobs_only_omits_execute_python(self, workspace):
        tools = create_shell_tools(workspace, jobs_only=True)
        names = {t.name for t in tools}
        assert "execute_python" not in names

    def test_shell_environment_jobs_only(self, workspace):
        cfg = SandboxConfig(workspace_dir=workspace)
        env = ShellEnvironment(cfg, jobs_only=True)
        names = {t.name for t in env.tools}
        assert names == {
            "shell_run",
            "shell_list",
            "shell_log",
            "shell_stop",
            "wait",
        }

    def test_job_tool_descriptions_describe_background_jobs(self, workspace):
        tools = create_shell_tools(workspace)
        by_name = {t.name: t for t in tools}
        assert "background job" in by_name["shell_run"].description
        assert "job_id" in by_name["shell_log"].description
        assert "until_output" in by_name["wait"].description

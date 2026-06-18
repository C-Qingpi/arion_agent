"""Tests for the recoverable background-job registry and shell job tools."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from arion_agent.environments.shell.jobs import JobRegistry, MAX_RUNNING_JOBS
from arion_agent.environments.shell.tools import create_shell_tools

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="background-job tests need a POSIX shell (bash)",
)


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


async def _wait_done(reg: JobRegistry, job_id: str, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        st = reg.job_state(job_id)
        if st is not None and st.state != "running":
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


class TestJobRegistry:
    @pytest.mark.asyncio
    async def test_run_records_exit_code_and_output(self, workspace):
        reg = JobRegistry(workspace)
        out = await reg.run("echo hello-jobs", description="echo")
        assert "Started job" in out
        job_id = "echo"
        await _wait_done(reg, job_id)
        st = reg.job_state(job_id)
        assert st.state == "exited"
        assert st.exit_code == 0
        assert "hello-jobs" in reg.read_log(job_id)

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, workspace):
        reg = JobRegistry(workspace)
        await reg.run("exit 3", description="fail")
        await _wait_done(reg, "fail")
        st = reg.job_state("fail")
        assert st.state == "exited"
        assert st.exit_code == 3

    @pytest.mark.asyncio
    async def test_recoverable_across_registry_instances(self, workspace):
        reg = JobRegistry(workspace)
        await reg.run("echo persisted", description="persist")
        await _wait_done(reg, "persist")
        fresh = JobRegistry(workspace)
        st = fresh.job_state("persist")
        assert st is not None
        assert st.exit_code == 0
        assert "persisted" in fresh.read_log("persist")

    @pytest.mark.asyncio
    async def test_stop_running_job(self, workspace):
        reg = JobRegistry(workspace)
        await reg.run("sleep 30", description="sleeper")
        assert reg.job_state("sleeper").state == "running"
        msg = await reg.stop("sleeper")
        assert "Stopped" in msg
        await _wait_done(reg, "sleeper")
        assert reg.job_state("sleeper").state == "stopped"

    @pytest.mark.asyncio
    async def test_find_in_log(self, workspace):
        reg = JobRegistry(workspace)
        await reg.run("echo marker-XYZ", description="finder")
        await _wait_done(reg, "finder")
        assert reg.find_in_log("finder", "marker-XYZ") is not None
        assert reg.find_in_log("finder", "absent") is None

    @pytest.mark.asyncio
    async def test_unique_ids_for_same_description(self, workspace):
        reg = JobRegistry(workspace)
        await reg.run("echo a", description="dup")
        await reg.run("echo b", description="dup")
        ids = sorted(p.stem for p in (workspace / ".arion" / "jobs").glob("*.json"))
        assert "dup" in ids
        assert "dup-2" in ids

    @pytest.mark.asyncio
    async def test_too_many_running_jobs(self, workspace):
        reg = JobRegistry(workspace)
        try:
            for i in range(MAX_RUNNING_JOBS):
                await reg.run("sleep 30", description=f"j{i}")
            out = await reg.run("sleep 30", description="overflow")
            assert "TooManyJobs" in out
        finally:
            await reg.cleanup_all()


class TestShellJobTools:
    @pytest.mark.asyncio
    async def test_wait_until_output(self, workspace):
        tools = create_shell_tools(workspace)
        by_name = {t.name: t for t in tools}
        await by_name["shell_run"].ainvoke(
            {"command": "echo READY; sleep 30", "description": "srv"}
        )
        try:
            res = await asyncio.wait_for(
                by_name["wait"].ainvoke({"job_id": "srv", "until_output": "READY"}),
                timeout=10,
            )
            assert "matched" in res
        finally:
            await by_name["shell_stop"].ainvoke({"job_id": "srv"})

    @pytest.mark.asyncio
    async def test_wait_until_job_exits(self, workspace):
        tools = create_shell_tools(workspace)
        by_name = {t.name: t for t in tools}
        await by_name["shell_run"].ainvoke({"command": "sleep 0.5", "description": "quick"})
        res = await asyncio.wait_for(
            by_name["wait"].ainvoke({"job_id": "quick"}), timeout=10
        )
        assert "finished" in res

    @pytest.mark.asyncio
    async def test_wait_until_output_requires_job_id(self, workspace):
        tools = create_shell_tools(workspace)
        wait = next(t for t in tools if t.name == "wait")
        res = await wait.ainvoke({"until_output": "x"})
        assert "requires job_id" in res

    @pytest.mark.asyncio
    async def test_shell_log_unknown_job(self, workspace):
        tools = create_shell_tools(workspace)
        log = next(t for t in tools if t.name == "shell_log")
        res = await log.ainvoke({"job_id": "nope"})
        assert "not found" in res

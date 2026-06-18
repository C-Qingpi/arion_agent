"""Unit tests for shell environment operations. No LLM needed."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from arion_agent.environments.shell.executor import run_command


class TestRunCommand:
    def test_echo(self):
        with tempfile.TemporaryDirectory() as td:
            result = asyncio.run(run_command(
                ["python", "-c", "print('hello')"],
                cwd=Path(td),
            ))
            assert result.exit_code == 0
            assert "hello" in result.output

    def test_stderr(self):
        with tempfile.TemporaryDirectory() as td:
            result = asyncio.run(run_command(
                ["python", "-c", "import sys; sys.stderr.write('err_msg\\n')"],
                cwd=Path(td),
            ))
            assert "[stderr]" in result.output
            assert "err_msg" in result.output

    def test_exit_code(self):
        with tempfile.TemporaryDirectory() as td:
            result = asyncio.run(run_command(
                ["python", "-c", "import sys; sys.exit(42)"],
                cwd=Path(td),
            ))
            assert result.exit_code == 42
            assert "Exit code: 42" in result.output

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            result = asyncio.run(run_command(
                ["python", "-c", "import time; time.sleep(999)"],
                cwd=Path(td),
                timeout=0.5,
            ))
            assert result.exit_code == 124
            assert "timed out" in result.output

    def test_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            result = asyncio.run(run_command(
                ["python", "-c", "import os; print(os.getcwd())"],
                cwd=Path(td),
            ))
            assert result.exit_code == 0
            assert td.replace("\\", "/") in result.output.replace("\\", "/") or \
                   td.replace("/", "\\") in result.output

    def test_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            result = asyncio.run(run_command(
                ["python", "-c", "print('x' * 1000)"],
                cwd=Path(td),
                max_output_bytes=100,
            ))
            assert result.truncated
            assert "truncated" in result.output.lower()

    def test_non_ascii_output(self):
        with tempfile.TemporaryDirectory() as td:
            result = asyncio.run(run_command(
                ["python", "-c", "print('\\u4f60\\u597d\\u4e16\\u754c')"],
                cwd=Path(td),
            ))
            assert result.exit_code == 0
            assert "\u4f60\u597d\u4e16\u754c" in result.output
            assert "\ufffd" not in result.output

    def test_non_ascii_stderr(self):
        with tempfile.TemporaryDirectory() as td:
            result = asyncio.run(run_command(
                ["python", "-c",
                 "import sys; sys.stderr.write('\\u9519\\u8bef\\u4fe1\\u606f\\n')"],
                cwd=Path(td),
            ))
            assert "[stderr]" in result.output
            assert "\u9519\u8bef\u4fe1\u606f" in result.output
            assert "\ufffd" not in result.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

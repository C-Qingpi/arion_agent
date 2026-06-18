"""Unit tests for file environment operations. No LLM needed."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from arion_agent.environments.file import ops
from arion_agent.environments._sandbox.paths import PathConfinementError, resolve_path


def _ws() -> tuple[tempfile.TemporaryDirectory, Path]:
    td = tempfile.TemporaryDirectory()
    ws = Path(td.name)
    (ws / ".recycle_bin").mkdir()
    return td, ws


def _extract_revision(read_result: str) -> str:
    for line in read_result.splitlines():
        if line.startswith("Revision: "):
            return line.split(": ", 1)[1]
    raise AssertionError(f"Revision token missing from read_file output:\n{read_result}")


def _extract_undo_token(result: str) -> str:
    marker = "Undo token: "
    if marker not in result:
        raise AssertionError(f"Undo token missing from operation result:\n{result}")
    return result.split(marker, 1)[1].split(".", 1)[0]


class TestPathConfinement:
    def test_relative_path(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            p = resolve_path("src/main.py", ws)
            assert str(p).startswith(str(ws))

    def test_absolute_slash(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            p = resolve_path("/src/main.py", ws)
            assert str(p).startswith(str(ws))

    def test_dotdot_escape_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "workspace"
            ws.mkdir()
            try:
                resolve_path("../../etc/passwd", ws)
                assert False, "Should have raised"
            except PathConfinementError:
                pass

    def test_tilde_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            try:
                resolve_path("~/.ssh/id_rsa", ws)
                assert False, "Should have raised"
            except PathConfinementError:
                pass

    def test_tilde_blocked_via_list_files(self):
        td, ws = _ws()
        with td:
            result = ops.list_files("~/.ssh", ws)
            assert "PathConfinement" in result
            assert "Workspace root (absolute):" in result
            assert 'path="."' in result

    def test_escape_blocked_via_list_files(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "workspace"
            ws.mkdir()
            result = ops.list_files("../../etc/passwd", ws)
            assert "PathConfinement" in result
            assert str(ws.resolve()) in result


class TestReadFile:
    def test_read_text(self):
        td, ws = _ws()
        with td:
            (ws / "hello.txt").write_text("line1\nline2\nline3")
            result = ops.read_file("hello.txt", ws)
            assert "line1" in result
            assert "Total lines: 3" in result

    def test_read_with_lines(self):
        td, ws = _ws()
        with td:
            (ws / "hello.py").write_text("def foo():\n    pass\n    return 42")
            result = ops.read_file("hello.py", ws, show_lines=True)
            assert "Revision: rev:" in result
            assert "L1|def foo():" in result
            assert "L2|    pass" in result

    def test_read_pagination(self):
        td, ws = _ws()
        with td:
            content = "\n".join(f"line {i}" for i in range(1, 501))
            (ws / "big.txt").write_text(content)
            result = ops.read_file("big.txt", ws, start_line=1, end_line=10)
            assert "Showing: lines 1-10" in result
            assert "490 more lines" in result

    def test_read_show_lines_absolute_line_numbers(self):
        """With show_lines=True, line markers must be absolute (file) line numbers, not 1-based per chunk."""
        td, ws = _ws()
        with td:
            content = "\n".join(f"line {i}" for i in range(1, 101))
            (ws / "mid.txt").write_text(content)
            result = ops.read_file("mid.txt", ws, start_line=46, end_line=75, show_lines=True)
            assert "Showing: lines 46-75" in result
            assert "L46|line 46" in result
            assert "L75|line 75" in result
            assert "L1|" not in result

    def test_read_nonexistent(self):
        td, ws = _ws()
        with td:
            result = ops.read_file("nope.txt", ws)
            assert "TOOL ERROR" in result
            assert "FileNotFound" in result

    def test_read_directory(self):
        td, ws = _ws()
        with td:
            (ws / "subdir").mkdir()
            result = ops.read_file("subdir", ws)
            assert "IsDirectory" in result

    def test_read_too_large(self):
        td, ws = _ws()
        with td:
            (ws / "huge.txt").write_text("x" * 100)
            result = ops.read_file("huge.txt", ws, max_readable_size=50)
            assert "FileTooLarge" in result


class TestWriteFile:
    def test_create(self):
        td, ws = _ws()
        with td:
            result = ops.write_file("new.txt", "hello", ws)
            assert "Created" in result
            assert (ws / "new.txt").read_text() == "hello"

    def test_create_with_dirs(self):
        td, ws = _ws()
        with td:
            result = ops.write_file("a/b/c.txt", "deep", ws)
            assert "Created" in result
            assert (ws / "a" / "b" / "c.txt").read_text() == "deep"

    def test_create_refuses_existing(self):
        td, ws = _ws()
        with td:
            (ws / "exists.txt").write_text("old")
            result = ops.write_file("exists.txt", "new", ws)
            assert "FileExists" in result
            assert (ws / "exists.txt").read_text() == "old"

    def test_append(self):
        td, ws = _ws()
        with td:
            (ws / "log.txt").write_text("first\n")
            result = ops.write_file("log.txt", "second\n", ws, mode="append")
            assert "Appended" in result
            assert (ws / "log.txt").read_text() == "first\nsecond\n"

    def test_prepend(self):
        td, ws = _ws()
        with td:
            (ws / "log.txt").write_text("second\n")
            result = ops.write_file("log.txt", "first\n", ws, mode="prepend")
            assert "Prepended" in result
            assert (ws / "log.txt").read_text() == "first\nsecond\n"

    def test_overwrite_undo(self):
        td, ws = _ws()
        with td:
            (ws / "note.txt").write_text("before")
            result = ops.write_file("note.txt", "after", ws, mode="overwrite")
            token = _extract_undo_token(result)
            assert (ws / "note.txt").read_text() == "after"
            undo_result = ops.undo_file_operation(token, ws)
            assert "Undid write_file(overwrite)" in undo_result
            assert (ws / "note.txt").read_text() == "before"


class TestEditFile:
    def test_replace_lines(self):
        td, ws = _ws()
        with td:
            (ws / "code.py").write_text("line1\nline2\nline3\nline4")
            revision = _extract_revision(ops.read_file("code.py", ws, show_lines=True))
            result = ops.edit_file("code.py", 2, 3, "replaced_a\nreplaced_b", revision, ws)
            token = _extract_undo_token(result)
            assert "replaced lines 2-3" in result
            assert "Removed 2 lines, added 2 lines, net change +0 lines" in result
            assert "call read_file again with show_lines=True" in result
            content = (ws / "code.py").read_text()
            assert "replaced_a\nreplaced_b" in content
            assert "line2" not in content
            undo_result = ops.undo_file_operation(token, ws)
            assert "Undid edit_file" in undo_result
            assert (ws / "code.py").read_text() == "line1\nline2\nline3\nline4"

    def test_delete_lines(self):
        td, ws = _ws()
        with td:
            (ws / "code.py").write_text("keep1\ndelete_me\nkeep2")
            revision = _extract_revision(ops.read_file("code.py", ws, show_lines=True))
            result = ops.edit_file("code.py", 2, 2, "", revision, ws)
            assert "replaced lines 2-2" in result
            assert "Removed 1 lines, added 0 lines, net change -1 lines" in result
            content = (ws / "code.py").read_text()
            assert "delete_me" not in content

    def test_invalid_range(self):
        td, ws = _ws()
        with td:
            (ws / "short.txt").write_text("only\ntwo")
            revision = _extract_revision(ops.read_file("short.txt", ws, show_lines=True))
            result = ops.edit_file("short.txt", 5, 6, "nope", revision, ws)
            assert "InvalidRange" in result

    def test_binary_rejected(self):
        td, ws = _ws()
        with td:
            (ws / "data.pdf").write_bytes(b"%PDF-1.4 binary content")
            result = ops.edit_file("data.pdf", 1, 1, "nope", "rev:unused", ws)
            assert "BinaryFile" in result

    def test_stale_revision_rejected(self):
        td, ws = _ws()
        with td:
            path = ws / "code.py"
            path.write_text("line1\nline2\nline3")
            revision = _extract_revision(ops.read_file("code.py", ws, show_lines=True))
            path.write_text("line1\ninserted\nline2\nline3")
            result = ops.edit_file("code.py", 2, 2, "updated", revision, ws)
            assert "StaleRead" in result
            assert "Line numbers may have shifted" in result


class TestDeleteFile:
    def test_soft_delete(self):
        td, ws = _ws()
        with td:
            (ws / "victim.txt").write_text("doomed")
            result = ops.delete_file("victim.txt", ws)
            token = _extract_undo_token(result)
            assert "recycle bin" in result
            assert not (ws / "victim.txt").exists()
            assert (ws / ".recycle_bin" / "victim.txt").exists()
            undo_result = ops.undo_file_operation(token, ws)
            assert "Undid delete_file" in undo_result
            assert (ws / "victim.txt").read_text() == "doomed"
            assert not (ws / ".recycle_bin" / "victim.txt").exists()

    def test_permanent_from_recycle(self):
        td, ws = _ws()
        with td:
            recycled = ws / ".recycle_bin" / "old.txt"
            recycled.write_text("already recycled")
            result = ops.delete_file(".recycle_bin/old.txt", ws)
            assert "Permanently deleted" in result
            assert not recycled.exists()

    def test_directory_rejected(self):
        td, ws = _ws()
        with td:
            (ws / "mydir").mkdir()
            result = ops.delete_file("mydir", ws)
            assert "IsDirectory" in result


class TestMoveFile:
    def test_move(self):
        td, ws = _ws()
        with td:
            (ws / "a.txt").write_text("content")
            result = ops.move_file("a.txt", "b.txt", ws)
            token = _extract_undo_token(result)
            assert "Moved" in result
            assert not (ws / "a.txt").exists()
            assert (ws / "b.txt").read_text() == "content"
            undo_result = ops.undo_file_operation(token, ws)
            assert "Undid move_file" in undo_result
            assert (ws / "a.txt").read_text() == "content"
            assert not (ws / "b.txt").exists()

    def test_move_into_dir(self):
        td, ws = _ws()
        with td:
            (ws / "src").mkdir()
            (ws / "file.txt").write_text("hi")
            result = ops.move_file("file.txt", "src", ws)
            assert "Moved" in result
            assert (ws / "src" / "file.txt").read_text() == "hi"

    def test_destination_exists(self):
        td, ws = _ws()
        with td:
            (ws / "a.txt").write_text("a")
            (ws / "b.txt").write_text("b")
            result = ops.move_file("a.txt", "b.txt", ws)
            assert "DestinationExists" in result


class TestUndoFileOperation:
    def test_token_expires_after_later_undoable_operation(self):
        td, ws = _ws()
        with td:
            (ws / "one.txt").write_text("one")
            first = ops.write_file("one.txt", "ONE", ws, mode="overwrite")
            first_token = _extract_undo_token(first)
            (ws / "two.txt").write_text("two")
            second = ops.write_file("two.txt", "TWO", ws, mode="overwrite")
            second_token = _extract_undo_token(second)
            expired = ops.undo_file_operation(first_token, ws)
            assert "InvalidUndoToken" in expired
            restored = ops.undo_file_operation(second_token, ws)
            assert "Undid write_file(overwrite)" in restored
            assert (ws / "two.txt").read_text() == "two"


class TestListFiles:
    def test_basic_list(self):
        td, ws = _ws()
        with td:
            (ws / "file1.txt").write_text("a")
            (ws / "subdir").mkdir()
            result = ops.list_files(".", ws)
            assert "[file]" in result
            assert "[dir]" in result
            assert ".recycle_bin" not in result

    def test_depth_one_excludes_nested_files(self):
        td, ws = _ws()
        with td:
            (ws / "src").mkdir()
            (ws / "src" / "main.py").write_text("pass")
            result = ops.list_files(".", ws, depth=1)
            assert "src/" in result or "[dir]  src/" in result
            assert "src/main.py" not in result

    def test_depth_two_includes_nested_files(self):
        td, ws = _ws()
        with td:
            (ws / "src").mkdir()
            (ws / "src" / "main.py").write_text("pass")
            result = ops.list_files(".", ws, depth=2)
            assert "src/main.py" in result

    def test_depth_limits_deeper_levels(self):
        td, ws = _ws()
        with td:
            (ws / "src").mkdir()
            (ws / "src" / "nested").mkdir()
            (ws / "src" / "nested" / "deep.py").write_text("pass")
            result = ops.list_files(".", ws, depth=2)
            assert "src/nested/" in result or "[dir]  src/nested/" in result
            assert "src/nested/deep.py" not in result

    def test_depth_invalid(self):
        td, ws = _ws()
        with td:
            result = ops.list_files(".", ws, depth=0)
            assert "InvalidParam" in result

    def test_ignore_by_extension(self):
        td, ws = _ws()
        with td:
            (ws / "main.py").write_text("pass")
            (ws / "cache.pyc").write_text("")
            (ws / "data.log").write_text("")
            result = ops.list_files(".", ws, ignore="*.pyc, *.log")
            assert "main.py" in result
            assert "cache.pyc" not in result
            assert "data.log" not in result

    def test_ignore_directory(self):
        td, ws = _ws()
        with td:
            (ws / "src").mkdir()
            (ws / "src" / "app.py").write_text("pass")
            (ws / "node_modules").mkdir()
            (ws / "node_modules" / "pkg.js").write_text("")
            result = ops.list_files(".", ws, ignore="node_modules/")
            assert "src" in result
            assert "node_modules" not in result

    def test_ignore_with_depth(self):
        td, ws = _ws()
        with td:
            (ws / "src").mkdir()
            (ws / "src" / "app.py").write_text("pass")
            (ws / "src" / "app.pyc").write_text("")
            (ws / "build").mkdir()
            (ws / "build" / "out.js").write_text("")
            result = ops.list_files(".", ws, depth=2, ignore="*.pyc, build/")
            assert "src/app.py" in result
            assert "app.pyc" not in result
            assert "build" not in result
            assert "out.js" not in result

    def test_ignore_empty_string_no_effect(self):
        td, ws = _ws()
        with td:
            (ws / "a.txt").write_text("a")
            result_no_ignore = ops.list_files(".", ws)
            result_empty = ops.list_files(".", ws, ignore="")
            assert result_no_ignore == result_empty


class TestSetDirectory:
    def test_create(self):
        td, ws = _ws()
        with td:
            result = ops.set_directory("create", "new_dir/sub", ws)
            assert "created" in result.lower()
            assert (ws / "new_dir" / "sub").is_dir()

    def test_rename(self):
        td, ws = _ws()
        with td:
            (ws / "old_name").mkdir()
            result = ops.set_directory("rename", "old_name", ws, new_name="new_name")
            assert "Renamed" in result
            assert (ws / "new_name").is_dir()

    def test_delete_to_recycle(self):
        td, ws = _ws()
        with td:
            (ws / "temp_dir").mkdir()
            (ws / "temp_dir" / "file.txt").write_text("inside")
            result = ops.set_directory("delete", "temp_dir", ws)
            assert "recycle bin" in result
            assert not (ws / "temp_dir").exists()
            assert (ws / ".recycle_bin" / "temp_dir" / "file.txt").exists()

    def test_move(self):
        td, ws = _ws()
        with td:
            (ws / "src").mkdir()
            (ws / "src" / "f.txt").write_text("hi")
            result = ops.set_directory("move", "src", ws, destination="dest/src")
            assert "Moved" in result
            assert (ws / "dest" / "src" / "f.txt").read_text() == "hi"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

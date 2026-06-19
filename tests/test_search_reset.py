"""Tests for semantic search index reset."""

from __future__ import annotations

from pathlib import Path

import pytest

from arion_agent.semantic_search.config import TEXT_EXTENSIONS, resolve_index_dir
from arion_agent.semantic_search.ignore import iter_indexable_files
from arion_agent.semantic_search.scope import resolve_index_scope
from arion_agent.semantic_search.service import SearchService
from arion_agent.semantic_search.store import ChunkStore


def test_reset_index_clears_store_and_rebuilds(tmp_path: Path) -> None:
    pytest.importorskip("fastembed")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "readme.md").write_text("# hello\n", encoding="utf-8")

    svc = SearchService(ws)
    svc.start()
    deadline = __import__("time").time() + 30
    while __import__("time").time() < deadline:
        st = svc.status()
        if st.indexed_files >= 1:
            break
        __import__("time").sleep(0.2)

    assert svc.store.chunk_count() > 0
    svc.reset_index()

    assert svc.store.chunk_count() == 0
    assert svc.store.load_manifest() == {}

    deadline = __import__("time").time() + 30
    while __import__("time").time() < deadline:
        st = svc.status()
        if st.indexed_files >= 1:
            break
        __import__("time").sleep(0.2)

    assert svc.status().indexed_files >= 1
    svc.stop()


def test_clear_index_files_without_service(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "readme.md").write_text("# hello\n", encoding="utf-8")
    index_dir = resolve_index_dir(ws)
    store = ChunkStore(index_dir)
    store.save_manifest({"readme.md": "abc"})
    store.replace_all([], [])

    store.clear()
    assert store.load_manifest() == {}
    assert store.chunk_count() == 0

    scope = resolve_index_scope(ws)
    files = iter_indexable_files(
        ws,
        scope.patterns,
        TEXT_EXTENSIONS,
        max_depth=scope.max_depth,
        only=scope.only,
        skip=scope.skip,
        allow=scope.allow,
    )
    assert files

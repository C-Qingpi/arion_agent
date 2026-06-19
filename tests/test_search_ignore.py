"""Tests for semantic search ignore patterns, depth limits, and path globs."""

from __future__ import annotations

from pathlib import Path

import pytest

from arion_agent.semantic_search.config import TEXT_EXTENSIONS
from arion_agent.semantic_search.ignore import (
    iter_indexable_files,
    load_ignore_patterns,
    path_matches_glob,
    should_ignore,
)
from arion_agent.semantic_search.retriever import hybrid_search


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "src" / "app").mkdir(parents=True)
    (root / "src" / "app" / "main.py").write_text("hello app\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "readme.md").write_text("# docs\n", encoding="utf-8")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "fake.py").write_text("ignored\n", encoding="utf-8")
    (root / ".uv" / "cache").mkdir(parents=True)
    (root / ".uv" / "cache" / "x.txt").write_text("ignored\n", encoding="utf-8")
    (root / ".recycle_bin").mkdir()
    (root / ".recycle_bin" / "old.md").write_text("old\n", encoding="utf-8")
    (root / "ArionAgentProd").mkdir()
    (root / "ArionAgentProd" / "nested.py").write_text("deploy\n", encoding="utf-8")
    deep = root / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "h" / "i" / "j" / "k"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("too deep\n", encoding="utf-8")
    return root


def test_default_ignores_uv_recycle_deploy(ws: Path) -> None:
    patterns = load_ignore_patterns(ws)
    assert should_ignore(".uv/cache/x.txt", patterns)
    assert should_ignore(".recycle_bin/old.md", patterns)
    assert should_ignore("ArionAgentProd/nested.py", patterns)
    assert should_ignore(".venv/lib/fake.py", patterns)
    assert not should_ignore("src/app/main.py", patterns)


def test_iter_indexable_skips_ignored_and_respects_depth(ws: Path) -> None:
    patterns = load_ignore_patterns(ws)
    files = iter_indexable_files(
        ws,
        patterns,
        TEXT_EXTENSIONS,
        max_depth=10,
    )
    rels = {p.relative_to(ws).as_posix() for p in files}
    assert "src/app/main.py" in rels
    assert "docs/readme.md" in rels
    assert ".venv/lib/fake.py" not in rels
    assert "ArionAgentProd/nested.py" not in rels
    assert not any(r.startswith("a/b/c/d/e/f/g/h/i/j") for r in rels)


def test_path_glob_filters_results(ws: Path) -> None:
    pytest.importorskip("fastembed")
    from arion_agent.semantic_search.service import SearchService

    svc = SearchService(ws)
    svc.start()
    deadline = __import__("time").time() + 30
    while __import__("time").time() < deadline:
        if svc.status().indexed_files >= 2:
            break
        __import__("time").sleep(0.2)

    py_hits = hybrid_search(
        "hello app docs",
        store=svc.store,
        path_glob="**/*.py",
        num_results=10,
        min_score=0.0,
    )
    assert py_hits
    assert all(h.path.endswith(".py") for h in py_hits)

    md_hits = hybrid_search(
        "docs readme",
        store=svc.store,
        target_directories=["docs"],
        path_glob="**/*.md",
        num_results=10,
        min_score=0.0,
    )
    assert md_hits
    assert all("docs" in h.path for h in md_hits)

    svc.stop()


def test_path_matches_glob_supports_double_star() -> None:
    assert path_matches_glob("src/app/main.py", "**/*.py")
    assert not path_matches_glob("docs/readme.md", "**/*.py")
    assert path_matches_glob("docs/readme.md", "docs/**")

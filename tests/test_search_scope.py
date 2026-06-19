"""Tests for .arion/search.json workspace indexing scope."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arion_agent.semantic_search.config import TEXT_EXTENSIONS
from arion_agent.semantic_search.ignore import iter_indexable_files
from arion_agent.semantic_search.indexer import scan_manifest
from arion_agent.semantic_search.scope import (
    SEARCH_CONFIG_REL,
    ensure_index_config_files,
    ensure_search_config,
    load_search_scope,
    path_indexable,
    resolve_index_scope,
    search_config_path,
)


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("app\n", encoding="utf-8")
    (root / "src" / "tests").mkdir()
    (root / "src" / "tests" / "test_app.py").write_text("test\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "readme.md").write_text("# docs\n", encoding="utf-8")
    (root / "final_exam_standalone").mkdir()
    (root / "final_exam_standalone" / "run1").mkdir(parents=True)
    (root / "final_exam_standalone" / "run1" / "analysis.md").write_text(
        "exam analysis\n",
        encoding="utf-8",
    )
    (root / "final_exam_standalone" / "run1" / "notes.txt").write_text("notes\n", encoding="utf-8")
    return root


def _write_search_config(root: Path, data: dict) -> None:
    cfg_dir = root / ".arion"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "search.json").write_text(json.dumps(data), encoding="utf-8")


def _write_search_config_raw(root: Path, text: str) -> None:
    cfg_dir = root / ".arion"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "search.json").write_text(text, encoding="utf-8")


def _indexed_rels(ws: Path) -> set[str]:
    scope = resolve_index_scope(ws)
    return {
        p.relative_to(ws).as_posix()
        for p in iter_indexable_files(
            ws,
            scope.patterns,
            TEXT_EXTENSIONS,
            max_depth=scope.max_depth,
            only=scope.only,
            skip=scope.skip,
            allow=scope.allow,
        )
    }


def test_factory_ignores_final_exam_by_default(ws: Path) -> None:
    rels = _indexed_rels(ws)
    assert "src/app.py" in rels
    assert not any(r.startswith("final_exam_standalone/") for r in rels)


def test_allow_and_only_index_analysis_in_skipped_tree(ws: Path) -> None:
    _write_search_config(ws, {
        "allow": ["final_exam_standalone/**"],
        "only": ["final_exam_standalone/**/*analysis.md"],
    })
    rels = _indexed_rels(ws)
    assert "final_exam_standalone/run1/analysis.md" in rels
    assert "final_exam_standalone/run1/notes.txt" not in rels
    assert "src/app.py" not in rels


def test_allow_alone_can_index_specific_paths_in_skipped_tree(ws: Path) -> None:
    _write_search_config(ws, {
        "allow": ["final_exam_standalone/**/*analysis.md"],
    })
    rels = _indexed_rels(ws)
    assert "final_exam_standalone/run1/analysis.md" in rels
    assert "final_exam_standalone/run1/notes.txt" not in rels
    assert "src/app.py" in rels


def test_only_restricts_without_disabling_rest_of_workspace(ws: Path) -> None:
    _write_search_config(ws, {
        "only": ["docs/**"],
    })
    manifest = scan_manifest(ws)
    assert set(manifest) == {"docs/readme.md"}


def test_skip_excludes_under_otherwise_allowed_scope(ws: Path) -> None:
    _write_search_config(ws, {
        "only": ["src/**", "docs/**"],
        "skip": ["src/**/test_*.py"],
    })
    rels = _indexed_rels(ws)
    assert "src/app.py" in rels
    assert "docs/readme.md" in rels
    assert "src/tests/test_app.py" not in rels


def test_skip_wins_over_allow(ws: Path) -> None:
    _write_search_config(ws, {
        "allow": ["final_exam_standalone/**"],
        "skip": ["final_exam_standalone/**"],
    })
    rels = _indexed_rels(ws)
    assert not any(r.startswith("final_exam_standalone/") for r in rels)


def test_max_depth_from_search_config(ws: Path) -> None:
    deep = ws / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "h" / "i" / "j" / "k"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("deep\n", encoding="utf-8")
    _write_search_config(ws, {"max_depth": 10})
    scope = load_search_scope(ws)
    assert scope.max_depth == 10
    manifest = scan_manifest(ws)
    assert not any(p.startswith("a/b/c/d/e/f/g/h/i/j") for p in manifest)


def test_legacy_keys_rejected(ws: Path) -> None:
    _write_search_config(ws, {"include_roots": ["docs"]})
    with pytest.raises(ValueError, match="include_roots"):
        load_search_scope(ws)


def test_path_indexable_precedence() -> None:
    patterns = ["blocked/"]
    assert path_indexable(
        "blocked/file.py",
        patterns,
        only=(),
        skip=(),
        allow=("blocked/**",),
    )
    assert not path_indexable(
        "blocked/file.py",
        patterns,
        only=(),
        skip=("blocked/**",),
        allow=("blocked/**",),
    )


def test_search_config_rel_constant() -> None:
    assert SEARCH_CONFIG_REL == ".arion/search.json"


def test_jsonc_comments_parsed(ws: Path) -> None:
    _write_search_config_raw(ws, """{
  // restrict to docs
  "only": ["docs/**"],
  "skip": [],
  "allow": []
}""")
    scope = load_search_scope(ws)
    assert scope.only == ("docs/**",)


def test_jsonc_does_not_strip_glob_double_slash_in_strings(ws: Path) -> None:
    _write_search_config_raw(ws, """{
  "only": ["src/**", "docs/**"],
  "skip": ["src/**/test_*.py"]
}""")
    scope = load_search_scope(ws)
    assert scope.only == ("src/**", "docs/**")
    assert scope.skip == ("src/**/test_*.py",)


def test_ensure_search_config_writes_commented_template(ws: Path) -> None:
    ensure_search_config(ws)
    text = search_config_path(ws).read_text(encoding="utf-8")
    assert "// Semantic index scope" in text
    assert '"max_depth"' in text
    assert load_search_scope(ws).max_depth == 12


def test_ensure_index_config_files_creates_both(ws: Path) -> None:
    ensure_index_config_files(ws)
    assert (ws / ".arion" / "search.json").is_file()
    assert (ws / ".searchignore").is_file()
    assert (ws / ".searchignore").read_text(encoding="utf-8").startswith("# Workspace-specific")

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from arion_agent.semantic_search.config import INDEX_MAX_DEPTH
from arion_agent.semantic_search.ignore import load_ignore_patterns, path_matches_glob, should_ignore

SEARCH_CONFIG_REL = ".arion/search.json"
SEARCHIGNORE_REL = ".searchignore"

DEFAULT_SEARCH_CONFIG = f"""{{
  // Semantic index scope for this workspace. Edit with write_file; indexer rescans on save.
  // Precedence: skip > only > factory defaults. allow bypasses factory/.searchignore skips.
  // Globs are workspace-relative. dir/** matches all files under dir.
  //
  // max_depth — max directory nesting to walk (null = no limit beyond factory default {INDEX_MAX_DEPTH})
  // skip      — extra blacklist: never index matching paths
  // only      — whitelist: when non-empty, index ONLY matching paths
  // allow     — override: index paths factory would normally skip
  //
  // Examples:
  //   {{"skip":["heavy_backup/**"]}}
  //   {{"only":["src/**","docs/**"]}}
  //   {{"only":["src/**"],"skip":["src/**/test_*.py"]}}
  //   {{"allow":["final_exam/**"],"only":["final_exam/**/*analysis.md"]}}
  //
  // UI "Reset index" clears .arion/index/ and triggers a full rebuild.

  "max_depth": {INDEX_MAX_DEPTH},
  "skip": [],
  "only": [],
  "allow": []
}}
"""

DEFAULT_SEARCHIGNORE = """\
# Workspace-specific semantic index exclusions (gitignore-style globs).
# Applied in addition to factory defaults (.venv, node_modules, .uv, etc.) and .gitignore.
# Lines starting with # are comments. Trailing / means directory.
#
# Examples:
#   heavy_project/
#   **/*.generated.ts
#   old_experiments/**

"""


@dataclass(frozen=True, slots=True)
class SearchScope:
    """Agent-editable indexing scope from .arion/search.json."""

    max_depth: int | None = INDEX_MAX_DEPTH
    only: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()
    allow: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexScope:
    patterns: list[str]
    max_depth: int | None
    only: tuple[str, ...]
    skip: tuple[str, ...]
    allow: tuple[str, ...]


def search_config_path(workspace: Path) -> Path:
    return workspace.resolve() / SEARCH_CONFIG_REL


def searchignore_path(workspace: Path) -> Path:
    return workspace.resolve() / SEARCHIGNORE_REL


def ensure_search_config(workspace: Path) -> Path:
    """Write commented default .arion/search.json when missing."""
    path = search_config_path(workspace)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_SEARCH_CONFIG, encoding="utf-8")
    return path


def ensure_searchignore(workspace: Path) -> Path:
    """Write commented default .searchignore when missing."""
    path = searchignore_path(workspace)
    if path.is_file():
        return path
    path.write_text(DEFAULT_SEARCHIGNORE, encoding="utf-8")
    return path


def ensure_index_config_files(workspace: Path) -> None:
    ensure_search_config(workspace)
    ensure_searchignore(workspace)


def is_search_config_rel(rel: str) -> bool:
    return rel.replace("\\", "/") == SEARCH_CONFIG_REL


def load_search_scope(workspace: Path) -> SearchScope:
    path = search_config_path(workspace)
    if not path.is_file():
        return SearchScope()

    data = _parse_jsonc(path.read_text(encoding="utf-8"))
    _reject_legacy_keys(data)

    max_depth = data.get("max_depth", INDEX_MAX_DEPTH)
    if max_depth is not None and not isinstance(max_depth, int):
        raise ValueError(f"{SEARCH_CONFIG_REL} max_depth must be an integer or null")

    return SearchScope(
        max_depth=max_depth,
        only=_str_list(data.get("only"), "only"),
        skip=_str_list(data.get("skip"), "skip"),
        allow=_str_list(data.get("allow"), "allow"),
    )


def resolve_index_scope(
    workspace: Path,
    *,
    extra_ignore: list[str] | None = None,
) -> IndexScope:
    scope = load_search_scope(workspace)
    patterns = load_ignore_patterns(
        workspace,
        extra_patterns=extra_ignore,
    )
    return IndexScope(
        patterns=patterns,
        max_depth=scope.max_depth,
        only=scope.only,
        skip=scope.skip,
        allow=scope.allow,
    )


def matches_scope_glob(rel_posix: str, globs: tuple[str, ...]) -> bool:
    if not globs:
        return False
    norm = rel_posix.replace("\\", "/")
    return any(path_matches_glob(norm, glob) for glob in globs)


def path_indexable(
    rel_posix: str,
    patterns: list[str],
    *,
    only: tuple[str, ...],
    skip: tuple[str, ...],
    allow: tuple[str, ...],
) -> bool:
    """Return whether a workspace-relative file path should be indexed."""
    norm = rel_posix.replace("\\", "/")

    if matches_scope_glob(norm, skip):
        return False

    blocked = should_ignore(norm, patterns)
    if blocked and not matches_scope_glob(norm, allow):
        return False

    if only and not matches_scope_glob(norm, only):
        return False

    return True


def should_prune_directory(
    dir_rel: str,
    patterns: list[str],
    *,
    only: tuple[str, ...],
    skip: tuple[str, ...],
    allow: tuple[str, ...],
) -> bool:
    """Return True when os.walk should not descend into dir_rel."""
    norm = dir_rel.replace("\\", "/").strip("/")

    if matches_scope_glob(norm, skip) or matches_scope_glob(f"{norm}/**", skip):
        return True

    if should_ignore(norm, patterns) or should_ignore(f"{norm}/", patterns):
        if subtree_might_match(norm, allow):
            return False
        return True

    if only and not subtree_might_match(norm, only) and not subtree_might_match(norm, allow):
        return True

    return False


def subtree_might_match(dir_prefix: str, globs: tuple[str, ...]) -> bool:
    if not globs:
        return False
    prefix = dir_prefix.replace("\\", "/").strip("/")
    for glob in globs:
        g = glob.replace("\\", "/").strip()
        if g.startswith("**/"):
            return True
        fixed = g.split("*")[0].strip("/")
        if not fixed:
            return True
        if fixed == prefix or fixed.startswith(prefix + "/") or prefix.startswith(fixed.split("/")[0]):
            return True
    return False


def walk_roots(
    workspace: Path,
    *,
    only: tuple[str, ...],
    allow: tuple[str, ...],
) -> list[Path]:
    """Narrow os.walk entry points when only restricts indexing to fixed prefixes."""
    workspace = workspace.resolve()
    if not only:
        return [workspace]

    roots = _pattern_roots(only)
    if not roots:
        return [workspace]

    out: list[Path] = []
    for rel in sorted(roots):
        candidate = (workspace / rel).resolve()
        if candidate.is_dir():
            out.append(candidate)
    return out or [workspace]


def search_config_mtime(workspace: Path) -> float | None:
    path = search_config_path(workspace)
    if not path.is_file():
        return None
    return path.stat().st_mtime


def _parse_jsonc(text: str) -> dict:
    """Parse JSON with // line comments (for agent-editable config files)."""
    lines: list[str] = []
    for line in text.splitlines():
        out: list[str] = []
        in_string = False
        escape = False
        i = 0
        while i < len(line):
            ch = line[i]
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\" and in_string:
                out.append(ch)
                escape = True
            elif ch == '"':
                out.append(ch)
                in_string = not in_string
            elif not in_string and ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            else:
                out.append(ch)
            i += 1
        lines.append("".join(out))
    payload = json.loads("\n".join(lines))
    if not isinstance(payload, dict):
        raise ValueError(f"{SEARCH_CONFIG_REL} must be a JSON object")
    return payload


def _str_list(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{SEARCH_CONFIG_REL} {field} must be a list of strings")
    return tuple(value)


def _pattern_roots(globs: tuple[str, ...]) -> set[str]:
    roots: set[str] = set()
    for glob in globs:
        g = glob.replace("\\", "/").strip()
        if g.startswith("**/"):
            return set()
        fixed = g.split("*")[0].strip("/")
        if not fixed:
            return set()
        roots.add(fixed.split("/")[0])
    return roots


_LEGACY_KEYS = frozenset({
    "include_roots",
    "include_globs",
    "extra_ignore",
    "unignore",
})


def _reject_legacy_keys(data: dict) -> None:
    found = sorted(k for k in data if k in _LEGACY_KEYS)
    if found:
        raise ValueError(
            f"{SEARCH_CONFIG_REL} uses removed keys {found}; "
            "use only, skip, and allow instead"
        )

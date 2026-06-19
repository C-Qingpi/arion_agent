"""Search environment middleware: optional semantic search over the workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from langchain_core.tools import BaseTool

from arion_agent.environments.search.config import SearchConfig
from arion_agent.environments.search.tools import create_search_tools
from arion_agent.middleware.base import ArionMiddleware

SEARCH_SYSTEM_PROMPT = """\
<semantic_search>
You have semantic_search to find workspace content by meaning, not exact keywords.
Use list_files with depth=2-3 first to orient, then search narrowly.
Always pass target_directories and/or path_glob (e.g. "**/*.py", "docs/**") on each query.

Indexing scope is configured in .arion/search.json (create or edit with write_file).
The file supports // line comments; a commented template is created on first index start.
.searchignore (same folder level as workspace root) adds gitignore-style exclusions with # comments.
All pattern fields use workspace-relative globs with ** support. Precedence: skip > only > factory defaults.

  max_depth — max directory nesting to walk (default 12)
  skip — never index paths matching these globs (extra blacklist)
  only — when non-empty, index only paths matching at least one glob (whitelist filter)
  allow — index paths even when factory/.searchignore would skip them (override blacklist)

Examples:
  Default workspace minus one heavy folder: {"skip":["my_backup/**"]}
  Only src and docs: {"only":["src/**","docs/**"]}
  Docs plus tests excluded: {"only":["src/**"],"skip":["src/**/test_*.py"]}
  One analysis file type inside a factory-skipped tree:
    {"allow":["final_exam_standalone/**"],"only":["final_exam_standalone/**/*analysis.md"]}

Factory defaults skip .venv, node_modules, .uv, site-packages, deploy checkouts, and build artifacts.
After editing .arion/search.json the indexer rescans automatically.
</semantic_search>"""


class SearchEnvironment(ArionMiddleware):
    """Optional middleware: background-indexed hybrid semantic search."""

    _services: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        workspace_dir: str | Path,
        config: SearchConfig | None = None,
        *,
        service: Any = None,
        system_prompt: str | None | bool = None,
    ) -> None:
        from arion_agent.semantic_search.service import SearchService, SearchServiceConfig

        self._workspace = Path(workspace_dir).resolve()
        self._config = config or SearchConfig()
        if system_prompt is False:
            self._system_prompt: str | None = None
        elif isinstance(system_prompt, str):
            self._system_prompt = system_prompt
        else:
            self._system_prompt = SEARCH_SYSTEM_PROMPT

        if service is not None:
            self._service = service
        else:
            key = str(self._workspace)
            existing = SearchEnvironment._services.get(key)
            if existing is not None:
                self._service = existing
            else:
                self._service = SearchService(
                    self._workspace,
                    self._config.index_dir,
                    config=SearchServiceConfig(
                        batch_size=self._config.batch_size,
                        extra_ignore=self._config.extra_ignore,
                        warmup_embedder=self._config.warmup_embedder,
                        enable_watcher=self._config.enable_watcher,
                    ),
                )
                SearchEnvironment._services[key] = self._service

        self._tools = create_search_tools(
            self._service,
            min_score=self._config.min_score,
            default_num_results=self._config.num_results,
        )

    @classmethod
    def reset_index_for_workspace(cls, workspace_dir: str | Path) -> None:
        from arion_agent.semantic_search.config import resolve_index_dir
        from arion_agent.semantic_search.store import ChunkStore

        workspace = Path(workspace_dir).resolve()
        key = str(workspace)
        existing = cls._services.get(key)
        if existing is not None:
            existing.reset_index()
            return
        ChunkStore(resolve_index_dir(workspace)).clear()

    @property
    def service(self) -> Any:
        return self._service

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        self._service.start()
        return None

    def after_agent(self, state: dict[str, Any]) -> None:
        return

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        if self._system_prompt:
            parts.append(self._system_prompt)
        return parts

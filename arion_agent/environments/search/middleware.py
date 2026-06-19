"""Search environment middleware: optional semantic search over the workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.environments.search.config import SearchConfig
from arion_agent.environments.search.tools import create_search_tools
from arion_agent.middleware.base import ArionMiddleware

SEARCH_SYSTEM_PROMPT = """\
<semantic_search>
You have semantic_search to find workspace content by meaning, not exact keywords.
Use it to locate relevant specs, docs, and code before reading files in depth.
Pass target_directories to narrow scope when you know the area (e.g. ["docs"]).
Indexing runs in the background when the agent starts. First run on a new
workspace can take minutes while models load; the tool reports startup status.
Early queries may return partial results until indexing completes.
</semantic_search>"""


class SearchEnvironment(ArionMiddleware):
    """Optional middleware: background-indexed hybrid semantic search."""

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

        self._tools = create_search_tools(
            self._service,
            min_score=self._config.min_score,
            default_num_results=self._config.num_results,
        )

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

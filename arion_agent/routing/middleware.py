"""Routing middleware: model selection within the ReAct loop.

Two signal mechanisms:
  1. signal_next_step tool - called alongside other tool calls (primary)
  2. [ROUTING: X] text tag - fallback for pure-text responses

Hooks:
  tools               - contributes the signal_next_step tool
  wrap_system_message  - injects the routing instruction
  wrap_tool_call       - intercepts signal_next_step results
  wrap_model_response  - parses text tag fallback, strips it
  get_model_spec       - called by model_node to select the model

Design decisions:
  - Signals from both strong and weak models are trusted.
  - Signals are consumed once (one-shot hint for the next step only).
  - Missing or unrecognized signals fall back to the strong model.
  - Text tags are stripped from content before checkpointing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from arion_agent.middleware.base import ArionMiddleware
from arion_agent.routing.config import ALL_CATEGORIES, RoutingConfig, WEAK_CATEGORIES

logger = logging.getLogger(__name__)

_ROUTING_TAG_RE = re.compile(
    r"\n?\[ROUTING:\s*(think|read|write|operate|digest)\s*\]\s*$",
    re.IGNORECASE,
)

_SIGNAL_TOOL_NAME = "signal_next_step"


class RoutingMiddleware(ArionMiddleware):
    """Intra-ReAct-loop model routing via signal tool + text tag fallback."""

    def __init__(self, config: RoutingConfig) -> None:
        self._config = config
        self._next_hint: str | None = None
        self._last_was_strong: bool = True
        self._strong_calls: int = 0
        self._weak_calls: int = 0
        self._signal_tool = self._build_signal_tool()

    @property
    def strong_calls(self) -> int:
        return self._strong_calls

    @property
    def weak_calls(self) -> int:
        return self._weak_calls

    @property
    def tools(self) -> list[BaseTool]:
        if self._config.enabled and self._config.weak_model:
            return [self._signal_tool]
        return []

    def _build_signal_tool(self) -> BaseTool:
        def signal_next_step(step: str) -> str:
            """Signal what type of processing the next step requires.

            Call this alongside your other tool calls. The step parameter
            must be one of: think, read, write, operate, digest.
            """
            step = step.strip().lower()
            if step not in ALL_CATEGORIES:
                return f"Unknown step type '{step}'. Use: think, read, write, operate, digest."
            # Hint is stored via wrap_tool_call, not here, because the
            # tool function runs before wrap_tool_call and we need the
            # middleware's _last_was_strong state to decide trust.
            return f"Acknowledged: next step tagged as '{step}'."

        return StructuredTool.from_function(
            func=signal_next_step,
            name=_SIGNAL_TOOL_NAME,
            description=(
                "Signal what the NEXT step requires so the system can "
                "optimize model selection. Call alongside other tool calls. "
                "step must be: think (complex reasoning), read (processing "
                "results or exploration), write (generating code or content), "
                "operate (running commands or simple dispatch), "
                "digest (synthesizing findings or summarizing results)."
            ),
        )

    def get_model_spec(self, default_spec: str) -> str:
        """Select model for the next LLM call. Called by model_node."""
        if (
            not self._config.enabled
            or not self._config.weak_model
            or self._config.weak_model == default_spec
        ):
            self._last_was_strong = True
            self._strong_calls += 1
            return default_spec

        hint = self._next_hint
        self._next_hint = None

        if hint in WEAK_CATEGORIES:
            self._last_was_strong = False
            self._weak_calls += 1
            logger.info("Routing: weak model (%s) for '%s' step", self._config.weak_model, hint)
            return self._config.weak_model

        self._last_was_strong = True
        self._strong_calls += 1
        if hint:
            logger.info("Routing: strong model for '%s' step", hint)
        return default_spec

    def before_agent(self, state: Any) -> dict | None:
        """Reset routing state at the start of each turn.

        Clears any stale hint from the previous turn so the first model
        call always uses the strong model (equivalent to a 'think' step).
        """
        self._next_hint = None
        self._last_was_strong = True
        return None

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        if self._config.enabled and self._config.weak_model:
            parts.append(self._config.instruction)
        return parts

    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
    ) -> Any:
        if tool_name != _SIGNAL_TOOL_NAME:
            return tool_result

        step = tool_args.get("step", "").strip().lower()
        if step not in ALL_CATEGORIES:
            return tool_result

        self._next_hint = step
        logger.info("Routing signal: [%s] (from %s model)", step, "strong" if self._last_was_strong else "weak")

        return tool_result

    def wrap_model_response(self, response: Any, **kwargs: Any) -> Any:
        """Parse text tag fallback for pure-text responses."""
        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content:
            return response

        match = _ROUTING_TAG_RE.search(content)
        if not match:
            return response

        hint = match.group(1).lower()
        cleaned = content[:match.start()].rstrip()
        response.content = cleaned

        if self._next_hint is None:
            self._next_hint = hint
            logger.info("Routing text tag: [%s]", hint)

        return response

    def reset(self) -> None:
        """Reset routing state between agent turns."""
        self._next_hint = None
        self._last_was_strong = True

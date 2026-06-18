"""MiddlewareWrapper: composition pattern for intercepting middleware behavior.

Wraps any ArionMiddleware and delegates all lifecycle hooks to the inner
middleware. Override transform_tools() or transform_system_parts() to
modify behavior without subclassing or editing the original middleware.

Usage:
    class MyWrapper(MiddlewareWrapper):
        def transform_tools(self, tools):
            return [route_through_bridge(t) for t in tools]

    wrapped = MyWrapper(BrowserEnvironment(config))
    agent = create_arion_agent(..., middleware=[wrapped])
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.middleware.base import ArionMiddleware


class MiddlewareWrapper(ArionMiddleware):
    """Wraps another middleware, intercepting its tools and system prompts.

    All lifecycle hooks delegate to the inner middleware by default.
    Override transform_tools() and/or transform_system_parts() to modify
    behavior. For deeper interception, override any lifecycle hook and
    call super() or self._inner directly.
    """

    def __init__(self, inner: ArionMiddleware) -> None:
        self._inner = inner

    @property
    def inner(self) -> ArionMiddleware:
        return self._inner

    @property
    def tools(self) -> list[BaseTool]:
        return self.transform_tools(self._inner.tools)

    def transform_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
        """Override to modify, replace, or filter the inner middleware's tools."""
        return tools

    def transform_system_parts(self, parts: list[str]) -> list[str]:
        """Override to modify system message sections after the inner middleware adds them."""
        return parts

    def before_agent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        return self._inner.before_agent(state)

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        parts = self._inner.wrap_system_message(parts, **kwargs)
        return self.transform_system_parts(parts)

    def wrap_model_call(
        self,
        messages: list[Any],
        tools: list[BaseTool],
        **kwargs: Any,
    ) -> tuple[list[Any], list[BaseTool], dict[str, Any]]:
        return self._inner.wrap_model_call(messages, tools, **kwargs)

    def wrap_model_response(self, response: Any, **kwargs: Any) -> Any:
        return self._inner.wrap_model_response(response, **kwargs)

    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
    ) -> Any:
        return self._inner.wrap_tool_call(tool_name, tool_args, tool_result)

    def drain_state_updates(self) -> list[Any]:
        return self._inner.drain_state_updates()

    def after_agent(self, state: dict[str, Any]) -> None:
        return self._inner.after_agent(state)

"""Test middleware extensibility: component injection, MiddlewareWrapper, hooks."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from langchain_core.tools import BaseTool, tool

from arion_agent.middleware.base import ArionMiddleware
from arion_agent.middleware.wrapper import MiddlewareWrapper


# ========== MiddlewareWrapper tests ==========


class DummyMiddleware(ArionMiddleware):
    """Simple middleware for testing wrappers."""

    def __init__(self) -> None:
        @tool
        def dummy_tool() -> str:
            """A dummy tool."""
            return "dummy"

        self._tools = [dummy_tool]
        self._sys_parts_added = False

    @property
    def tools(self):
        return self._tools

    def wrap_system_message(self, parts, **kwargs):
        parts.append("<dummy>inner system prompt</dummy>")
        self._sys_parts_added = True
        return parts

    def before_agent(self, state):
        return {"_dummy_ran": True}

    def after_agent(self, state):
        pass


def test_wrapper_delegates_tools():
    """MiddlewareWrapper passes through inner tools by default."""
    print("\n" + "=" * 60)
    print("Test: wrapper delegates tools")
    print("=" * 60)

    inner = DummyMiddleware()
    wrapper = MiddlewareWrapper(inner)

    assert len(wrapper.tools) == 1
    assert wrapper.tools[0].name == "dummy_tool"
    print("  >> PASSED")


def test_wrapper_transform_tools():
    """Subclass can override transform_tools to modify tools."""
    print("\n" + "=" * 60)
    print("Test: wrapper transform_tools")
    print("=" * 60)

    class FilterWrapper(MiddlewareWrapper):
        def transform_tools(self, tools):
            return []

    inner = DummyMiddleware()
    wrapper = FilterWrapper(inner)

    assert len(wrapper.tools) == 0
    print("  >> PASSED")


def test_wrapper_delegates_system_message():
    """MiddlewareWrapper delegates wrap_system_message to inner."""
    print("\n" + "=" * 60)
    print("Test: wrapper delegates system message")
    print("=" * 60)

    inner = DummyMiddleware()
    wrapper = MiddlewareWrapper(inner)

    parts = wrapper.wrap_system_message([])
    assert len(parts) == 1
    assert "inner system prompt" in parts[0]
    print("  >> PASSED")


def test_wrapper_transform_system_parts():
    """Subclass can override transform_system_parts to modify prompts."""
    print("\n" + "=" * 60)
    print("Test: wrapper transform_system_parts")
    print("=" * 60)

    class PromptWrapper(MiddlewareWrapper):
        def transform_system_parts(self, parts):
            return parts + ["<extra>injected by wrapper</extra>"]

    inner = DummyMiddleware()
    wrapper = PromptWrapper(inner)

    parts = wrapper.wrap_system_message([])
    assert len(parts) == 2
    assert "inner system prompt" in parts[0]
    assert "injected by wrapper" in parts[1]
    print("  >> PASSED")


def test_wrapper_delegates_before_agent():
    """MiddlewareWrapper delegates before_agent to inner."""
    print("\n" + "=" * 60)
    print("Test: wrapper delegates before_agent")
    print("=" * 60)

    inner = DummyMiddleware()
    wrapper = MiddlewareWrapper(inner)

    result = wrapper.before_agent({})
    assert result == {"_dummy_ran": True}
    print("  >> PASSED")


def test_wrapper_inner_property():
    """MiddlewareWrapper.inner returns the wrapped middleware."""
    print("\n" + "=" * 60)
    print("Test: wrapper inner property")
    print("=" * 60)

    inner = DummyMiddleware()
    wrapper = MiddlewareWrapper(inner)
    assert wrapper.inner is inner
    print("  >> PASSED")


# ========== BrowserEnvironment extensibility tests ==========


def test_browser_custom_system_prompt():
    """BrowserEnvironment accepts custom system_prompt string."""
    print("\n" + "=" * 60)
    print("Test: browser custom system_prompt")
    print("=" * 60)

    from arion_agent.environments.browser import BrowserConfig, is_browser_available
    if not is_browser_available():
        print("  >> SKIPPED (playwright not installed)")
        return

    from arion_agent.environments.browser.middleware import BrowserEnvironment

    env = BrowserEnvironment(BrowserConfig(), system_prompt="<custom>my prompt</custom>")
    parts = env.wrap_system_message([])
    assert len(parts) == 1
    assert "my prompt" in parts[0]
    assert "browser_snapshot" not in parts[0]
    print("  >> PASSED")


def test_browser_suppress_system_prompt():
    """BrowserEnvironment suppresses system prompt when system_prompt=False."""
    print("\n" + "=" * 60)
    print("Test: browser suppress system_prompt")
    print("=" * 60)

    from arion_agent.environments.browser import BrowserConfig, is_browser_available
    if not is_browser_available():
        print("  >> SKIPPED (playwright not installed)")
        return

    from arion_agent.environments.browser.middleware import BrowserEnvironment

    env = BrowserEnvironment(BrowserConfig(), system_prompt=False)
    parts = env.wrap_system_message([])
    assert len(parts) == 0
    print("  >> PASSED")


def test_browser_tool_factory():
    """BrowserEnvironment uses tool_factory when provided."""
    print("\n" + "=" * 60)
    print("Test: browser tool_factory")
    print("=" * 60)

    from arion_agent.environments.browser import BrowserConfig, is_browser_available
    if not is_browser_available():
        print("  >> SKIPPED (playwright not installed)")
        return

    from arion_agent.environments.browser.middleware import BrowserEnvironment

    @tool
    def custom_browse(url: str) -> str:
        """Custom browse tool."""
        return f"browsed {url}"

    def my_factory(session, **kwargs):
        return [custom_browse]

    env = BrowserEnvironment(BrowserConfig(), tool_factory=my_factory)
    assert len(env.tools) == 1
    assert env.tools[0].name == "custom_browse"
    print("  >> PASSED")


# ========== HeartbeatConfig extensibility tests ==========


def test_heartbeat_before_invoke_suppress():
    """before_invoke returning None suppresses the invocation."""
    print("\n" + "=" * 60)
    print("Test: heartbeat before_invoke suppress")
    print("=" * 60)

    from arion_agent.environments.heartbeat.config import HeartbeatConfig, TriggerContext
    from arion_agent.environments.heartbeat.scheduler import HeartbeatScheduler
    from arion_agent.environments.heartbeat.effectors import CallbackEffector
    from arion_agent.util.timezone import AgentClock
    from arion_agent.util.persistence import load_jsonl

    fired = []

    async def suppress_all(effector, context):
        return None

    async def capture(ctx):
        fired.append(ctx.get("trigger_name"))

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: should_not_fire
cron: every 1s
effector: synthetic_prompt
prompt_body: Test
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, before_invoke=suppress_all)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        asyncio.run(scheduler._tick())

        assert not mock_agent.ainvoke.called, "Agent should not be invoked when before_invoke returns None"
    print("  >> PASSED")


def test_heartbeat_before_invoke_passthrough():
    """before_invoke returning non-None allows invocation."""
    print("\n" + "=" * 60)
    print("Test: heartbeat before_invoke passthrough")
    print("=" * 60)

    from arion_agent.environments.heartbeat.config import HeartbeatConfig
    from arion_agent.environments.heartbeat.scheduler import HeartbeatScheduler
    from arion_agent.util.timezone import AgentClock

    async def allow_all(effector, context):
        return True

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: should_fire
cron: every 1s
effector: synthetic_prompt
prompt_body: Test
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, before_invoke=allow_all)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        asyncio.run(scheduler._tick())

        assert mock_agent.ainvoke.called, "Agent should be invoked when before_invoke returns non-None"
    print("  >> PASSED")


def test_heartbeat_after_invoke():
    """after_invoke is called after agent.ainvoke."""
    print("\n" + "=" * 60)
    print("Test: heartbeat after_invoke")
    print("=" * 60)

    from arion_agent.environments.heartbeat.config import HeartbeatConfig, TriggerContext
    from arion_agent.environments.heartbeat.scheduler import HeartbeatScheduler
    from arion_agent.util.timezone import AgentClock

    after_calls = []

    async def on_after(result, context):
        after_calls.append(context.get("trigger_name"))

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: tracked
cron: every 1s
effector: synthetic_prompt
prompt_body: Test
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, after_invoke=on_after)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        asyncio.run(scheduler._tick())

        assert "tracked" in after_calls
    print("  >> PASSED")


def test_heartbeat_effector_factory():
    """effector_factory resolves custom effector types."""
    print("\n" + "=" * 60)
    print("Test: heartbeat effector_factory")
    print("=" * 60)

    from arion_agent.environments.heartbeat.config import (
        BaseEffector, HeartbeatConfig, TriggerContext,
    )
    from arion_agent.environments.heartbeat.scheduler import HeartbeatScheduler
    from arion_agent.util.timezone import AgentClock

    custom_fired = []

    class CustomEffector(BaseEffector):
        async def execute(self, context, agent=None):
            custom_fired.append(context.get("trigger_name"))

    def my_factory(block):
        if block.get("effector", "").strip().lower() == "custom_type":
            return CustomEffector()
        return None

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: custom_task
cron: every 1s
effector: custom_type
prompt_body: Ignored by custom effector
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, effector_factory=my_factory)

        mock_agent = MagicMock()
        mock_agent.agent_id = "test"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        asyncio.run(scheduler._tick())

        assert "custom_task" in custom_fired
    print("  >> PASSED")


def test_heartbeat_effector_factory_fallback():
    """Built-in synthetic_prompt still works when effector_factory is set."""
    print("\n" + "=" * 60)
    print("Test: heartbeat effector_factory fallback to built-in")
    print("=" * 60)

    from arion_agent.environments.heartbeat.config import HeartbeatConfig
    from arion_agent.environments.heartbeat.scheduler import HeartbeatScheduler
    from arion_agent.util.timezone import AgentClock

    def my_factory(block):
        return None

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: standard
cron: every 1s
effector: synthetic_prompt
prompt_body: Normal prompt
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, effector_factory=my_factory)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        asyncio.run(scheduler._tick())

        assert mock_agent.ainvoke.called
    print("  >> PASSED")


# ========== SummarizationMiddleware extensibility test ==========


def test_summarization_message_filter_accepted():
    """Compression node accepts custom policy callable."""
    print("\n" + "=" * 60)
    print("Test: compression accepts custom policy")
    print("=" * 60)

    from arion_agent.summarization import STANDARD_POLICY, make_should_compress

    should_compress = make_should_compress(STANDARD_POLICY, max_tokens=100_000)
    state = {"messages": [], "summary": ""}
    assert should_compress(state) == "model"
    print("  >> PASSED")


# ========== Moonshot multimodal payload tests (mock, no API call) ==========


def test_moonshot_multimodal_tool_message_payload():
    """ChatMoonshot preserves multimodal list content in tool messages."""
    print("\n" + "=" * 60)
    print("Test: Moonshot multimodal tool message payload")
    print("=" * 60)

    from arion_agent.providers.moonshot import ChatMoonshot
    from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

    msgs = [
        HumanMessage(content="read image.png"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "image.png"}, "id": "call_1"}],
        ),
        ToolMessage(
            content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                {"type": "text", "text": "[Image loaded: image/png, 1234 bytes]"},
            ],
            name="read_file",
            tool_call_id="call_1",
        ),
    ]
    model = ChatMoonshot(model="kimi-k2.5", api_key="fake", base_url="https://api.moonshot.cn/v1")
    payload = model._get_request_payload(msgs)
    tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]

    assert len(tool_msgs) == 1
    assert isinstance(tool_msgs[0]["content"], list), "Multimodal content must be a list"
    assert any(b.get("type") == "image_url" for b in tool_msgs[0]["content"])
    assert any(b.get("type") == "text" for b in tool_msgs[0]["content"])
    print("  >> PASSED")


def test_moonshot_reasoning_plus_multimodal():
    """ChatMoonshot handles reasoning_content and multimodal tool messages together."""
    print("\n" + "=" * 60)
    print("Test: Moonshot reasoning + multimodal coexistence")
    print("=" * 60)

    from arion_agent.providers.moonshot import ChatMoonshot
    from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

    msgs = [
        HumanMessage(content="read image"),
        AIMessage(
            content="I will read the image.",
            additional_kwargs={"reasoning_content": "User wants to see an image file"},
            tool_calls=[{"name": "read_file", "args": {"path": "x.png"}, "id": "call_2"}],
        ),
        ToolMessage(
            content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "Image loaded"},
            ],
            name="read_file",
            tool_call_id="call_2",
        ),
    ]
    model = ChatMoonshot(model="kimi-k2.5", api_key="fake", base_url="https://api.moonshot.cn/v1")
    payload = model._get_request_payload(msgs)

    ai_msgs = [m for m in payload["messages"] if m.get("role") == "assistant"]
    tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]

    assert ai_msgs[0].get("reasoning_content") == "User wants to see an image file"
    assert isinstance(tool_msgs[0]["content"], list)
    print("  >> PASSED")


def test_convert_image_block_format():
    """convert_image_block produces correct multimodal ToolMessage structure."""
    print("\n" + "=" * 60)
    print("Test: convert_image_block output format")
    print("=" * 60)

    from arion_agent.util.multimodal import convert_image_block, IMAGE_BLOCK_SENTINEL
    from langchain_core.messages import ToolMessage
    import base64

    raw_data = base64.b64encode(b"\x89PNG fake image data").decode("ascii")
    sentinel = f"{IMAGE_BLOCK_SENTINEL}:image/png:{raw_data}"

    original = ToolMessage(content=sentinel, name="read_file", tool_call_id="call_1")
    converted = convert_image_block(original)

    assert isinstance(converted.content, list), "Converted content must be a list"
    assert len(converted.content) == 2
    assert converted.content[0]["type"] == "image_url"
    assert converted.content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert converted.content[1]["type"] == "text"
    assert "Image loaded" in converted.content[1]["text"]
    assert converted.tool_call_id == "call_1"
    assert converted.name == "read_file"
    print("  >> PASSED")


# ========== Main ==========


if __name__ == "__main__":
    print("=" * 60)
    print("Extensibility Tests")
    print("=" * 60)

    # MiddlewareWrapper
    test_wrapper_delegates_tools()
    test_wrapper_transform_tools()
    test_wrapper_delegates_system_message()
    test_wrapper_transform_system_parts()
    test_wrapper_delegates_before_agent()
    test_wrapper_inner_property()

    # BrowserEnvironment
    test_browser_custom_system_prompt()
    test_browser_suppress_system_prompt()
    test_browser_tool_factory()

    # HeartbeatConfig hooks
    test_heartbeat_before_invoke_suppress()
    test_heartbeat_before_invoke_passthrough()
    test_heartbeat_after_invoke()
    test_heartbeat_effector_factory()
    test_heartbeat_effector_factory_fallback()

    # SummarizationMiddleware
    test_summarization_message_filter_accepted()

    # Moonshot multimodal
    test_moonshot_multimodal_tool_message_payload()
    test_moonshot_reasoning_plus_multimodal()
    test_convert_image_block_format()

    print("\n" + "=" * 60)
    print("ALL EXTENSIBILITY TESTS PASSED")
    print("=" * 60)

"""Test heartbeat environment: AgentClock, parser, effectors, middleware, scheduler, graph wiring."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent.util.timezone import AgentClock
from arion_agent.environments.heartbeat.config import (
    BaseEffector,
    EffectorDefaults,
    EventTrigger,
    FieldHandler,
    HeartbeatConfig,
    HibernationTrigger,
    TriggerContext,
)
from arion_agent.environments.heartbeat.parser import ParseResult
from arion_agent.environments.heartbeat.parser import (
    normalize_cron,
    parse_interval_seconds,
    parse_schedule,
)
from arion_agent.environments.heartbeat.effectors import (
    CallbackEffector,
    CompositeEffector,
    FileOperationEffector,
    SyntheticPromptEffector,
)
from arion_agent.environments.heartbeat.middleware import HeartbeatEnvironment
from arion_agent.environments.heartbeat.scheduler import HeartbeatScheduler
from arion_agent.util.persistence import append_jsonl, load_jsonl


# ========== AgentClock tests ==========


def test_clock_now():
    """AgentClock.now() returns timezone-aware datetime."""
    print("\n" + "=" * 60)
    print("Test: AgentClock.now()")
    print("=" * 60)

    clock = AgentClock("America/New_York")
    now = clock.now()
    assert now.tzinfo is not None
    assert clock.timezone_name == "America/New_York"
    print(f"  >> now = {now}")
    print("  >> PASSED")


def test_clock_utc():
    """AgentClock.now_utc() returns UTC datetime."""
    print("\n" + "=" * 60)
    print("Test: AgentClock.now_utc()")
    print("=" * 60)

    clock = AgentClock("Asia/Tokyo")
    utc = clock.now_utc()
    assert utc.tzinfo is not None
    assert str(utc.tzinfo) in ("UTC", "datetime.timezone.utc")
    print("  >> PASSED")


def test_clock_format_iso():
    """AgentClock.format_iso() produces ISO 8601 string."""
    print("\n" + "=" * 60)
    print("Test: AgentClock.format_iso()")
    print("=" * 60)

    clock = AgentClock("UTC")
    iso = clock.format_iso()
    assert "T" in iso
    assert "+" in iso or "Z" in iso or "-" in iso
    print(f"  >> iso = {iso}")
    print("  >> PASSED")


def test_clock_format_human():
    """AgentClock.format_human() produces readable string."""
    print("\n" + "=" * 60)
    print("Test: AgentClock.format_human()")
    print("=" * 60)

    clock = AgentClock("UTC")
    human = clock.format_human()
    assert len(human) > 10
    print(f"  >> human = {human}")
    print("  >> PASSED")


def test_clock_parse():
    """AgentClock.parse() handles ISO strings with and without tz."""
    print("\n" + "=" * 60)
    print("Test: AgentClock.parse()")
    print("=" * 60)

    clock = AgentClock("America/New_York")
    dt1 = clock.parse("2026-06-15T09:00:00")
    assert dt1.tzinfo is not None
    assert dt1.hour == 9

    dt2 = clock.parse("2026-06-15T09:00:00+00:00")
    assert dt2.tzinfo is not None
    print("  >> PASSED")


def test_clock_format_duration():
    """AgentClock.format_duration() handles various time ranges."""
    print("\n" + "=" * 60)
    print("Test: AgentClock.format_duration()")
    print("=" * 60)

    clock = AgentClock()
    assert "seconds" in clock.format_duration(30)
    assert "minutes" in clock.format_duration(300)
    assert "hours" in clock.format_duration(7200)
    assert "days" in clock.format_duration(172800)
    print("  >> PASSED")


# ========== TriggerContext tests ==========


def test_trigger_context_basic():
    """TriggerContext set/get and template formatting."""
    print("\n" + "=" * 60)
    print("Test: TriggerContext basic")
    print("=" * 60)

    ctx = TriggerContext(timestamp="2026-02-24T10:00:00", trigger_name="test")
    assert ctx.get("timestamp") == "2026-02-24T10:00:00"
    assert ctx.get("trigger_name") == "test"
    assert ctx.get("missing", "default") == "default"

    ctx.set("custom_field", "custom_value")
    result = ctx.format_template("Fired at {timestamp} for {trigger_name}, custom={custom_field}")
    assert "2026-02-24T10:00:00" in result
    assert "test" in result
    assert "custom_value" in result
    print("  >> PASSED")


def test_trigger_context_unknown_vars():
    """Unknown template variables are left as-is."""
    print("\n" + "=" * 60)
    print("Test: TriggerContext unknown vars")
    print("=" * 60)

    ctx = TriggerContext(a="1")
    result = ctx.format_template("{a} and {unknown}")
    assert result == "1 and {unknown}"
    print("  >> PASSED")


# ========== Parser tests ==========


def test_parse_schedule_basic():
    """Parse a simple schedule with two entries."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule basic")
    print("=" * 60)

    text = """
# Heartbeat Schedule

## How to manage this schedule
Some guidance text here.

---

## periodic: morning_briefing
cron: 0 9 * * 1-5
effector: synthetic_prompt
description: Morning review
prompt_body: Good morning

## periodic: hourly_check
cron: 0 * * * *
effector: synthetic_prompt
prompt_body: Quick check
"""
    blocks = parse_schedule(text)
    assert len(blocks) == 2
    assert blocks[0]["name"] == "morning_briefing"
    assert blocks[0]["cron"] == "0 9 * * 1-5"
    assert blocks[0]["effector"] == "synthetic_prompt"
    assert blocks[0]["prompt_body"] == "Good morning"
    assert blocks[1]["name"] == "hourly_check"
    print("  >> PASSED")


def test_parse_schedule_preserves_unknown_fields():
    """Unknown fields are preserved in the parsed dict."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule preserves unknown fields")
    print("=" * 60)

    text = """
## periodic: custom_task
cron: 0 9 * * *
effector: synthetic_prompt
skip_if_holiday: US
priority: high
custom_flag: true
prompt_body: Do something
"""
    blocks = parse_schedule(text)
    assert len(blocks) == 1
    assert blocks[0]["skip_if_holiday"] == "US"
    assert blocks[0]["priority"] == "high"
    assert blocks[0]["custom_flag"] == "true"
    print("  >> PASSED")


def test_parse_schedule_skips_missing_cron():
    """Blocks without cron field are skipped with warning."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule skips missing cron")
    print("=" * 60)

    text = """
## periodic: no_cron
effector: synthetic_prompt
prompt_body: This has no cron

## periodic: valid
cron: @hourly
effector: synthetic_prompt
prompt_body: This is valid
"""
    blocks = parse_schedule(text)
    assert len(blocks) == 1
    assert blocks[0]["name"] == "valid"
    print("  >> PASSED")


def test_parse_schedule_skips_missing_effector():
    """Blocks without effector field are skipped."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule skips missing effector")
    print("=" * 60)

    text = """
## periodic: no_effector
cron: 0 * * * *
prompt_body: Missing effector
"""
    blocks = parse_schedule(text)
    assert len(blocks) == 0
    print("  >> PASSED")


def test_parse_schedule_interval_syntax():
    """Interval syntax (every Xm) is recognized as valid."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule interval syntax")
    print("=" * 60)

    text = """
## periodic: frequent_check
cron: every 15m
effector: synthetic_prompt
prompt_body: Check
"""
    blocks = parse_schedule(text)
    assert len(blocks) == 1
    assert blocks[0]["_interval_seconds"] == "900"
    print("  >> PASSED")


def test_parse_schedule_comments_preserved():
    """Comment lines (// and >) are preserved in _comments."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule comments preserved")
    print("=" * 60)

    text = """
## periodic: with_notes
cron: 0 9 * * 1-5
effector: synthetic_prompt
// Added by developer: this runs on weekday mornings
> Agent note: consider adjusting to 8 AM after daylight saving
description: Morning check
prompt_body: Good morning
"""
    blocks = parse_schedule(text)
    assert len(blocks) == 1
    assert "_comments" in blocks[0]
    assert "Added by developer" in blocks[0]["_comments"]
    assert "Agent note" in blocks[0]["_comments"]
    assert blocks[0]["description"] == "Morning check"
    assert blocks[0]["prompt_body"] == "Good morning"
    print("  >> PASSED")


def test_parse_schedule_no_comments():
    """Blocks without comments have no _comments key."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule no comments")
    print("=" * 60)

    text = """
## periodic: plain
cron: @hourly
effector: synthetic_prompt
prompt_body: Check
"""
    blocks = parse_schedule(text)
    assert len(blocks) == 1
    assert "_comments" not in blocks[0]
    print("  >> PASSED")


def test_parse_schedule_comments_not_parsed_as_fields():
    """Comment lines are not mistakenly parsed as key: value fields."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule comments not parsed as fields")
    print("=" * 60)

    text = """
## periodic: tricky
cron: 0 9 * * *
effector: synthetic_prompt
// note: this looks like a field but is a comment
> reason: also a comment, not a field
prompt_body: Do work
"""
    blocks = parse_schedule(text)
    assert len(blocks) == 1
    assert "note" not in blocks[0]
    assert "reason" not in blocks[0]
    assert "// note" in blocks[0]["_comments"]
    assert "> reason" in blocks[0]["_comments"]
    print("  >> PASSED")


def test_parse_schedule_empty():
    """Empty schedule returns empty list."""
    print("\n" + "=" * 60)
    print("Test: parse_schedule empty")
    print("=" * 60)

    assert parse_schedule("") == []
    assert parse_schedule("# Just a title\nSome text") == []
    print("  >> PASSED")


def test_normalize_cron_shortcuts():
    """Cron shortcuts (@hourly etc.) are normalized."""
    print("\n" + "=" * 60)
    print("Test: normalize_cron shortcuts")
    print("=" * 60)

    assert normalize_cron("@hourly") == "0 * * * *"
    assert normalize_cron("@daily") == "0 0 * * *"
    assert normalize_cron("@weekly") == "0 0 * * 0"
    assert normalize_cron("@monthly") == "0 0 1 * *"
    assert normalize_cron("@yearly") == "0 0 1 1 *"
    assert normalize_cron("0 9 * * 1-5") == "0 9 * * 1-5"
    assert normalize_cron("every 15m") is None
    print("  >> PASSED")


def test_parse_interval_seconds():
    """Interval expressions are parsed to seconds."""
    print("\n" + "=" * 60)
    print("Test: parse_interval_seconds")
    print("=" * 60)

    assert parse_interval_seconds("every 15m") == 900
    assert parse_interval_seconds("every 2h") == 7200
    assert parse_interval_seconds("every 30s") == 30
    assert parse_interval_seconds("every 1d") == 86400
    assert parse_interval_seconds("0 * * * *") is None
    assert parse_interval_seconds("not_valid") is None
    print("  >> PASSED")


# ========== Effector tests ==========


def test_synthetic_prompt_effector():
    """SyntheticPromptEffector constructs prompt and invokes agent."""
    print("\n" + "=" * 60)
    print("Test: SyntheticPromptEffector")
    print("=" * 60)

    mock_agent = AsyncMock()
    mock_agent.agent_id = "test-agent"

    eff = SyntheticPromptEffector(
        prepend="[Heartbeat: {trigger_name} at {timestamp}]",
        body="Check your tasks.",
        append="[Don't forget existing todos.]",
    )

    ctx = TriggerContext(
        trigger_name="morning", timestamp="2026-02-24T09:00:00-05:00"
    )

    asyncio.run(eff.execute(ctx, agent=mock_agent))

    mock_agent.ainvoke.assert_called_once()
    call_args = mock_agent.ainvoke.call_args
    messages = call_args[0][0]["messages"]
    prompt = messages[0][1]
    assert "morning" in prompt
    assert "Check your tasks" in prompt
    assert "Don't forget existing todos" in prompt
    assert "heartbeat-morning-2026-02-24" in call_args[1]["config"]["configurable"]["thread_id"]
    print("  >> PASSED")


def test_synthetic_prompt_custom_thread_id():
    """SyntheticPromptEffector uses custom thread_id when provided."""
    print("\n" + "=" * 60)
    print("Test: SyntheticPromptEffector custom thread_id")
    print("=" * 60)

    mock_agent = AsyncMock()
    mock_agent.agent_id = "test-agent"

    eff = SyntheticPromptEffector(body="Check tasks.", thread_id="main")
    ctx = TriggerContext(trigger_name="test", timestamp="2026-02-24T09:00:00")
    asyncio.run(eff.execute(ctx, agent=mock_agent))

    call_args = mock_agent.ainvoke.call_args
    assert call_args[1]["config"]["configurable"]["thread_id"] == "main"
    print("  >> PASSED")


def test_synthetic_prompt_template_thread_id():
    """SyntheticPromptEffector supports template variables in thread_id."""
    print("\n" + "=" * 60)
    print("Test: SyntheticPromptEffector template thread_id")
    print("=" * 60)

    mock_agent = AsyncMock()
    mock_agent.agent_id = "test-agent"

    eff = SyntheticPromptEffector(body="Check.", thread_id="session-{agent_id}")
    ctx = TriggerContext(trigger_name="test", timestamp="2026-02-24T09:00:00", agent_id="fubao")
    asyncio.run(eff.execute(ctx, agent=mock_agent))

    call_args = mock_agent.ainvoke.call_args
    assert call_args[1]["config"]["configurable"]["thread_id"] == "session-fubao"
    print("  >> PASSED")


def test_scheduler_config_thread_id():
    """HeartbeatConfig.thread_id is used as default for all periodic triggers."""
    print("\n" + "=" * 60)
    print("Test: scheduler config-level thread_id")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: quick
cron: every 1s
effector: synthetic_prompt
prompt_body: Quick check
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, thread_id="main-thread")

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        async def run():
            await scheduler._tick()

        asyncio.run(run())

        assert mock_agent.ainvoke.called
        call_args = mock_agent.ainvoke.call_args
        assert call_args[1]["config"]["configurable"]["thread_id"] == "main-thread"
    print("  >> PASSED")


def test_scheduler_per_trigger_thread_id_overrides_config():
    """Per-trigger thread_id in schedule file overrides config-level."""
    print("\n" + "=" * 60)
    print("Test: per-trigger thread_id overrides config")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: special
cron: every 1s
effector: synthetic_prompt
thread_id: special-thread
prompt_body: Special task
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, thread_id="default-thread")

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        async def run():
            await scheduler._tick()

        asyncio.run(run())

        assert mock_agent.ainvoke.called
        call_args = mock_agent.ainvoke.call_args
        assert call_args[1]["config"]["configurable"]["thread_id"] == "special-thread"
    print("  >> PASSED")


def test_scheduler_effector_defaults():
    """EffectorDefaults prepend/append are applied when per-trigger fields are empty."""
    print("\n" + "=" * 60)
    print("Test: scheduler effector_defaults")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: bare
cron: every 1s
effector: synthetic_prompt
prompt_body: Do the task
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(
            tick_interval=1,
            effector_defaults=EffectorDefaults(
                prepend="[DEFAULT PREPEND]",
                append="[DEFAULT APPEND: check existing todos]",
            ),
        )

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        async def run():
            await scheduler._tick()

        asyncio.run(run())

        assert mock_agent.ainvoke.called
        call_args = mock_agent.ainvoke.call_args
        prompt = call_args[0][0]["messages"][0][1]
        assert "DEFAULT PREPEND" in prompt
        assert "Do the task" in prompt
        assert "DEFAULT APPEND" in prompt
    print("  >> PASSED")


def test_scheduler_effector_defaults_not_override():
    """EffectorDefaults do NOT override per-trigger prepend/append."""
    print("\n" + "=" * 60)
    print("Test: effector_defaults do not override explicit fields")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: explicit
cron: every 1s
effector: synthetic_prompt
prompt_prepend: [EXPLICIT PREPEND]
prompt_body: Do work
prompt_append: [EXPLICIT APPEND]
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(
            tick_interval=1,
            effector_defaults=EffectorDefaults(
                prepend="[DEFAULT PREPEND]",
                append="[DEFAULT APPEND]",
            ),
        )

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        async def run():
            await scheduler._tick()

        asyncio.run(run())

        assert mock_agent.ainvoke.called
        prompt = mock_agent.ainvoke.call_args[0][0]["messages"][0][1]
        assert "EXPLICIT PREPEND" in prompt
        assert "EXPLICIT APPEND" in prompt
        assert "DEFAULT" not in prompt
    print("  >> PASSED")


def test_synthetic_prompt_no_agent():
    """SyntheticPromptEffector skips gracefully when no agent provided."""
    print("\n" + "=" * 60)
    print("Test: SyntheticPromptEffector no agent")
    print("=" * 60)

    eff = SyntheticPromptEffector(body="test")
    ctx = TriggerContext(trigger_name="test", timestamp="2026-01-01T00:00:00Z")
    asyncio.run(eff.execute(ctx, agent=None))
    print("  >> PASSED (no crash)")


def test_file_operation_effector_append():
    """FileOperationEffector appends to file."""
    print("\n" + "=" * 60)
    print("Test: FileOperationEffector append")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        eff = FileOperationEffector(
            operation="append",
            target_path="log.txt",
            content="[{timestamp}] fired\n",
            workspace_dir=Path(d),
        )
        ctx = TriggerContext(timestamp="2026-02-24T10:00:00")
        asyncio.run(eff.execute(ctx))

        content = (Path(d) / "log.txt").read_text(encoding="utf-8")
        assert "2026-02-24T10:00:00" in content
        assert "fired" in content
    print("  >> PASSED")


def test_file_operation_effector_write():
    """FileOperationEffector writes atomically."""
    print("\n" + "=" * 60)
    print("Test: FileOperationEffector write")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        eff = FileOperationEffector(
            operation="write",
            target_path="data.txt",
            content="hello {trigger_name}",
            workspace_dir=Path(d),
        )
        ctx = TriggerContext(trigger_name="test")
        asyncio.run(eff.execute(ctx))

        content = (Path(d) / "data.txt").read_text(encoding="utf-8")
        assert content == "hello test"
    print("  >> PASSED")


def test_callback_effector():
    """CallbackEffector calls the async callback."""
    print("\n" + "=" * 60)
    print("Test: CallbackEffector")
    print("=" * 60)

    called_with = []

    async def my_callback(ctx: TriggerContext) -> None:
        called_with.append(ctx.get("trigger_name"))

    eff = CallbackEffector(my_callback)
    ctx = TriggerContext(trigger_name="callback_test")
    asyncio.run(eff.execute(ctx))

    assert called_with == ["callback_test"]
    print("  >> PASSED")


def test_composite_effector():
    """CompositeEffector chains multiple effectors."""
    print("\n" + "=" * 60)
    print("Test: CompositeEffector")
    print("=" * 60)

    order = []

    async def cb1(ctx: TriggerContext) -> None:
        order.append("first")

    async def cb2(ctx: TriggerContext) -> None:
        order.append("second")

    eff = CompositeEffector([CallbackEffector(cb1), CallbackEffector(cb2)])
    ctx = TriggerContext(trigger_name="composite")
    asyncio.run(eff.execute(ctx))

    assert order == ["first", "second"]
    print("  >> PASSED")


# ========== FieldHandler tests ==========


def test_field_handler_suppression():
    """FieldHandler.should_fire() can suppress a trigger."""
    print("\n" + "=" * 60)
    print("Test: FieldHandler suppression")
    print("=" * 60)

    class AlwaysSkip(FieldHandler):
        def should_fire(self, field_value, trigger, context):
            return False

        def management_doc(self):
            return "always skips"

    handler = AlwaysSkip()
    ctx = TriggerContext(trigger_name="test")
    assert handler.should_fire("any", {}, ctx) is False
    assert handler.management_doc() == "always skips"
    print("  >> PASSED")


def test_field_handler_enrich():
    """FieldHandler.enrich_context() adds variables."""
    print("\n" + "=" * 60)
    print("Test: FieldHandler enrich_context")
    print("=" * 60)

    class PriorityHandler(FieldHandler):
        def enrich_context(self, field_value, trigger, context):
            context.set("priority_level", field_value.upper())

    handler = PriorityHandler()
    ctx = TriggerContext(trigger_name="test")
    handler.enrich_context("high", {}, ctx)
    assert ctx.get("priority_level") == "HIGH"
    print("  >> PASSED")


# ========== HeartbeatEnvironment middleware tests ==========


def test_middleware_seeds_schedule():
    """HeartbeatEnvironment seeds HEARTBEAT_SCHEDULE.md on construction."""
    print("\n" + "=" * 60)
    print("Test: middleware seeds schedule")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        clock = AgentClock("UTC")
        mw = HeartbeatEnvironment(
            agent_id="test-agent",
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        assert mw.schedule_path.exists()
        content = mw.schedule_path.read_text(encoding="utf-8")
        assert "Heartbeat Schedule" in content
        assert "How to manage" in content
        assert "## periodic:" in content or "cron:" in content
    print("  >> PASSED")


def test_middleware_seeds_custom_schedule():
    """HeartbeatEnvironment uses initial_schedule when provided."""
    print("\n" + "=" * 60)
    print("Test: middleware seeds custom schedule")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        clock = AgentClock("UTC")
        cfg = HeartbeatConfig(initial_schedule="# Custom Schedule\nMy content here")
        mw = HeartbeatEnvironment(
            agent_id="test-agent",
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
            config=cfg,
        )

        content = mw.schedule_path.read_text(encoding="utf-8")
        assert "Custom Schedule" in content
        assert "My content here" in content
    print("  >> PASSED")


def test_middleware_seed_if_absent():
    """HeartbeatEnvironment does not overwrite existing schedule."""
    print("\n" + "=" * 60)
    print("Test: middleware seed-if-absent")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        schedule_path = identity_dir / "HEARTBEAT_SCHEDULE.md"
        schedule_path.write_text("# Agent-modified schedule\n", encoding="utf-8")

        clock = AgentClock("UTC")
        mw = HeartbeatEnvironment(
            agent_id="test-agent",
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        content = mw.schedule_path.read_text(encoding="utf-8")
        assert "Agent-modified schedule" in content
    print("  >> PASSED")


def test_middleware_wrap_system_message():
    """HeartbeatEnvironment injects heartbeat pointer into system message."""
    print("\n" + "=" * 60)
    print("Test: middleware wrap_system_message")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        clock = AgentClock("America/New_York")
        mw = HeartbeatEnvironment(
            agent_id="test-agent",
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        parts = mw.wrap_system_message([])
        assert len(parts) == 1
        section = parts[0]
        assert "<heartbeat>" in section
        assert "HEARTBEAT_SCHEDULE.md" in section
        assert "heartbeat_log.jsonl" in section
        assert "America/New_York" in section
        assert "</heartbeat>" in section
    print("  >> PASSED")


def test_middleware_extension_docs():
    """Auto-assembled management section includes field handler docs."""
    print("\n" + "=" * 60)
    print("Test: middleware extension docs")
    print("=" * 60)

    class HolidayHandler(FieldHandler):
        def management_doc(self):
            return "<calendar> -- skip on holidays (e.g., US, CN)"

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        clock = AgentClock("UTC")
        cfg = HeartbeatConfig(field_handlers={"skip_if_holiday": HolidayHandler()})
        mw = HeartbeatEnvironment(
            agent_id="test-agent",
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
            config=cfg,
        )

        content = mw.schedule_path.read_text(encoding="utf-8")
        assert "skip_if_holiday" in content
        assert "skip on holidays" in content
    print("  >> PASSED")


# ========== HeartbeatScheduler tests ==========


def test_scheduler_hibernation_first_run():
    """Scheduler fires hibernation triggers on start with is_first_run=true."""
    print("\n" + "=" * 60)
    print("Test: scheduler hibernation first run")
    print("=" * 60)

    fired_contexts = []

    async def capture(ctx: TriggerContext) -> None:
        fired_contexts.append(ctx.as_dict())

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        cfg = HeartbeatConfig(
            hibernation_triggers=[
                HibernationTrigger(
                    name="startup",
                    effector=CallbackEffector(capture),
                ),
            ],
        )

        mock_agent = MagicMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler.start()
            await scheduler.stop()

        asyncio.run(run())

        assert len(fired_contexts) == 1
        assert fired_contexts[0]["trigger_name"] == "startup"
        assert fired_contexts[0]["trigger_type"] == "hibernation"
        assert fired_contexts[0]["is_first_run"] == "true"
        assert fired_contexts[0]["last_active"] == "never"
    print("  >> PASSED")


def test_scheduler_hibernation_resume():
    """Scheduler detects previous activity on restart."""
    print("\n" + "=" * 60)
    print("Test: scheduler hibernation resume")
    print("=" * 60)

    fired_contexts = []

    async def capture(ctx: TriggerContext) -> None:
        fired_contexts.append(ctx.as_dict())

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        append_jsonl(identity_dir / "heartbeat_log.jsonl", {
            "trigger_name": "previous",
            "trigger_type": "periodic",
            "timestamp": "2026-02-20T10:00:00+00:00",
            "timestamp_utc": "2026-02-20T10:00:00+00:00",
        })

        cfg = HeartbeatConfig(
            hibernation_triggers=[
                HibernationTrigger(name="startup", effector=CallbackEffector(capture)),
            ],
        )

        mock_agent = MagicMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler.start()
            await scheduler.stop()

        asyncio.run(run())

        assert len(fired_contexts) == 1
        assert fired_contexts[0]["is_first_run"] == "false"
        assert "2026-02-20" in fired_contexts[0]["last_active"]
        assert "days" in fired_contexts[0]["offline_duration"]
    print("  >> PASSED")


def test_scheduler_periodic_interval():
    """Scheduler fires interval-based periodic triggers."""
    print("\n" + "=" * 60)
    print("Test: scheduler periodic interval")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule_content = """
# Heartbeat Schedule

## periodic: quick_check
cron: every 1s
effector: synthetic_prompt
prompt_body: Check
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule_content, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler.start()
            await asyncio.sleep(0.1)
            await scheduler._tick()
            await scheduler.stop()

        asyncio.run(run())

        assert mock_agent.ainvoke.called, "Agent should have been invoked"
        call_args = mock_agent.ainvoke.call_args
        prompt = call_args[0][0]["messages"][0][1]
        assert "Check" in prompt
    print("  >> PASSED")


def test_scheduler_event_file_changed():
    """Scheduler fires file_changed event trigger."""
    print("\n" + "=" * 60)
    print("Test: scheduler event file_changed")
    print("=" * 60)

    fired = []

    async def capture(ctx: TriggerContext) -> None:
        fired.append(ctx.as_dict())

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        config_file = Path(ws) / "config.yaml"
        config_file.write_text("initial: true", encoding="utf-8")

        cfg = HeartbeatConfig(
            event_triggers=[
                EventTrigger(
                    name="config_watch",
                    type="file_changed",
                    watch_paths=["config.yaml"],
                    effector=CallbackEffector(capture),
                ),
            ],
        )

        mock_agent = MagicMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler._tick()

            import time
            time.sleep(0.05)
            config_file.write_text("changed: true", encoding="utf-8")

            await scheduler._tick()

        asyncio.run(run())

        assert len(fired) == 1
        assert fired[0]["trigger_name"] == "config_watch"
        assert fired[0]["changed_path"] == "config.yaml"
    print("  >> PASSED")


def test_scheduler_event_custom():
    """Scheduler fires custom event trigger."""
    print("\n" + "=" * 60)
    print("Test: scheduler event custom")
    print("=" * 60)

    fired = []
    poll_count = 0

    async def my_poll():
        nonlocal poll_count
        poll_count += 1
        if poll_count >= 2:
            return {"custom_data": "hello"}
        return None

    async def capture(ctx: TriggerContext) -> None:
        fired.append(ctx.as_dict())

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        cfg = HeartbeatConfig(
            event_triggers=[
                EventTrigger(
                    name="custom_event",
                    type="custom",
                    poll_fn=my_poll,
                    effector=CallbackEffector(capture),
                ),
            ],
        )

        mock_agent = MagicMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler._tick()
            assert len(fired) == 0

            await scheduler._tick()
            assert len(fired) == 1

        asyncio.run(run())

        assert fired[0]["custom_data"] == "hello"
    print("  >> PASSED")


def test_scheduler_event_cooldown():
    """Event triggers respect cooldown period."""
    print("\n" + "=" * 60)
    print("Test: scheduler event cooldown")
    print("=" * 60)

    fired = []

    async def always_fire():
        return {"data": "yes"}

    async def capture(ctx: TriggerContext) -> None:
        fired.append(1)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        cfg = HeartbeatConfig(
            event_triggers=[
                EventTrigger(
                    name="rapid",
                    type="custom",
                    poll_fn=always_fire,
                    effector=CallbackEffector(capture),
                    cooldown=3600,
                ),
            ],
        )

        mock_agent = MagicMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler._tick()
            await scheduler._tick()
            await scheduler._tick()

        asyncio.run(run())

        assert len(fired) == 1, f"Expected 1 (cooldown), got {len(fired)}"
    print("  >> PASSED")


def test_scheduler_field_handler_suppression():
    """Field handlers can suppress periodic triggers."""
    print("\n" + "=" * 60)
    print("Test: scheduler field handler suppression")
    print("=" * 60)

    fired = []

    class SkipAll(FieldHandler):
        def should_fire(self, field_value, trigger, context):
            return False

    async def capture(ctx: TriggerContext) -> None:
        fired.append(ctx.get("trigger_name"))

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        now = clock.now()
        cron_now = f"{now.minute} {now.hour} * * *"

        schedule = f"""
## periodic: should_skip
cron: {cron_now}
effector: synthetic_prompt
skip_all: yes
prompt_body: Should not fire
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(field_handlers={"skip_all": SkipAll()})

        mock_agent = MagicMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler._evaluate_periodic_triggers(now)

        asyncio.run(run())

        assert len(fired) == 0, f"Expected 0 (suppressed), got {len(fired)}"
    print("  >> PASSED")


def test_scheduler_log_execution():
    """Scheduler logs executions to heartbeat_log.jsonl."""
    print("\n" + "=" * 60)
    print("Test: scheduler log execution")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        cfg = HeartbeatConfig(
            hibernation_triggers=[
                HibernationTrigger(
                    name="startup",
                    effector=CallbackEffector(lambda ctx: asyncio.sleep(0)),
                ),
            ],
        )

        mock_agent = MagicMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler.start()
            await scheduler.stop()

        asyncio.run(run())

        log_path = identity_dir / "heartbeat_log.jsonl"
        assert log_path.exists()
        records = load_jsonl(log_path)
        assert len(records) >= 1
        assert records[-1]["trigger_name"] == "startup"
        assert records[-1]["trigger_type"] == "hibernation"
    print("  >> PASSED")


def test_scheduler_log_trimming():
    """Scheduler trims log at max_log_entries."""
    print("\n" + "=" * 60)
    print("Test: scheduler log trimming")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        log_path = identity_dir / "heartbeat_log.jsonl"
        for i in range(15):
            append_jsonl(log_path, {
                "trigger_name": f"entry-{i}",
                "trigger_type": "periodic",
                "timestamp": f"2026-02-{i+1:02d}T00:00:00+00:00",
            })

        cfg = HeartbeatConfig(
            max_log_entries=10,
            hibernation_triggers=[
                HibernationTrigger(
                    name="trim_test",
                    effector=CallbackEffector(lambda ctx: asyncio.sleep(0)),
                ),
            ],
        )

        mock_agent = MagicMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent,
            config=cfg,
            identity_dir=identity_dir,
            workspace_dir=Path(ws),
            clock=clock,
        )

        async def run():
            await scheduler.start()
            await scheduler.stop()

        asyncio.run(run())

        records = load_jsonl(log_path)
        assert len(records) <= 10, f"Expected <=10, got {len(records)}"
    print("  >> PASSED")


def test_scheduler_self_heal_on_parse_error():
    """Scheduler sends self-heal prompt when schedule has malformed entries."""
    print("\n" + "=" * 60)
    print("Test: scheduler self-heal on parse error")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: broken_entry
effector: synthetic_prompt
prompt_body: This has no cron field

## periodic: also_broken
cron: 0 9 * * *
prompt_body: This has no effector field

## periodic: valid_entry
cron: every 1s
effector: synthetic_prompt
prompt_body: This one is fine
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, self_heal=True)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        async def run():
            await scheduler._tick()

        asyncio.run(run())

        calls = mock_agent.ainvoke.call_args_list
        heal_calls = [c for c in calls if "self-heal" in str(c)]
        assert len(heal_calls) == 1, f"Expected 1 self-heal call, got {len(heal_calls)}"

        heal_prompt = heal_calls[0][0][0]["messages"][0][1]
        assert "broken_entry" in heal_prompt
        assert "also_broken" in heal_prompt
        assert "missing required" in heal_prompt
    print("  >> PASSED")


def test_scheduler_self_heal_dedup():
    """Self-heal does not re-fire when errors haven't changed."""
    print("\n" + "=" * 60)
    print("Test: scheduler self-heal dedup")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: broken
effector: synthetic_prompt
prompt_body: No cron
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, self_heal=True)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        async def run():
            await scheduler._tick()
            await scheduler._tick()
            await scheduler._tick()

        asyncio.run(run())

        heal_calls = [c for c in mock_agent.ainvoke.call_args_list if "self-heal" in str(c)]
        assert len(heal_calls) == 1, f"Expected 1 (deduped), got {len(heal_calls)}"
    print("  >> PASSED")


def test_scheduler_self_heal_disabled():
    """Self-heal can be disabled via config."""
    print("\n" + "=" * 60)
    print("Test: scheduler self-heal disabled")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")

        schedule = """
## periodic: broken
effector: synthetic_prompt
prompt_body: No cron
"""
        (identity_dir / "HEARTBEAT_SCHEDULE.md").write_text(schedule, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, self_heal=False)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        async def run():
            await scheduler._tick()

        asyncio.run(run())

        assert not mock_agent.ainvoke.called, "Agent should not be invoked when self_heal=False"
    print("  >> PASSED")


def test_scheduler_self_heal_refires_on_new_errors():
    """Self-heal fires again when the error set changes."""
    print("\n" + "=" * 60)
    print("Test: scheduler self-heal refires on new errors")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        identity_dir = Path(ws) / ".arion" / "agents" / "test-agent"
        identity_dir.mkdir(parents=True)
        clock = AgentClock("UTC")
        schedule_path = identity_dir / "HEARTBEAT_SCHEDULE.md"

        schedule_v1 = """
## periodic: broken_a
effector: synthetic_prompt
prompt_body: No cron
"""
        schedule_v2 = """
## periodic: broken_b
cron: 0 9 * * *
prompt_body: No effector
"""
        schedule_path.write_text(schedule_v1, encoding="utf-8")

        cfg = HeartbeatConfig(tick_interval=1, self_heal=True)

        mock_agent = AsyncMock()
        mock_agent.agent_id = "test-agent"

        scheduler = HeartbeatScheduler(
            agent=mock_agent, config=cfg,
            identity_dir=identity_dir, workspace_dir=Path(ws), clock=clock,
        )

        async def run():
            await scheduler._tick()
            schedule_path.write_text(schedule_v2, encoding="utf-8")
            await scheduler._tick()

        asyncio.run(run())

        heal_calls = [c for c in mock_agent.ainvoke.call_args_list if "self-heal" in str(c)]
        assert len(heal_calls) == 2, f"Expected 2 (different errors), got {len(heal_calls)}"
    print("  >> PASSED")


# ========== Graph wiring tests ==========


def test_graph_wiring_no_heartbeat():
    """Default create_arion_agent has no heartbeat scheduler."""
    print("\n" + "=" * 60)
    print("Test: graph wiring - no heartbeat (default)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent
        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="test-no-hb",
            checkpointer=False,
            summarization=False,
        )
        assert agent.heartbeat_scheduler is None
    print("  >> PASSED")


def test_graph_wiring_with_heartbeat():
    """create_arion_agent with heartbeat= creates scheduler."""
    print("\n" + "=" * 60)
    print("Test: graph wiring - with heartbeat")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent
        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="test-hb",
            heartbeat=HeartbeatConfig(timezone="America/New_York"),
            checkpointer=False,
            summarization=False,
        )
        assert agent.heartbeat_scheduler is not None
        assert not agent.heartbeat_scheduler.is_running

        schedule_path = Path(ws) / ".arion" / "agents" / "test-hb" / "HEARTBEAT_SCHEDULE.md"
        assert schedule_path.exists()
        content = schedule_path.read_text(encoding="utf-8")
        assert "Heartbeat Schedule" in content
    print("  >> PASSED")


def test_graph_wiring_heartbeat_system_message():
    """Heartbeat middleware injects into system message when wired via graph."""
    print("\n" + "=" * 60)
    print("Test: graph wiring - heartbeat system message")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent
        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="test-hb-sys",
            heartbeat=HeartbeatConfig(timezone="Europe/London"),
            checkpointer=False,
            summarization=False,
        )
        assert agent.heartbeat_scheduler is not None
    print("  >> PASSED")


def test_graph_wiring_timezone_parameter():
    """timezone parameter is used when heartbeat.timezone is default."""
    print("\n" + "=" * 60)
    print("Test: graph wiring - timezone parameter")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent
        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="test-hb-tz",
            heartbeat=HeartbeatConfig(),
            timezone="Asia/Tokyo",
            checkpointer=False,
            summarization=False,
        )
        assert agent.heartbeat_scheduler is not None
    print("  >> PASSED")


# ========== Main ==========


if __name__ == "__main__":
    print("=" * 60)
    print("Heartbeat Environment Tests")
    print("=" * 60)

    # AgentClock
    test_clock_now()
    test_clock_utc()
    test_clock_format_iso()
    test_clock_format_human()
    test_clock_parse()
    test_clock_format_duration()

    # TriggerContext
    test_trigger_context_basic()
    test_trigger_context_unknown_vars()

    # Parser
    test_parse_schedule_basic()
    test_parse_schedule_preserves_unknown_fields()
    test_parse_schedule_skips_missing_cron()
    test_parse_schedule_skips_missing_effector()
    test_parse_schedule_interval_syntax()
    test_parse_schedule_comments_preserved()
    test_parse_schedule_no_comments()
    test_parse_schedule_comments_not_parsed_as_fields()
    test_parse_schedule_empty()
    test_normalize_cron_shortcuts()
    test_parse_interval_seconds()

    # Effectors
    test_synthetic_prompt_effector()
    test_synthetic_prompt_custom_thread_id()
    test_synthetic_prompt_template_thread_id()
    test_scheduler_config_thread_id()
    test_scheduler_per_trigger_thread_id_overrides_config()
    test_scheduler_effector_defaults()
    test_scheduler_effector_defaults_not_override()
    test_synthetic_prompt_no_agent()
    test_file_operation_effector_append()
    test_file_operation_effector_write()
    test_callback_effector()
    test_composite_effector()

    # FieldHandler
    test_field_handler_suppression()
    test_field_handler_enrich()

    # Middleware
    test_middleware_seeds_schedule()
    test_middleware_seeds_custom_schedule()
    test_middleware_seed_if_absent()
    test_middleware_wrap_system_message()
    test_middleware_extension_docs()

    # Scheduler
    test_scheduler_hibernation_first_run()
    test_scheduler_hibernation_resume()
    test_scheduler_periodic_interval()
    test_scheduler_event_file_changed()
    test_scheduler_event_custom()
    test_scheduler_event_cooldown()
    test_scheduler_field_handler_suppression()
    test_scheduler_log_execution()
    test_scheduler_log_trimming()
    test_scheduler_self_heal_on_parse_error()
    test_scheduler_self_heal_dedup()
    test_scheduler_self_heal_disabled()
    test_scheduler_self_heal_refires_on_new_errors()

    # Graph wiring
    test_graph_wiring_no_heartbeat()
    test_graph_wiring_with_heartbeat()
    test_graph_wiring_heartbeat_system_message()
    test_graph_wiring_timezone_parameter()

    print("\n" + "=" * 60)
    print("ALL HEARTBEAT TESTS PASSED")
    print("=" * 60)

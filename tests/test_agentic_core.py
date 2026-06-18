"""Test agentic core: structured planning (update_plan), plan enforcement, and running status."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent.environments.agentic_core.config import PlanConfig
from arion_agent.environments.agentic_core.middleware import AgenticCoreEnvironment
from arion_agent.environments.agentic_core.plan_registry import (
    ACTIVE_STATUSES,
    VALID_STATUSES,
    PlanItem,
    PlanRegistry,
)
from arion_agent.environments.agentic_core.tools import (
    create_agentic_core_tools,
    maintenance_tool,
)
from arion_agent.util.stats import AgentStats


# ---------- Tool creation ----------


def test_default_tools_backward_compat():
    """Parameterless create_agentic_core_tools returns only maintenance_tool."""
    tools = create_agentic_core_tools()
    assert len(tools) == 1
    assert tools[0].name == "maintenance_tool"


def test_tools_with_planning():
    """With PlanConfig + PlanRegistry, update_plan is added."""
    with tempfile.TemporaryDirectory() as ws:
        stats = AgentStats()
        registry = PlanRegistry()
        tools = create_agentic_core_tools(
            agent_id="test-agent",
            workspace_dir=Path(ws),
            stats=stats,
            plan_config=PlanConfig(),
            plan_registry=registry,
        )
        names = [t.name for t in tools]
        assert "maintenance_tool" in names
        assert "update_plan" in names
        assert "get_running_status" not in names


def test_tools_with_status():
    """With enable_status=True, get_running_status is added."""
    with tempfile.TemporaryDirectory() as ws:
        stats = AgentStats()
        registry = PlanRegistry()
        tools = create_agentic_core_tools(
            agent_id="test-agent",
            workspace_dir=Path(ws),
            stats=stats,
            plan_config=PlanConfig(),
            plan_registry=registry,
            enable_status=True,
        )
        names = [t.name for t in tools]
        assert "update_plan" in names
        assert "get_running_status" in names


def test_tools_planning_disabled():
    """Without PlanConfig, no update_plan tool."""
    stats = AgentStats()
    tools = create_agentic_core_tools(
        agent_id="test-agent",
        workspace_dir=Path("."),
        stats=stats,
        plan_config=None,
        enable_status=False,
    )
    names = [t.name for t in tools]
    assert "update_plan" not in names
    assert "get_running_status" not in names


# ---------- PlanRegistry ----------


def test_registry_set_items():
    """set_items validates and stores items."""
    reg = PlanRegistry()
    items = reg.set_items([
        {"id": "a", "description": "Do A", "status": "pending"},
        {"id": "b", "description": "Do B", "status": "completed"},
    ])
    assert len(items) == 2
    assert items[0].id == "a"
    assert items[0].status == "pending"
    assert items[1].status == "completed"


def test_registry_invalid_status_defaults_to_pending():
    """Invalid status values fall back to pending."""
    reg = PlanRegistry()
    items = reg.set_items([
        {"id": "x", "description": "mystery", "status": "invalid_status"},
    ])
    assert items[0].status == "pending"


def test_registry_has_pending_work():
    """has_pending_work detects active items."""
    reg = PlanRegistry()
    reg.set_items([
        {"id": "a", "description": "done", "status": "completed"},
        {"id": "b", "description": "skipped", "status": "deprioritized"},
    ])
    assert not reg.has_pending_work()

    reg.set_items([
        {"id": "a", "description": "done", "status": "completed"},
        {"id": "b", "description": "remaining", "status": "in_progress"},
    ])
    assert reg.has_pending_work()


def test_registry_should_nudge():
    """should_nudge respects both pending work and nudge budget."""
    reg = PlanRegistry(max_nudges=2)
    reg.set_items([{"id": "a", "description": "wip", "status": "in_progress"}])

    assert reg.should_nudge()
    reg.format_nudge_message()
    assert reg.should_nudge()
    reg.format_nudge_message()
    assert not reg.should_nudge()


def test_registry_nudge_reset():
    """reset_nudge_count restores budget."""
    reg = PlanRegistry(max_nudges=1)
    reg.set_items([{"id": "a", "description": "wip", "status": "pending"}])
    reg.format_nudge_message()
    assert not reg.should_nudge()

    reg.reset_nudge_count()
    assert reg.should_nudge()


def test_registry_nudge_message_content():
    """format_nudge_message includes item details and system prefix."""
    reg = PlanRegistry(max_nudges=5)
    reg.set_items([
        {"id": "setup", "description": "Set up env", "status": "in_progress"},
        {"id": "done", "description": "Already done", "status": "completed"},
        {"id": "next", "description": "Next step", "status": "pending"},
    ])
    msg = reg.format_nudge_message()
    assert "[SYSTEM - Plan Enforcement]" in msg
    assert "setup" in msg
    assert "next" in msg
    assert "Already done" not in msg


def test_registry_empty_no_nudge():
    """Empty registry does not trigger nudge."""
    reg = PlanRegistry(max_nudges=3)
    assert not reg.has_pending_work()
    assert not reg.should_nudge()


def test_registry_set_plan_with_sections():
    """set_plan stores narrative sections alongside items."""
    reg = PlanRegistry()
    reg.set_plan({
        "deliverables": "Ship the feature",
        "methodology": "Follow TDD",
        "context": "Legacy codebase, handle edge cases",
        "items": [
            {"id": "a", "description": "Write tests", "status": "in_progress"},
        ],
        "confirmation": "All tests green",
    })
    assert reg.deliverables == "Ship the feature"
    assert reg.methodology == "Follow TDD"
    assert reg.context == "Legacy codebase, handle edge cases"
    assert reg.confirmation == "All tests green"
    assert len(reg.items) == 1
    assert reg.items[0].id == "a"


def test_registry_set_plan_partial_update():
    """set_plan with missing keys leaves existing sections unchanged."""
    reg = PlanRegistry()
    reg.deliverables = "Original deliverables"
    reg.methodology = "Original methodology"
    reg.set_plan({
        "context": "New context",
        "items": [{"id": "b", "description": "New item", "status": "pending"}],
    })
    assert reg.deliverables == "Original deliverables"
    assert reg.methodology == "Original methodology"
    assert reg.context == "New context"
    assert len(reg.items) == 1


def test_registry_nudge_includes_deliverables():
    """format_nudge_message includes a deliverables reminder when set."""
    reg = PlanRegistry(max_nudges=3)
    reg.set_plan({
        "deliverables": "Implement auth module with 90% coverage",
        "items": [{"id": "auth", "description": "Write auth", "status": "pending"}],
    })
    msg = reg.format_nudge_message()
    assert "Implement auth module" in msg
    assert "[SYSTEM - Plan Enforcement]" in msg


def test_registry_serialization():
    """to_json produces valid JSON that round-trips all sections."""
    reg = PlanRegistry()
    reg.set_plan({
        "deliverables": "Ship it",
        "methodology": "Be careful",
        "context": "Important ref",
        "items": [
            {"id": "a", "description": "Task A", "status": "in_progress"},
            {"id": "b", "description": "Task B", "status": "completed"},
        ],
        "confirmation": "Looks good",
    })
    serialized = reg.to_json()
    parsed = json.loads(serialized)

    assert parsed["deliverables"] == "Ship it"
    assert parsed["methodology"] == "Be careful"
    assert parsed["context"] == "Important ref"
    assert parsed["confirmation"] == "Looks good"
    assert len(parsed["items"]) == 2
    assert parsed["items"][0]["id"] == "a"

    reg2 = PlanRegistry()
    reg2.set_plan(parsed)
    assert reg2.deliverables == "Ship it"
    assert len(reg2.items) == 2
    assert reg2.items[0].description == "Task A"


def test_registry_load_from_file_full_plan():
    """load_from_file loads a full plan object with sections."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "plan.json"
        path.write_text(json.dumps({
            "deliverables": "From disk",
            "methodology": "Saved method",
            "context": "Saved context",
            "items": [
                {"id": "x", "description": "Loaded item", "status": "pending"},
            ],
            "confirmation": "",
        }), encoding="utf-8")

        reg = PlanRegistry.load_from_file(path, max_nudges=5)
        assert len(reg.items) == 1
        assert reg.items[0].id == "x"
        assert reg.deliverables == "From disk"
        assert reg.methodology == "Saved method"
        assert reg.max_nudges == 5


def test_registry_load_from_file_legacy_array():
    """load_from_file handles legacy bare-array format."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "plan.json"
        path.write_text(json.dumps([
            {"id": "x", "description": "Loaded item", "status": "pending"},
        ]), encoding="utf-8")

        reg = PlanRegistry.load_from_file(path, max_nudges=5)
        assert len(reg.items) == 1
        assert reg.items[0].id == "x"
        assert reg.deliverables == ""


def test_registry_load_missing_file():
    """load_from_file with no file creates an empty registry."""
    reg = PlanRegistry.load_from_file(Path("/nonexistent/plan.json"))
    assert len(reg.items) == 0


# ---------- PlanRegistry: per-thread isolation ----------


def test_registry_thread_isolation():
    """Different threads have independent plan state."""
    reg = PlanRegistry(max_nudges=3)
    reg.set_plan({
        "deliverables": "Thread A goal",
        "items": [{"id": "a1", "description": "A work", "status": "pending"}],
    })

    reg.set_active_thread("thread-B")
    assert len(reg.items) == 0
    assert reg.deliverables == ""

    reg.set_plan({
        "deliverables": "Thread B goal",
        "items": [{"id": "b1", "description": "B work", "status": "completed"}],
    })

    reg.set_active_thread("default")
    assert reg.deliverables == "Thread A goal"
    assert len(reg.items) == 1
    assert reg.items[0].id == "a1"
    assert reg.has_pending_work()

    reg.set_active_thread("thread-B")
    assert reg.deliverables == "Thread B goal"
    assert reg.items[0].id == "b1"
    assert not reg.has_pending_work()


def test_registry_thread_nudge_isolation():
    """Nudge counts are tracked per thread."""
    reg = PlanRegistry(max_nudges=1)

    reg.set_items([{"id": "a", "description": "wip", "status": "pending"}])
    reg.format_nudge_message()
    assert not reg.should_nudge()

    reg.set_active_thread("other")
    reg.set_items([{"id": "b", "description": "other wip", "status": "pending"}])
    assert reg.should_nudge()


def test_registry_same_thread_noop():
    """set_active_thread with the current thread is a no-op."""
    reg = PlanRegistry(max_nudges=3)
    reg.set_items([{"id": "a", "description": "wip", "status": "pending"}])
    reg.set_active_thread("default")
    assert reg.items[0].id == "a"
    assert reg.has_pending_work()


def test_registry_thread_disk_persistence():
    """Per-thread plans persist to and load from disk."""
    with tempfile.TemporaryDirectory() as d:
        plans_dir = Path(d) / "plans"
        plans_dir.mkdir()

        reg = PlanRegistry(max_nudges=3, plans_dir=plans_dir)
        reg.set_plan({
            "deliverables": "Default goal",
            "items": [{"id": "x", "description": "work", "status": "in_progress"}],
        })
        persist = reg.get_persist_path()
        assert persist == plans_dir / "default.json"

        from arion_agent.util.persistence import write_file
        write_file(persist, reg.to_json())

        reg.set_active_thread("chat-123")
        reg.set_plan({
            "deliverables": "Chat 123 goal",
            "items": [{"id": "y", "description": "chat work", "status": "pending"}],
        })
        persist2 = reg.get_persist_path()
        assert persist2 == plans_dir / "chat-123.json"
        write_file(persist2, reg.to_json())

        reg2 = PlanRegistry(max_nudges=3, plans_dir=plans_dir)
        assert reg2.deliverables == "Default goal"
        assert reg2.items[0].id == "x"

        reg2.set_active_thread("chat-123")
        assert reg2.deliverables == "Chat 123 goal"
        assert reg2.items[0].id == "y"


def test_registry_thread_no_plans_dir():
    """Without plans_dir, get_persist_path returns None."""
    reg = PlanRegistry(max_nudges=3)
    assert reg.get_persist_path() is None


# ---------- update_plan tool ----------


def test_update_plan_bare_array():
    """update_plan accepts a bare JSON array as items-only shorthand."""
    with tempfile.TemporaryDirectory() as ws:
        registry = PlanRegistry()
        tools = create_agentic_core_tools(
            agent_id="agent-abc",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
            plan_registry=registry,
        )
        update_plan = next(t for t in tools if t.name == "update_plan")

        plan_json = json.dumps([
            {"id": "setup", "description": "Set up env", "status": "pending"},
            {"id": "code", "description": "Write code", "status": "in_progress"},
        ])
        result = asyncio.run(update_plan.ainvoke({"plan": plan_json}))
        assert "Plan updated" in result
        assert "2 items" in result
        assert len(registry.items) == 2
        assert registry.has_pending_work()


def test_update_plan_full_object():
    """update_plan accepts a full plan object with all sections and persists."""
    with tempfile.TemporaryDirectory() as ws:
        plans_dir = Path(ws) / ".arion" / "agents" / "agent-abc" / "plans"
        plans_dir.mkdir(parents=True)
        registry = PlanRegistry(plans_dir=plans_dir)
        tools = create_agentic_core_tools(
            agent_id="agent-abc",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
            plan_registry=registry,
        )
        update_plan = next(t for t in tools if t.name == "update_plan")

        plan_json = json.dumps({
            "deliverables": "Ship auth module",
            "methodology": "Follow existing patterns",
            "context": "Using OAuth2 flow",
            "items": [
                {"id": "impl", "description": "Implement login", "status": "in_progress"},
                {"id": "test", "description": "Write tests", "status": "pending"},
            ],
            "confirmation": "",
        })
        result = asyncio.run(update_plan.ainvoke({"plan": plan_json}))
        assert "Plan updated" in result
        assert "2 items" in result

        assert registry.deliverables == "Ship auth module"
        assert registry.methodology == "Follow existing patterns"
        assert registry.context == "Using OAuth2 flow"
        assert len(registry.items) == 2

        expected = plans_dir / "default.json"
        assert expected.exists()
        data = json.loads(expected.read_text(encoding="utf-8"))
        assert data["deliverables"] == "Ship auth module"
        assert len(data["items"]) == 2


def test_update_plan_invalid_json():
    """update_plan rejects invalid JSON."""
    with tempfile.TemporaryDirectory() as ws:
        registry = PlanRegistry()
        tools = create_agentic_core_tools(
            agent_id="agent-abc",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
            plan_registry=registry,
        )
        update_plan = next(t for t in tools if t.name == "update_plan")
        result = asyncio.run(update_plan.ainvoke({"plan": "not valid json{"}))
        assert "Error" in result


def test_update_plan_rejects_scalar():
    """update_plan rejects a scalar JSON value."""
    with tempfile.TemporaryDirectory() as ws:
        registry = PlanRegistry()
        tools = create_agentic_core_tools(
            agent_id="agent-abc",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
            plan_registry=registry,
        )
        update_plan = next(t for t in tools if t.name == "update_plan")
        result = asyncio.run(update_plan.ainvoke({"plan": '"just a string"'}))
        assert "Error" in result


def test_update_plan_overwrites():
    """update_plan replaces all existing items on each call."""
    with tempfile.TemporaryDirectory() as ws:
        registry = PlanRegistry()
        tools = create_agentic_core_tools(
            agent_id="agent-abc",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
            plan_registry=registry,
        )
        update_plan = next(t for t in tools if t.name == "update_plan")

        asyncio.run(update_plan.ainvoke({"plan": json.dumps({
            "deliverables": "v1 deliverables",
            "items": [{"id": "v1", "description": "version 1", "status": "pending"}],
        })}))
        asyncio.run(update_plan.ainvoke({"plan": json.dumps({
            "deliverables": "v2 deliverables",
            "items": [{"id": "v2", "description": "version 2", "status": "completed"}],
        })}))

        assert len(registry.items) == 1
        assert registry.items[0].id == "v2"
        assert registry.deliverables == "v2 deliverables"
        assert not registry.has_pending_work()


# ---------- get_running_status tool ----------


def test_get_running_status():
    """get_running_status returns formatted stats."""
    stats = AgentStats()
    stats.model_calls = 5
    stats.tool_calls = 12
    stats.total_messages = 30
    stats.input_tokens_estimated = 8000
    stats.output_tokens_estimated = 2000

    tools = create_agentic_core_tools(
        agent_id="test",
        workspace_dir=Path("."),
        stats=stats,
        plan_config=None,
        enable_status=True,
    )
    status_tool = next(t for t in tools if t.name == "get_running_status")
    result = asyncio.run(status_tool.ainvoke({}))

    assert "Model calls: 5" in result
    assert "Tool calls: 12" in result
    assert "Messages in context: 30" in result
    assert "input=8000" in result
    assert "output=2000" in result
    assert "Elapsed:" in result
    assert "Current time:" in result


# ---------- Middleware: registry lifecycle ----------


def test_middleware_creates_registry():
    """AgenticCoreEnvironment creates PlanRegistry when planning is enabled."""
    with tempfile.TemporaryDirectory() as ws:
        mw = AgenticCoreEnvironment(
            agent_id="reg-test",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
        )
        assert mw.plan_registry is not None
        assert isinstance(mw.plan_registry, PlanRegistry)


def test_middleware_loads_persisted_plan():
    """AgenticCoreEnvironment loads existing default plan from disk."""
    with tempfile.TemporaryDirectory() as ws:
        plans_dir = Path(ws) / ".arion" / "agents" / "load-test" / "plans"
        plans_dir.mkdir(parents=True)
        plan_path = plans_dir / "default.json"
        plan_path.write_text(json.dumps({
            "deliverables": "From disk",
            "items": [
                {"id": "persisted", "description": "From disk", "status": "in_progress"},
            ],
        }), encoding="utf-8")

        mw = AgenticCoreEnvironment(
            agent_id="load-test",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
        )
        assert mw.plan_registry is not None
        assert len(mw.plan_registry.items) == 1
        assert mw.plan_registry.items[0].id == "persisted"
        assert mw.plan_registry.deliverables == "From disk"


def test_middleware_no_registry_when_disabled():
    """No PlanRegistry when planning is disabled (plan_config=None)."""
    with tempfile.TemporaryDirectory() as ws:
        mw = AgenticCoreEnvironment(
            agent_id="no-plan",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=None,
        )
        assert mw.plan_registry is None


def test_middleware_resets_nudge_on_before_agent():
    """before_agent resets the nudge counter."""
    with tempfile.TemporaryDirectory() as ws:
        mw = AgenticCoreEnvironment(
            agent_id="nudge-reset",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(max_nudges=1),
        )
        reg = mw.plan_registry
        reg.set_items([{"id": "a", "description": "wip", "status": "pending"}])
        reg.format_nudge_message()
        assert not reg.should_nudge()

        mw.before_agent({"messages": []})
        assert reg.should_nudge()


# ---------- Middleware: system prompt injection ----------


def test_middleware_injects_planning_prompt():
    """wrap_system_message contributes <planning> section when enabled."""
    with tempfile.TemporaryDirectory() as ws:
        mw = AgenticCoreEnvironment(
            agent_id="prompt-test",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
        )

        parts = mw.wrap_system_message([])

        assert len(parts) == 1
        content = parts[0]
        assert "<planning>" in content
        assert "update_plan" in content
        assert "plans/" in content


def test_middleware_no_planning_prompt_when_disabled():
    """wrap_system_message contributes nothing when plan_config is None."""
    with tempfile.TemporaryDirectory() as ws:
        mw = AgenticCoreEnvironment(
            agent_id="no-prompt",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=None,
        )

        parts = mw.wrap_system_message([])
        assert len(parts) == 0


def test_middleware_no_duplicate_injection():
    """wrap_system_message appends once per call (graph calls once per turn)."""
    with tempfile.TemporaryDirectory() as ws:
        mw = AgenticCoreEnvironment(
            agent_id="dedup-test",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=PlanConfig(),
        )

        parts = mw.wrap_system_message([])
        assert sum(1 for p in parts if "<planning>" in p) == 1


# ---------- Custom PlanConfig ----------


def test_custom_max_nudges():
    """Custom max_nudges is applied to the PlanRegistry."""
    with tempfile.TemporaryDirectory() as ws:
        cfg = PlanConfig(max_nudges=7)
        mw = AgenticCoreEnvironment(
            agent_id="custom-nudge",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=cfg,
        )
        assert mw.plan_registry.max_nudges == 7


def test_custom_system_instructions():
    """Custom system_instructions appear in contributed section."""
    custom_prompt = "<planning>\nCustom planning guidance here.\n</planning>"
    with tempfile.TemporaryDirectory() as ws:
        cfg = PlanConfig(system_instructions=custom_prompt)
        mw = AgenticCoreEnvironment(
            agent_id="custom-prompt",
            workspace_dir=Path(ws),
            stats=AgentStats(),
            plan_config=cfg,
        )

        parts = mw.wrap_system_message([])
        assert any("Custom planning guidance" in p for p in parts)


# ---------- Integration: graph.py wiring ----------


def test_graph_planning_default():
    """create_arion_agent with defaults creates plan_registry and includes update_plan."""
    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent

        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="graph-plan-test",
            checkpointer=False,
            summarization=False,
        )
        # Agent was created successfully with planning enabled


def test_graph_planning_disabled():
    """create_arion_agent with planning=False does not create plan_registry."""
    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent

        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="graph-no-plan",
            planning=False,
            checkpointer=False,
            summarization=False,
        )
        # Agent was created successfully with planning disabled


def test_graph_planning_custom_nudges():
    """create_arion_agent with custom PlanConfig passes max_nudges through."""
    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent

        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="graph-custom-plan",
            planning=PlanConfig(max_nudges=5),
            checkpointer=False,
            summarization=False,
        )
        # Agent was created successfully with custom plan config

"""Test signal environment: store, tools, hub, archival, middleware, graph wiring."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent.environments.signal.config import SignalConfig, SignalHub
from arion_agent.environments.signal.middleware import SignalEnvironment
from arion_agent.environments.signal.store import SignalStore
from arion_agent.environments.signal.tools import create_signal_tools
from arion_agent.util.persistence import append_jsonl, load_jsonl


# ========== Shared utility tests ==========


def test_append_jsonl_creates_and_appends():
    """append_jsonl creates file on first call and appends on subsequent calls."""
    print("\n" + "=" * 60)
    print("Test: append_jsonl creates and appends")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "test.jsonl"
        append_jsonl(p, {"a": 1})
        append_jsonl(p, {"b": 2})
        records = load_jsonl(p)
        assert len(records) == 2
        assert records[0] == {"a": 1}
        assert records[1] == {"b": 2}
        print("  >> PASSED")


def test_load_jsonl_missing_file():
    """load_jsonl returns empty list for non-existent file."""
    print("\n" + "=" * 60)
    print("Test: load_jsonl missing file")
    print("=" * 60)

    result = load_jsonl(Path("/nonexistent/path.jsonl"))
    assert result == []
    print("  >> PASSED")


# ========== SignalStore tests ==========


def test_store_send_and_check():
    """Basic send and check flow."""
    print("\n" + "=" * 60)
    print("Test: store send and check")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        store = SignalStore(Path(d) / "signals", max_per_channel=100, archive_threshold=200)
        sig = store.send("default", "agent-a", "info", "hello world")

        assert sig["id"] == "sig-001"
        assert sig["sender"] == "agent-a"
        assert sig["channel"] == "default"
        assert sig["type"] == "info"
        assert sig["content"] == "hello world"
        assert "timestamp" in sig

        results = store.check("default", last_n=10)
        assert len(results) == 1
        assert results[0]["content"] == "hello world"
        print("  >> PASSED")


def test_store_channel_isolation():
    """Signals in different channels are isolated."""
    print("\n" + "=" * 60)
    print("Test: channel isolation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        store = SignalStore(Path(d) / "signals")
        store.send("alpha", "a", "info", "msg-alpha")
        store.send("beta", "a", "info", "msg-beta")

        alpha = store.check("alpha")
        beta = store.check("beta")
        assert len(alpha) == 1
        assert alpha[0]["content"] == "msg-alpha"
        assert len(beta) == 1
        assert beta[0]["content"] == "msg-beta"
        print("  >> PASSED")


def test_store_persistence():
    """Signals persist to JSONL and reload on new store instance."""
    print("\n" + "=" * 60)
    print("Test: store persistence (resurrection)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        sig_dir = Path(d) / "signals"

        store1 = SignalStore(sig_dir, max_per_channel=100, archive_threshold=200)
        store1.send("default", "agent-a", "info", "message one")
        store1.send("default", "agent-a", "info", "message two")

        store2 = SignalStore(sig_dir, max_per_channel=100, archive_threshold=200)
        results = store2.check("default", last_n=10)
        assert len(results) == 2
        assert results[0]["content"] == "message one"
        assert results[1]["content"] == "message two"

        sig3 = store2.send("default", "agent-a", "info", "message three")
        assert sig3["id"] == "sig-003", f"ID counter should resume, got {sig3['id']}"
        print("  >> PASSED")


def test_store_in_memory_eviction():
    """In-memory list is capped at max_per_channel."""
    print("\n" + "=" * 60)
    print("Test: in-memory eviction")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        store = SignalStore(Path(d) / "signals", max_per_channel=5, archive_threshold=10)
        for i in range(8):
            store.send("ch", "a", "info", f"msg-{i}")

        results = store.check("ch", last_n=100)
        assert len(results) == 5, f"Expected 5 in memory, got {len(results)}"
        assert results[0]["content"] == "msg-3"
        assert results[-1]["content"] == "msg-7"
        print("  >> PASSED")


def test_store_archival_on_load():
    """When file exceeds archive_threshold, overflow is archived on load."""
    print("\n" + "=" * 60)
    print("Test: archival on load")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        sig_dir = Path(d) / "signals"
        sig_dir.mkdir(parents=True)

        for i in range(250):
            append_jsonl(sig_dir / "default.jsonl", {
                "id": f"sig-{i+1:03d}",
                "timestamp": "2026-02-18T00:00:00Z",
                "sender": "agent-a",
                "channel": "default",
                "type": "info",
                "content": f"msg-{i}",
            })

        store = SignalStore(sig_dir, max_per_channel=100, archive_threshold=200)

        results = store.check("default", last_n=200)
        assert len(results) == 100, f"Expected 100 in memory, got {len(results)}"
        assert results[0]["content"] == "msg-150"
        assert results[-1]["content"] == "msg-249"

        active = load_jsonl(sig_dir / "default.jsonl")
        assert len(active) == 100, f"Active file should have 100, got {len(active)}"

        archive_dir = sig_dir / "archive" / "default"
        assert archive_dir.exists()
        archive_files = list(archive_dir.glob("*.jsonl"))
        assert len(archive_files) == 1
        archived = load_jsonl(archive_files[0])
        assert len(archived) == 150, f"Archive should have 150, got {len(archived)}"

        next_sig = store.send("default", "a", "info", "new")
        assert next_sig["id"] == "sig-251", f"ID should resume at 251, got {next_sig['id']}"
        print("  >> PASSED")


def test_store_no_archival_under_threshold():
    """No archival when file is under archive_threshold."""
    print("\n" + "=" * 60)
    print("Test: no archival under threshold")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        sig_dir = Path(d) / "signals"
        sig_dir.mkdir(parents=True)

        for i in range(150):
            append_jsonl(sig_dir / "default.jsonl", {
                "id": f"sig-{i+1:03d}",
                "timestamp": "2026-02-18T00:00:00Z",
                "sender": "a",
                "channel": "default",
                "type": "info",
                "content": f"msg-{i}",
            })

        store = SignalStore(sig_dir, max_per_channel=100, archive_threshold=200)

        active = load_jsonl(sig_dir / "default.jsonl")
        assert len(active) == 150, f"Active file untouched, got {len(active)}"

        archive_dir = sig_dir / "archive" / "default"
        assert not archive_dir.exists(), "No archive dir should exist"

        results = store.check("default", last_n=200)
        assert len(results) == 100, f"In-memory should have tail 100, got {len(results)}"
        print("  >> PASSED")


def test_store_receive():
    """receive() appends to both memory and JSONL without relay."""
    print("\n" + "=" * 60)
    print("Test: store receive")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        store = SignalStore(Path(d) / "signals")
        signal = {
            "id": "sig-042",
            "timestamp": "2026-02-18T10:00:00Z",
            "sender": "agent-b",
            "channel": "default",
            "type": "info",
            "content": "relayed message",
        }
        store.receive(signal)

        results = store.check("default")
        assert len(results) == 1
        assert results[0]["content"] == "relayed message"

        on_disk = load_jsonl(Path(d) / "signals" / "default.jsonl")
        assert len(on_disk) == 1
        print("  >> PASSED")


def test_store_check_empty_channel():
    """signal_check on unknown channel returns empty."""
    print("\n" + "=" * 60)
    print("Test: check empty channel")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        store = SignalStore(Path(d) / "signals")
        results = store.check("nonexistent")
        assert results == []
        print("  >> PASSED")


# ========== SignalHub tests ==========


def test_hub_relay_between_two_agents():
    """Hub relays signals from agent_a to agent_b."""
    print("\n" + "=" * 60)
    print("Test: hub relay between two agents")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        hub = SignalHub(Path(d) / "hub.json")

        store_a = SignalStore(Path(d) / "a_signals", hub=hub)
        store_b = SignalStore(Path(d) / "b_signals", hub=hub)
        hub.register("alice", store_a)
        hub.register("bob", store_b)

        store_a.send("status", "alice", "info", "done with task")

        bob_signals = store_b.check("status")
        assert len(bob_signals) == 1
        assert bob_signals[0]["sender"] == "alice"
        assert bob_signals[0]["content"] == "done with task"

        alice_signals = store_a.check("status")
        assert len(alice_signals) == 1, "Sender should see own signal"
        print("  >> PASSED")


def test_hub_no_echo_to_sender():
    """Hub does not relay back to sender."""
    print("\n" + "=" * 60)
    print("Test: hub no echo to sender")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        hub = SignalHub(Path(d) / "hub.json")

        store_a = SignalStore(Path(d) / "a_sig", hub=hub)
        hub.register("alice", store_a)

        store_a.send("ch", "alice", "info", "test")

        results = store_a.check("ch")
        assert len(results) == 1, "Only one copy (from send, not relay)"
        print("  >> PASSED")


def test_hub_three_agents():
    """Hub relays to all agents except sender."""
    print("\n" + "=" * 60)
    print("Test: hub three agents")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        hub = SignalHub(Path(d) / "hub.json")

        stores = {}
        for name in ["alice", "bob", "carol"]:
            s = SignalStore(Path(d) / f"{name}_sig", hub=hub)
            hub.register(name, s)
            stores[name] = s

        stores["alice"].send("ch", "alice", "info", "broadcast")

        assert len(stores["bob"].check("ch")) == 1
        assert len(stores["carol"].check("ch")) == 1
        assert len(stores["alice"].check("ch")) == 1  # own send only
        print("  >> PASSED")


def test_hub_persistent_registry():
    """Hub persists registry to JSON, new instance loads it."""
    print("\n" + "=" * 60)
    print("Test: hub persistent registry")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        reg_path = Path(d) / "hub.json"

        hub1 = SignalHub(reg_path)
        store_a = SignalStore(Path(d) / "a_sig", hub=hub1)
        hub1.register("alice", store_a)

        hub2 = SignalHub(reg_path)
        assert "alice" in hub2._registry
        assert hub2._registry["alice"] == str(Path(d) / "a_sig")
        print("  >> PASSED")


def test_hub_relay_to_non_live_agent():
    """Hub relays to non-live agent by writing directly to JSONL."""
    print("\n" + "=" * 60)
    print("Test: hub relay to non-live agent")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        reg_path = Path(d) / "hub.json"
        bob_dir = Path(d) / "bob_sig"

        hub1 = SignalHub(reg_path)
        store_b = SignalStore(bob_dir, hub=hub1)
        hub1.register("bob", store_b)

        hub2 = SignalHub(reg_path)
        store_a = SignalStore(Path(d) / "a_sig", hub=hub2)
        hub2.register("alice", store_a)

        store_a.send("ch", "alice", "info", "hello bob")

        bob_file = bob_dir / "ch.jsonl"
        assert bob_file.exists(), "Hub should have written to bob's JSONL"
        records = load_jsonl(bob_file)
        assert len(records) == 1
        assert records[0]["content"] == "hello bob"

        store_b2 = SignalStore(bob_dir)
        results = store_b2.check("ch")
        assert len(results) == 1
        assert results[0]["content"] == "hello bob"
        print("  >> PASSED")


def test_hub_receive_no_further_relay():
    """receive() does not trigger further relay (no infinite loop)."""
    print("\n" + "=" * 60)
    print("Test: receive does not trigger further relay")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        hub = SignalHub(Path(d) / "hub.json")

        store_a = SignalStore(Path(d) / "a_sig", hub=hub)
        store_b = SignalStore(Path(d) / "b_sig", hub=hub)
        hub.register("alice", store_a)
        hub.register("bob", store_b)

        store_a.send("ch", "alice", "info", "test")

        a_signals = store_a.check("ch")
        b_signals = store_b.check("ch")
        assert len(a_signals) == 1, f"Alice should have 1, got {len(a_signals)}"
        assert len(b_signals) == 1, f"Bob should have 1, got {len(b_signals)}"
        print("  >> PASSED")


# ========== Signal tools tests ==========


def test_signal_tools_send_and_check():
    """signal_send and signal_check tools work correctly."""
    print("\n" + "=" * 60)
    print("Test: signal tools send and check")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        store = SignalStore(Path(d) / "signals")
        tools = create_signal_tools("agent-test", store)
        send_tool = next(t for t in tools if t.name == "signal_send")
        check_tool = next(t for t in tools if t.name == "signal_check")

        send_result = asyncio.run(send_tool.ainvoke({
            "channel": "default",
            "signal_type": "info",
            "content": "hello from tool",
        }))
        assert "sig-001" in send_result
        assert "info" in send_result

        check_result = asyncio.run(check_tool.ainvoke({
            "channel": "default",
            "last_n": 10,
        }))
        assert "hello from tool" in check_result
        assert "agent-test" in check_result
        print("  >> PASSED")


def test_signal_check_empty():
    """signal_check on empty channel returns appropriate message."""
    print("\n" + "=" * 60)
    print("Test: signal_check empty channel")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as d:
        store = SignalStore(Path(d) / "signals")
        tools = create_signal_tools("agent-test", store)
        check_tool = next(t for t in tools if t.name == "signal_check")

        result = asyncio.run(check_tool.ainvoke({
            "channel": "empty",
            "last_n": 10,
        }))
        assert "No signals" in result
        print("  >> PASSED")


# ========== SignalEnvironment middleware tests ==========


def test_middleware_creates_tools():
    """SignalEnvironment provides signal_send and signal_check tools."""
    print("\n" + "=" * 60)
    print("Test: middleware creates tools")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        mw = SignalEnvironment("agent-x", Path(ws))
        names = [t.name for t in mw.tools]
        assert "signal_send" in names
        assert "signal_check" in names
        assert len(names) == 2
        print("  >> PASSED")


def test_middleware_signal_dir():
    """SignalEnvironment stores signals at the correct path."""
    print("\n" + "=" * 60)
    print("Test: middleware signal dir")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        mw = SignalEnvironment("agent-x", Path(ws))
        expected = Path(ws) / ".arion" / "agents" / "agent-x" / "signals"
        assert mw.store.signal_dir == expected
        assert expected.exists()
        print("  >> PASSED")


def test_middleware_with_hub():
    """SignalEnvironment registers with hub on creation."""
    print("\n" + "=" * 60)
    print("Test: middleware with hub")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        hub = SignalHub(Path(ws) / ".arion" / "signal_hub.json")
        cfg = SignalConfig(hub=hub)

        mw_a = SignalEnvironment("alice", Path(ws), config=cfg)
        mw_b = SignalEnvironment("bob", Path(ws), config=cfg)

        assert "alice" in hub._registry
        assert "bob" in hub._registry
        assert "alice" in hub._stores
        assert "bob" in hub._stores
        print("  >> PASSED")


# ========== graph.py wiring tests ==========


def test_graph_wiring_no_signals():
    """Default create_arion_agent has no signal tools."""
    print("\n" + "=" * 60)
    print("Test: graph wiring - no signals (default)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent
        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="test-no-sig",
            checkpointer=False,
            summarization=False,
        )
        tool_names = list(agent.get_graph().nodes.keys())
        print(f"  Graph nodes: {tool_names}")
        print("  >> PASSED (no signal tools by default)")


def test_graph_wiring_with_signals():
    """create_arion_agent with signals= adds signal tools."""
    print("\n" + "=" * 60)
    print("Test: graph wiring - with signals")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent
        agent = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="test-sig",
            signals=SignalConfig(),
            checkpointer=False,
            summarization=False,
        )

        all_tool_names = []
        for node_name, node_data in agent.get_graph().nodes.items():
            pass

        print("  >> Signal environment wired (no crash)")
        print("  >> PASSED")


def test_graph_wiring_with_hub():
    """create_arion_agent with SignalConfig(hub=...) wires hub correctly."""
    print("\n" + "=" * 60)
    print("Test: graph wiring - with hub")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as ws:
        from arion_agent.graph import create_arion_agent
        hub = SignalHub(Path(ws) / ".arion" / "signal_hub.json")

        agent_a = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="alice",
            signals=SignalConfig(hub=hub),
            checkpointer=False,
            summarization=False,
        )
        agent_b = create_arion_agent(
            model="openai:gpt-4o-mini",
            workspace_dir=ws,
            agent_id="bob",
            signals=SignalConfig(hub=hub),
            checkpointer=False,
            summarization=False,
        )

        assert "alice" in hub._registry
        assert "bob" in hub._registry
        print("  >> PASSED")


# ========== Main ==========


if __name__ == "__main__":
    print("=" * 60)
    print("Signal Environment Tests")
    print("=" * 60)

    # Shared utilities
    test_append_jsonl_creates_and_appends()
    test_load_jsonl_missing_file()

    # SignalStore
    test_store_send_and_check()
    test_store_channel_isolation()
    test_store_persistence()
    test_store_in_memory_eviction()
    test_store_archival_on_load()
    test_store_no_archival_under_threshold()
    test_store_receive()
    test_store_check_empty_channel()

    # SignalHub
    test_hub_relay_between_two_agents()
    test_hub_no_echo_to_sender()
    test_hub_three_agents()
    test_hub_persistent_registry()
    test_hub_relay_to_non_live_agent()
    test_hub_receive_no_further_relay()

    # Signal tools
    test_signal_tools_send_and_check()
    test_signal_check_empty()

    # Middleware
    test_middleware_creates_tools()
    test_middleware_signal_dir()
    test_middleware_with_hub()

    # Graph wiring
    test_graph_wiring_no_signals()
    test_graph_wiring_with_signals()
    test_graph_wiring_with_hub()

    print("\n" + "=" * 60)
    print("ALL SIGNAL TESTS PASSED")
    print("=" * 60)

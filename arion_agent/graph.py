"""ArionAgent graph builder - the ReAct loop with compression node.

Builds a LangGraph StateGraph with:
  - prefetch:  Headroom trigger; starts background summarization (non-blocking)
  - compress:  Hard trigger; awaits prefetch or runs sync, then evicts messages
  - model:     Calls the LLM with messages + bound tools
  - tools:     Executes tool calls via ToolExecutor
  - Conditional routing: route_compression (should_compress / must_compress)
  - Persistent per-agent checkpointer (checkpoints.sqlite)
  - Hot-switchable model via config["configurable"]["model"]

Graph structure:
  START -> [route_compression] -> prefetch -> model  or  compress -> model  or  model
  model -> [should_continue?] -> tools   or  END
  tools -> [route_compression] -> prefetch -> model  or  compress -> model  or  model

Middleware assembly is delegated to assembly.py. Shared construction-time
state flows through AgentContext (context.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from arion_agent.middleware.base import ArionMiddleware
from arion_agent.providers.resolver import resolve_model
from arion_agent.tool_manager.executor import ToolExecutor
from arion_agent.util.multimodal import convert_image_block as _convert_image_block
from arion_agent.util.streaming import (
    LlmStreamCallback,
    invoke_with_visual_stream,
    supports_visual_stream,
)

logger = logging.getLogger(__name__)


def _ai_message_has_tool_invocation(message: Any) -> bool:
    """Whether the assistant message requests tool execution (next edge should be tools).

    Do not use ``if msg.tool_calls:`` — an empty list is falsy. Gemini 3.x often
    returns empty text with tool calls; some responses only populate
    ``invalid_tool_calls`` (parse errors) or legacy ``additional_kwargs['function_call']``.
    """
    if getattr(message, "type", "") != "ai":
        return False
    tcs = getattr(message, "tool_calls", None)
    if isinstance(tcs, list) and len(tcs) > 0:
        return True
    inv = getattr(message, "invalid_tool_calls", None)
    if isinstance(inv, list) and len(inv) > 0:
        return True
    ak = getattr(message, "additional_kwargs", None) or {}
    fc = ak.get("function_call")
    if isinstance(fc, dict) and fc.get("name"):
        return True
    return False


def _ai_message_text_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip() if content else ""


def _ai_message_is_usable_response(message: Any) -> bool:
    """True when an assistant message has visible text or requests tools."""
    if getattr(message, "type", "") != "ai":
        return True
    return bool(_ai_message_text_content(message)) or _ai_message_has_tool_invocation(message)


# Max chars per message and total for request log snippet (~2-3 lines)
_REQUEST_LOG_SNIPPET_CHARS = 60
_REQUEST_LOG_SNIPPET_TOTAL = 200


def _message_snippet_for_log(
    messages: Sequence[AnyMessage],
    max_per_msg: int = _REQUEST_LOG_SNIPPET_CHARS,
    max_total: int = _REQUEST_LOG_SNIPPET_TOTAL,
) -> str:
    """Build a short, safe snippet of the message list for request logging."""
    parts: list[str] = []
    for m in messages:
        name = type(m).__name__
        content = getattr(m, "content", None)
        if content is None:
            part = name
        elif isinstance(content, str):
            text = content.strip()[:max_per_msg]
            if len(content.strip()) > max_per_msg:
                text += "..."
            part = f"{name}: {text}"
        elif isinstance(content, list):
            part = f"{name}: [{len(content)} parts]"
        else:
            part = f"{name}: {str(content)[:max_per_msg]}"
        parts.append(part)

    result = " | ".join(parts)
    if len(result) > max_total:
        cut = result[: max_total - 20]
        truncated = cut.rsplit(" | ", 1)[0] if " | " in cut else cut
        n_shown = truncated.count(" | ") + 1
        omitted = len(messages) - n_shown
        result = truncated + (" ... (+%d more)" % omitted if omitted > 0 else " ...")
    return result


class AgentAborted(Exception):
    """Raised by graph nodes when an external abort signal is received.

    The abort_check callback passed to create_arion_agent is polled before
    expensive operations (LLM calls, tool execution). When it returns True,
    nodes raise this exception to unwind the graph execution promptly.
    """


class ArionState(TypedDict):
    """Agent state that flows through the graph.

    messages: Conversation messages (add_messages reducer appends/removes).
    summary: Wrapped inheritable context (with preamble). Injected to model.
    summary_raw: Raw LLM summary output (no wrapper). Used as previous_summary
        in subsequent compactions to avoid wrapper contamination.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    summary_raw: str


# ---------------------------------------------------------------------------
# Self-heal: retry on provider errors or empty model responses (no content and
# no tool calls). On Gemini "Corrupted thought signature", also collapse native
# Gemini tool rounds (recover_from_thought_signature_error) before re-invoking —
# see _heal_tool_call_sequence and patch_tool_calls._collapse_tool_exchanges_*.
# ---------------------------------------------------------------------------

MAX_SELF_HEAL_RETRIES = 3


class EmptyModelResponseError(RuntimeError):
    """Model returned no text and no tool calls."""

_DANGLING_TC_RE = re.compile(
    r"tool_call_ids?\s+did\s+not\s+have\s+response\s+messages?:\s*(.+)",
    re.IGNORECASE,
)


def _parse_dangling_ids_from_error(exc: BaseException) -> list[str]:
    """Extract missing tool_call_ids from a provider 400 error, if any."""
    match = _DANGLING_TC_RE.search(str(exc))
    if not match:
        return []
    raw = match.group(1).strip().rstrip("'\"}")
    return [t.strip().strip("'\"") for t in re.split(r",\s*", raw) if t.strip()]


def _is_corrupted_thought_signature_error(exc: BaseException) -> bool:
    """Gemini 3+ may reject replayed function_call parts (checkpoint / proxy / compression)."""
    s = str(exc).lower()
    if "corrupted thought signature" in s:
        return True
    return "invalid_argument" in s and "thought" in s and "signature" in s


def _heal_tool_call_sequence(
    messages: list[Any],
    exc: BaseException,
    model: Any = None,
) -> list[Any]:
    """Repair tool_call sequence errors by re-patching and targeted injection.

    Runs the full dangling/orphaned patching pipeline a second time (catches
    issues the first pass missed due to conversion or sanitization side-effects),
    then injects synthetic ToolMessages for any specific IDs the provider
    named in the error message. Also runs provider conversion (e.g. adds
    reasoning_content for Kimi) so synthetic AIMessages are valid.
    """
    from langchain_core.messages import ToolMessage

    from arion_agent.middleware.patch_tool_calls import (
        _collapse_tool_exchanges_for_gemini_thought_signature,
        _convert_messages_for_provider,
        _patch_dangling_tool_calls,
        _patch_orphaned_tool_messages,
        _sanitize_tool_call_ids,
    )

    patched = _patch_dangling_tool_calls(messages)
    patched = _patch_orphaned_tool_messages(patched)

    missing_ids = _parse_dangling_ids_from_error(exc)
    if missing_ids:
        needed = set(missing_ids)
        result: list[Any] = []
        for msg in patched:
            result.append(msg)
            if getattr(msg, "type", "") == "ai" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id in needed:
                        result.append(ToolMessage(
                            content="[interrupted] Tool call was not executed.",
                            name=tc.get("name", "unknown"),
                            tool_call_id=tc_id,
                        ))
                        needed.discard(tc_id)
        for orphan_id in needed:
            result.append(ToolMessage(
                content="[interrupted] Tool call was not executed.",
                name="unknown",
                tool_call_id=orphan_id,
            ))
        patched = result

    patched = _sanitize_tool_call_ids(patched)
    if (
        model is not None
        and type(model).__name__ == "ChatGoogleGenerativeAI"
        and _is_corrupted_thought_signature_error(exc)
    ):
        patched = _collapse_tool_exchanges_for_gemini_thought_signature(
            patched,
            model,
            recover_from_thought_signature_error=True,
        )
    patched = _convert_messages_for_provider(patched, model)
    return patched


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_react_graph(
    mw_stack: list[ArionMiddleware],
    all_tools: list[BaseTool],
    tool_map: dict[str, BaseTool],
    executor: ToolExecutor,
    default_model_spec: Any,
    extra_model_kwargs: dict[str, Any],
    compression_config: dict[str, Any] | None,
    abort_check: Any | None = None,
    on_llm_stream: LlmStreamCallback | None = None,
    plan_registry: Any | None = None,
    routing_mw: Any | None = None,
) -> StateGraph:
    """Build the ReAct graph with optional compression node."""

    def _get_model(
        config: RunnableConfig | dict[str, Any] | None = None,
        *,
        _routing_mw: Any | None = routing_mw,
    ) -> BaseChatModel:
        cfg = config if isinstance(config, dict) else {}
        configurable = cfg.get("configurable", {})
        explicit_override = configurable.get("model")
        if explicit_override:
            return resolve_model(explicit_override, **extra_model_kwargs)
        if _routing_mw is not None:
            routed_spec = _routing_mw.get_model_spec(default_model_spec)
            return resolve_model(routed_spec, **extra_model_kwargs)
        return resolve_model(default_model_spec, **extra_model_kwargs)

    async def model_node(state: ArionState, config: RunnableConfig) -> dict[str, Any]:
        if abort_check is not None and abort_check():
            raise AgentAborted("Aborted before model call")

        current_model = _get_model(config)

        messages = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        active_tools = list(all_tools)

        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = configurable.get("thread_id", "default")
        mw_kwargs: dict[str, Any] = {"thread_id": thread_id, "model": current_model}

        sys_parts: list[str] = []
        for mw in mw_stack:
            sys_parts = mw.wrap_system_message(sys_parts, **mw_kwargs)

        system_msg: SystemMessage | None = None
        if sys_parts:
            system_msg = SystemMessage(content="\n\n".join(sys_parts))
            logger.debug(
                "System message assembled (%d sections, %d chars) for thread '%s'",
                len(sys_parts), len(system_msg.content), thread_id,
            )

        for mw in mw_stack:
            messages, active_tools, mw_kwargs = mw.wrap_model_call(
                messages, active_tools, **mw_kwargs
            )

        summary = state.get("summary", "")
        if summary:
            messages = [HumanMessage(content=summary)] + messages

        if system_msg is not None:
            messages = [system_msg] + messages

        bound = current_model.bind_tools(active_tools) if active_tools else current_model
        snippet = _message_snippet_for_log(messages)
        logger.info(
            "LLM request: %d messages, tools=%s | snippet: %s",
            len(messages), bool(active_tools), snippet,
        )
        async def _invoke_model(msgs: list[Any]) -> Any:
            if on_llm_stream is not None and supports_visual_stream(current_model):
                return await invoke_with_visual_stream(
                    bound,
                    msgs,
                    thread_id=thread_id,
                    callback=on_llm_stream,
                )
            return await bound.ainvoke(msgs)

        async def _invoke_with_abort(msgs: list[Any]) -> Any:
            """Race the LLM call against abort_check, polling every 150ms.

            When the user hits stop, cancel_event fires — the next poll
            catches it and cancels the in-flight HTTP request mid-stream,
            rather than waiting for the full response.
            """
            if abort_check is None:
                return await _invoke_model(msgs)
            task = asyncio.create_task(_invoke_model(msgs))
            try:
                while not task.done():
                    if abort_check():
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                        raise AgentAborted("Aborted during model call")
                    await asyncio.sleep(0.15)
                return await task
            except asyncio.CancelledError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise

        last_exc: BaseException | None = None
        response = None
        for attempt in range(MAX_SELF_HEAL_RETRIES + 1):
            try:
                response = await _invoke_with_abort(messages)
                for mw in mw_stack:
                    response = mw.wrap_model_response(response, **mw_kwargs)
                if not _ai_message_is_usable_response(response):
                    raise EmptyModelResponseError(
                        "Model returned empty response (no content, no tool calls)"
                    )
                break
            except EmptyModelResponseError as exc:
                last_exc = exc
                if attempt >= MAX_SELF_HEAL_RETRIES:
                    raise
                logger.warning(
                    "Self-heal: empty model response (attempt %d/%d); retrying",
                    attempt + 1, MAX_SELF_HEAL_RETRIES + 1,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= MAX_SELF_HEAL_RETRIES:
                    raise
                logger.warning(
                    "Self-heal: LLM request failed (attempt %d/%d); "
                    "re-patching and retrying: %s",
                    attempt + 1, MAX_SELF_HEAL_RETRIES + 1,
                    str(exc)[:200],
                )
                messages = _heal_tool_call_sequence(messages, exc, current_model)
        else:
            raise last_exc  # type: ignore[misc]

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if tc.get("id") is None:
                    tc["id"] = f"call_{tc.get('name', 'unknown')}_{uuid.uuid4().hex[:8]}"
        elif hasattr(response, "invalid_tool_calls") and response.invalid_tool_calls:
            for itc in response.invalid_tool_calls:
                if isinstance(itc, dict) and itc.get("id") is None:
                    itc["id"] = f"call_invalid_{uuid.uuid4().hex[:8]}"

        state_messages: list[Any] = []
        for mw in mw_stack:
            state_messages.extend(mw.drain_state_updates())
        state_messages.append(response)

        return {"messages": state_messages}

    async def tool_node(state: ArionState) -> dict[str, Any]:
        if abort_check is not None and abort_check():
            raise AgentAborted("Aborted before tool execution")

        last_message = state["messages"][-1]
        tool_calls: list[dict[str, Any]] = list(
            getattr(last_message, "tool_calls", None) or []
        )

        if not tool_calls:
            ak = getattr(last_message, "additional_kwargs", None) or {}
            fc = ak.get("function_call")
            if isinstance(fc, dict) and fc.get("name"):
                raw_args = fc.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                except (json.JSONDecodeError, TypeError, ValueError):
                    args = {}
                tc_id = fc.get("id") or f"call_{fc['name']}_{uuid.uuid4().hex[:8]}"
                tool_calls = [{"name": fc["name"], "args": args, "id": tc_id}]

        if not tool_calls:
            inv = getattr(last_message, "invalid_tool_calls", None) or []
            if inv:
                from langchain_core.messages import ToolMessage

                results: list[Any] = []
                for itc in inv:
                    if not isinstance(itc, dict):
                        continue
                    tid = itc.get("id") or f"invalid_{uuid.uuid4().hex[:8]}"
                    name = itc.get("name") or "unknown"
                    err = itc.get("error") or "invalid tool call"
                    results.append(
                        ToolMessage(
                            content=f"TOOL PARSE ERROR ({name}): {err}",
                            name=name,
                            tool_call_id=tid,
                        )
                    )
                return {"messages": results} if results else {"messages": []}
            return {"messages": []}

        results = []
        for tc in tool_calls:
            tool = tool_map.get(tc["name"])
            if tool is None:
                from langchain_core.messages import ToolMessage
                results.append(ToolMessage(
                    content=f"TOOL ERROR ({tc['name']})\nType: ToolNotFound\n"
                            f"Message: No tool named '{tc['name']}' is available.",
                    name=tc["name"],
                    tool_call_id=tc["id"],
                ))
                continue

            result_msg = await executor.execute(tool, tc)

            for mw in mw_stack:
                result_msg = mw.wrap_tool_call(tc["name"], tc["args"], result_msg)

            result_msg = _convert_image_block(result_msg)
            results.append(result_msg)

        return {"messages": results}

    def should_continue(state: ArionState) -> str:
        last = state["messages"][-1]
        if _ai_message_has_tool_invocation(last):
            return "tools"
        if plan_registry is not None and plan_registry.should_nudge():
            return "plan_guard"
        return END

    async def plan_guard_node(state: ArionState) -> dict[str, Any]:
        """Inject a synthetic nudge when the agent stops with incomplete plan items."""
        nudge_text = plan_registry.format_nudge_message()
        logger.info(
            "Plan guard: nudging agent (nudge %d/%d, %s)",
            plan_registry.nudge_count,
            plan_registry.max_nudges,
            plan_registry.pending_summary(),
        )
        return {"messages": [HumanMessage(content=nudge_text)]}

    enable_plan_guard = plan_registry is not None and plan_registry.max_nudges > 0

    graph = StateGraph(ArionState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)

    if enable_plan_guard:
        graph.add_node("plan_guard", plan_guard_node)

    model_edges: dict[str, str] = {"tools": "tools", END: END}
    if enable_plan_guard:
        model_edges["plan_guard"] = "plan_guard"

    if compression_config is not None:
        route_compression = compression_config["route_compression"]
        graph.add_node("prefetch", compression_config["prefetch_node"])
        graph.add_node("compress", compression_config["compress_node"])

        compression_edges = {
            "compress": "compress",
            "prefetch": "prefetch",
            "model": "model",
        }

        graph.add_conditional_edges(
            START, route_compression,
            compression_edges,
        )
        graph.add_edge("prefetch", "model")
        graph.add_edge("compress", "model")
        graph.add_conditional_edges(
            "model", should_continue,
            model_edges,
        )
        graph.add_conditional_edges(
            "tools", route_compression,
            compression_edges,
        )
    else:
        graph.set_entry_point("model")
        graph.add_conditional_edges(
            "model", should_continue,
            model_edges,
        )
        graph.add_edge("tools", "model")

    if enable_plan_guard:
        graph.add_edge("plan_guard", "model")

    return graph


def _setup_checkpointer(
    identity_dir: Path,
    checkpointer: Checkpointer | None | bool,
) -> Checkpointer | None:
    """Resolve checkpointer: True=per-agent SQLite, False/None=stateless, instance=use as-is."""
    if checkpointer is True:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import aiosqlite
        import asyncio
        from arion_agent.util.runtime import is_container

        identity_dir.mkdir(parents=True, exist_ok=True)
        db_path = identity_dir / "checkpoints.sqlite"
        docker_safe = is_container()

        async def _create_saver() -> AsyncSqliteSaver:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

            if docker_safe:
                uri = f"file:{db_path}?vfs=unix-none"
                conn = await aiosqlite.connect(uri, uri=True)
                await conn.execute("PRAGMA journal_mode=DELETE")
                await conn.execute("PRAGMA mmap_size=0")
                logger.info("Checkpointer using docker-safe VFS (unix-none): %s", db_path)
            else:
                conn = await aiosqlite.connect(str(db_path))
            saver = AsyncSqliteSaver(conn, serde=JsonPlusSerializer())
            await saver.setup()
            return saver

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(_create_saver())).result()
        else:
            return asyncio.run(_create_saver())

    if checkpointer is not None and checkpointer is not False:
        return checkpointer

    return None


def _finalize_agent(
    compiled: CompiledStateGraph,
    mw_stack: list[ArionMiddleware],
    effective_checkpointer: Checkpointer | None,
    agent_stats: Any,
    agent_id: str,
    recursion_limit: int,
    plan_registry: Any | None = None,
) -> CompiledStateGraph:
    """Patch ainvoke with before/after hooks, attach cleanup and metadata."""
    original_ainvoke = compiled.ainvoke

    async def patched_ainvoke(
        input: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if plan_registry is not None:
            configurable = (config or {}).get("configurable", {})
            thread_id = configurable.get("thread_id", "default")
            plan_registry.set_active_thread(thread_id)

        if input is None:
            # Resume from checkpoint — still need hooks but no state to patch.
            dummy_state: dict[str, Any] = {"messages": []}
            for mw in mw_stack:
                mw.before_agent(dummy_state)
            result = await original_ainvoke(input, config=config, **kwargs)
            for mw in mw_stack:
                mw.after_agent(result)
            return result

        state = dict(input)
        for mw in mw_stack:
            patch = mw.before_agent(state)
            if patch:
                state.update(patch)
        result = await original_ainvoke(state, config=config, **kwargs)
        for mw in mw_stack:
            mw.after_agent(result)
        return result

    compiled.ainvoke = patched_ainvoke  # type: ignore[assignment]

    async def _cleanup() -> None:
        await asyncio.sleep(0.5)
        if effective_checkpointer and hasattr(effective_checkpointer, "conn"):
            try:
                await effective_checkpointer.conn.close()
            except Exception:
                pass
        if hasattr(compiled, "_io_backend"):
            try:
                compiled._io_backend.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    compiled.cleanup = _cleanup  # type: ignore[attr-defined]

    result = compiled.with_config({"recursion_limit": recursion_limit})
    result.stats = agent_stats  # type: ignore[attr-defined]
    result.agent_id = agent_id  # type: ignore[attr-defined]
    result.cleanup = _cleanup  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# Public API: create_arion_agent
# ---------------------------------------------------------------------------


def create_arion_agent(
    model: str | BaseChatModel,
    workspace_dir: str,
    *,
    remote_url: str | None = None,
    agent_id: str | None = None,
    soul: Any = None,
    deep_memory: Any = None,
    shallow_memory: Any = None,
    pinned_instructions: str | None = None,
    tools: Sequence[BaseTool] | None = None,
    middleware: Sequence[ArionMiddleware] | None = None,
    subagents: Any | None = None,
    skills: Any | None = None,
    summarization: Any | None = None,
    planning: Any | None = None,
    enable_status: bool = False,
    signals: Any | None = None,
    heartbeat: Any | None = None,
    timezone: str = "UTC",
    mounts: Sequence | None = None,
    confinement: str = "auto",
    network_allowed: bool = False,
    shell_backend: Any = None,
    session_log: bool = False,
    checkpointer: Checkpointer | None | bool = True,
    tool_executor: ToolExecutor | None = None,
    recursion_limit: int | None = None,
    max_recursion_depth: int | None = None,
    on_compress: Any | None = None,
    abort_check: Any | None = None,
    on_llm_stream: LlmStreamCallback | None = None,
    cutoff_fn: Any | None = None,
    routing: Any | None = None,
    _parent_agent_id: str | None = None,
    _parent_thread_id: str | None = None,
    **model_kwargs: Any,
) -> CompiledStateGraph:
    """Create an ArionAgent with a ReAct loop and optional compression.

    Args:
        model: Required. Model spec (BaseChatModel, "provider:model", or plain name).
        workspace_dir: Required. Workspace directory for files, shell, and identity.
        remote_url: URL of a host-side I/O service. When set, all file operations
            route through HTTP instead of direct filesystem access. Solves grpcfuse
            file descriptor degradation on Docker bind mounts (Phase 17+).
        agent_id: Optional. Stable identity for resumable agents. None = auto-generate.
        soul: Agent identity. SoulConfig for structured template + instructions,
            plain string for simple identity, or None for minimal default.
        deep_memory: Curated long-term memory. MemoryConfig for template + instructions,
            plain string for simple seed, or None for minimal default.
        shallow_memory: ShallowMemoryConfig for memories/ folder structure guidance.
        pinned_instructions: Non-editable guardrails injected before SOUL in context.
        tools: Additional user-provided tools.
        middleware: Additional ArionMiddleware instances.
        subagents: List of SubAgentSpec for spawnable subagent classes.
        skills: SkillMiddleware instance for agent skills.
        summarization: Compression configuration. None = auto with STANDARD_POLICY.
            False = disable compression entirely.
        planning: PlanConfig instance for work planning, or False to disable.
        enable_status: If True, add get_running_status tool.
        signals: SignalConfig instance to enable signal environment.
        heartbeat: HeartbeatConfig instance to enable heartbeat environment.
        timezone: IANA timezone for the agent.
        mounts: List of MountSpec for local directory mounts.
        confinement: Shell sandboxing mode ("auto", "bwrap", "none").
        network_allowed: If False and confinement active, block network.
        shell_backend: ShellBackend instance for inline execution routing.
            None = local (default). Use RemoteShellBackend(url) to proxy
            commands to a host-side service.
        session_log: Enable JSONL session logging.
        checkpointer: True = per-agent SQLite. False/None = no persistence.
        tool_executor: Custom ToolExecutor with timeout/error handling.
        recursion_limit: Max graph steps before forced stop. None = auto-derive
            from summarization policy: 3 * 1.2 * trigger_messages (min 200).
        max_recursion_depth: Max subagent nesting depth.
        on_compress: SummarizationCallback fired before/after the summary LLM call.
        abort_check: Callable returning True when the agent should stop. Checked
            before each LLM call (model, compression) and tool execution.
        on_llm_stream: Optional callback for visual-only LLM streaming (UI progress).
            The graph still commits only the fully aggregated AIMessage. Supported
            for ChatDeepSeek and ChatMoonshot; other models use ainvoke.
        routing: RoutingConfig for intra-loop model switching. None = disabled.
            When enabled, the model self-tags each response to route the next
            step to either the strong (default) or weak (fast) model.
        **model_kwargs: Forwarded to init_chat_model.

    Returns:
        A compiled LangGraph StateGraph.
    """
    from arion_agent.assembly import (
        add_agentic_core,
        add_file_and_shell,
        add_heartbeat,
        add_identity,
        add_routing,
        add_signals,
        add_skills,
        add_stats,
        add_subagents,
        collect_tools,
        configure_compression,
        ensure_patch_tool_calls,
    )
    from arion_agent.context import AgentContext
    from arion_agent.util.stats import AgentStats
    from arion_agent.util.timezone import AgentClock

    if agent_id is None:
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"

    ws = Path(workspace_dir)
    identity_dir = ws / ".arion" / "agents" / agent_id

    # ---- Set up I/O backend (Phase 17+) ----
    from arion_agent.util.io_backend import LocalIOBackend
    from arion_agent.util.persistence import set_default_backend
    if remote_url:
        from arion_agent.util.remote_io import RemoteIOBackend
        _io_backend = RemoteIOBackend(remote_url)
    else:
        _io_backend = LocalIOBackend(ws)
    set_default_backend(_io_backend, ws)

    ctx = AgentContext(
        agent_id=agent_id,
        identity_dir=identity_dir,
        workspace_dir=ws,
        clock=AgentClock(timezone),
        stats=AgentStats(),
        default_model_spec=model,
        extra_model_kwargs=dict(model_kwargs),
    )

    # ---- Assemble middleware stack ----

    mw_stack = list(middleware or [])
    add_identity(ctx, mw_stack, soul=soul, deep_memory=deep_memory,
                 shallow_memory=shallow_memory, pinned_instructions=pinned_instructions)
    plan_registry = add_agentic_core(ctx, mw_stack, planning=planning, enable_status=enable_status)
    add_file_and_shell(ctx, mw_stack, mounts=mounts, confinement=confinement,
                       network_allowed=network_allowed, shell_backend=shell_backend)
    resolved_heartbeat = add_heartbeat(ctx, mw_stack, heartbeat=heartbeat)
    add_signals(ctx, mw_stack, signals=signals)
    add_stats(ctx, mw_stack, session_log=session_log)

    compression_config = configure_compression(ctx, summarization=summarization,
                                                on_compress=on_compress,
                                                abort_check=abort_check,
                                                cutoff_fn=cutoff_fn)

    if recursion_limit is None:
        if compression_config is not None:
            trigger = compression_config.get("trigger_messages", 80)
            # 3 graph steps per tool-call cycle (model, tools, compress)
            # 1.2x safety margin over the summarization trigger count
            recursion_limit = max(5000, int(3 * 1.2 * trigger))
        else:
            recursion_limit = 5000

    add_skills(ctx, mw_stack, skills=skills)
    add_subagents(ctx, mw_stack, subagents=subagents, user_tools=tools,
                  soul=soul, deep_memory=deep_memory,
                  shallow_memory=shallow_memory, skills=skills,
                  mounts=mounts,
                  max_recursion_depth=max_recursion_depth,
                  checkpointer=checkpointer)
    routing_mw = add_routing(ctx, mw_stack, routing=routing)
    ensure_patch_tool_calls(mw_stack)

    # ---- Build graph ----

    all_tools, tool_map = collect_tools(mw_stack, user_tools=tools)

    timeout_overrides: dict[str, int | None] = {}
    for mw in mw_stack:
        if hasattr(mw, "tool_timeout_overrides"):
            timeout_overrides.update(mw.tool_timeout_overrides)
    executor = tool_executor or ToolExecutor(timeout_overrides=timeout_overrides)

    graph = _build_react_graph(
        mw_stack, all_tools, tool_map, executor,
        ctx.default_model_spec, ctx.extra_model_kwargs,
        compression_config,
        abort_check=abort_check,
        on_llm_stream=on_llm_stream,
        plan_registry=plan_registry,
        routing_mw=routing_mw,
    )

    effective_checkpointer = _setup_checkpointer(identity_dir, checkpointer)
    compiled = graph.compile(checkpointer=effective_checkpointer)
    compiled._io_backend = _io_backend  # type: ignore[attr-defined]

    result = _finalize_agent(
        compiled, mw_stack, effective_checkpointer,
        ctx.stats, agent_id, recursion_limit,
        plan_registry=plan_registry,
    )

    # ---- Post-compilation attachments ----

    if resolved_heartbeat is not None:
        from arion_agent.environments.heartbeat.scheduler import HeartbeatScheduler
        result.heartbeat_scheduler = HeartbeatScheduler(  # type: ignore[attr-defined]
            agent=result,
            config=resolved_heartbeat,
            identity_dir=identity_dir,
            workspace_dir=ws,
            clock=ctx.clock,
        )
    else:
        result.heartbeat_scheduler = None  # type: ignore[attr-defined]

    return result

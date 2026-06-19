"""Modular middleware assemblers for create_arion_agent.

Each function adds one middleware layer to the stack. Adding a new
environment means writing one function here and one call in the
orchestrator (graph.py). No editing the middle of a long function.

Compression (formerly summarization) is configured here but wired
as a graph node, not a middleware layer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.context import AgentContext
from arion_agent.middleware.base import ArionMiddleware
from arion_agent.middleware.patch_tool_calls import PatchToolCallsMiddleware

logger = logging.getLogger(__name__)


def add_identity(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    soul: Any,
    deep_memory: Any,
    shallow_memory: Any,
    pinned_instructions: str | None,
) -> None:
    """Insert IdentityMiddleware at position 0 (before user middleware)."""
    from arion_agent.identity.middleware import IdentityMiddleware

    mw_stack.insert(0, IdentityMiddleware(
        ctx.identity_dir,
        agent_id=ctx.agent_id,
        workspace_dir=ctx.workspace_dir,
        soul=soul,
        deep_memory=deep_memory,
        shallow_memory=shallow_memory,
        pinned_instructions=pinned_instructions,
    ))


def add_agentic_core(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    planning: Any,
    enable_status: bool,
) -> Any | None:
    """Insert AgenticCoreEnvironment at position 1.

    Returns the PlanRegistry if planning is enabled, None otherwise.
    The caller passes this to the graph builder for plan_guard wiring.
    """
    from arion_agent.environments.agentic_core.config import PlanConfig
    from arion_agent.environments.agentic_core.middleware import AgenticCoreEnvironment

    if planning is False:
        effective_plan_config = None
    elif isinstance(planning, PlanConfig):
        effective_plan_config = planning
    else:
        effective_plan_config = PlanConfig()

    mw = AgenticCoreEnvironment(
        agent_id=ctx.agent_id,
        workspace_dir=ctx.workspace_dir,
        stats=ctx.stats,
        plan_config=effective_plan_config,
        enable_status=enable_status,
        clock=ctx.clock,
    )
    mw_stack.insert(1, mw)
    return mw.plan_registry


def add_file_and_shell(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    mounts: Sequence | None = None,
    confinement: str = "auto",
    network_allowed: bool = False,
    shell_backend: object | None = None,
    abort_check: Any | None = None,
    jobs_only: bool = False,
) -> None:
    """Insert FileEnvironment at position 2 and ShellEnvironment at position 3."""
    from arion_agent.environments._sandbox.config import SandboxConfig
    from arion_agent.environments.file.middleware import FileEnvironment
    from arion_agent.environments.shell.middleware import ShellEnvironment

    sandbox_cfg = SandboxConfig(
        workspace_dir=ctx.workspace_dir,
        mounts=list(mounts or []),
        confinement=confinement,
        network_allowed=network_allowed,
    )
    mw_stack.insert(2, FileEnvironment(sandbox_cfg))
    mw_stack.insert(3, ShellEnvironment(
        sandbox_cfg,
        shell_backend=shell_backend,
        abort_check=abort_check,
        jobs_only=jobs_only,
    ))


def add_heartbeat(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    heartbeat: Any,
) -> Any | None:
    """Append HeartbeatEnvironment if heartbeat config is provided.

    May override ctx.clock if heartbeat specifies a non-UTC timezone.
    Returns the resolved HeartbeatConfig (needed later for scheduler), or None.
    """
    if heartbeat is None:
        return None

    from arion_agent.environments.heartbeat.config import HeartbeatConfig as HbCfg
    from arion_agent.environments.heartbeat.middleware import HeartbeatEnvironment
    from arion_agent.util.timezone import AgentClock

    if not isinstance(heartbeat, HbCfg):
        heartbeat = HbCfg()
    if heartbeat.timezone != "UTC":
        ctx.clock = AgentClock(heartbeat.timezone)

    mw_stack.append(HeartbeatEnvironment(
        agent_id=ctx.agent_id,
        identity_dir=ctx.identity_dir,
        workspace_dir=ctx.workspace_dir,
        clock=ctx.clock,
        config=heartbeat,
    ))
    return heartbeat


def add_signals(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    signals: Any,
) -> None:
    """Append SignalEnvironment if signal config is provided."""
    if signals is None:
        return

    from arion_agent.environments.signal.config import SignalConfig as SigCfg
    from arion_agent.environments.signal.middleware import SignalEnvironment

    if not isinstance(signals, SigCfg):
        signals = SigCfg()

    mw_stack.append(SignalEnvironment(
        agent_id=ctx.agent_id,
        workspace_dir=ctx.workspace_dir,
        config=signals,
        clock=ctx.clock,
    ))


def add_stats(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    session_log: bool,
) -> None:
    """Append StatsMiddleware (always on)."""
    from arion_agent.util.stats import SessionLogger
    from arion_agent.middleware.stats import StatsMiddleware

    sess_logger = None
    if session_log:
        log_path = ctx.identity_dir / "session_logs" / "session.jsonl"
        sess_logger = SessionLogger(log_path)

    mw_stack.append(StatsMiddleware(ctx.stats, session_logger=sess_logger))


def configure_compression(
    ctx: AgentContext,
    *,
    summarization: Any,
    on_compress: Any | None = None,
    abort_check: Any | None = None,
    cutoff_fn: Any | None = None,
) -> dict[str, Any] | None:
    """Configure compression node parameters. Returns config dict or None if disabled.

    This replaces the old add_summarization that appended SummarizationMiddleware.
    Compression is a graph node, not a middleware layer.

    on_compress: Optional SummarizationCallback fired before/after the summary
        LLM call. Useful for external status updates (e.g. GUI state file).
    """
    if summarization is False:
        return None

    from arion_agent.providers.resolver import resolve_model
    from arion_agent.summarization.compress import (
        PrefetchRegistry,
        make_compress_node,
        make_prefetch_node,
        make_route_compression,
    )
    from arion_agent.summarization.policies import STANDARD_POLICY, STANDARD_PREFETCH_POLICY

    if summarization is None:
        policy = STANDARD_POLICY
        summary_model_spec = None
        is_perpetual = False
        summary_prompt = None
        summary_budget = None
        arg_truncation_trigger = None
        arg_truncation_keep = None
        arg_max_length = None
        max_tokens = None
    elif hasattr(summarization, "policy"):
        policy = getattr(summarization, "policy", None) or STANDARD_POLICY
        summary_model_spec = None
        is_perpetual = getattr(summarization, "is_perpetual", False)
        summary_prompt = getattr(summarization, "summary_prompt", None)
        summary_budget = getattr(summarization, "summary_budget", None)
        arg_truncation_trigger = getattr(summarization, "arg_truncation_trigger", None)
        arg_truncation_keep = getattr(summarization, "arg_truncation_keep", None)
        arg_max_length = getattr(summarization, "arg_max_length", None)
        max_tokens = getattr(summarization, "max_tokens", None)
    else:
        policy = STANDARD_POLICY
        summary_model_spec = None
        is_perpetual = False
        summary_prompt = None
        summary_budget = None
        arg_truncation_trigger = None
        arg_truncation_keep = None
        arg_max_length = None
        max_tokens = None

    summary_model = summary_model_spec or resolve_model(
        ctx.default_model_spec, **ctx.extra_model_kwargs
    )

    from arion_agent.summarization.config import PolicyDecision, SummarizationPolicy
    from arion_agent.summarization.policies import DEFAULT_TRIGGER_MESSAGES

    if isinstance(policy, SummarizationPolicy) and policy.trigger_messages is not None:
        trigger_messages = policy.trigger_messages
    else:
        trigger_messages = DEFAULT_TRIGGER_MESSAGES

    prefetch_policy: Callable[..., PolicyDecision | None] | None = None
    if hasattr(summarization, "prefetch_policy"):
        prefetch_policy = getattr(summarization, "prefetch_policy", None)
    elif policy is STANDARD_POLICY or summarization is None:
        prefetch_policy = STANDARD_PREFETCH_POLICY

    registry = PrefetchRegistry()
    compress_kwargs_shared: dict[str, Any] = {
        "history_dir": ctx.identity_dir,
        "workspace_dir": ctx.workspace_dir,
        "max_tokens": max_tokens,
        "is_perpetual": is_perpetual,
        "abort_check": abort_check,
        "cutoff_fn": cutoff_fn,
    }
    if summary_prompt is not None:
        compress_kwargs_shared["summary_prompt"] = summary_prompt
    if summary_budget is not None:
        compress_kwargs_shared["summary_budget"] = summary_budget
    if arg_max_length is not None:
        compress_kwargs_shared["arg_max_length"] = arg_max_length

    route_compression = make_route_compression(
        policy, max_tokens=max_tokens, prefetch_policy=prefetch_policy,
    )
    prefetch_node = make_prefetch_node(
        policy,
        summary_model,
        registry,
        prefetch_policy=prefetch_policy,
        on_event=on_compress,
        **compress_kwargs_shared,
    )
    compress_node = make_compress_node(
        policy,
        summary_model,
        on_event=on_compress,
        registry=registry,
        arg_truncation_trigger=arg_truncation_trigger,
        arg_truncation_keep=arg_truncation_keep,
        **compress_kwargs_shared,
    )

    return {
        "route_compression": route_compression,
        "prefetch_node": prefetch_node,
        "compress_node": compress_node,
        "trigger_messages": trigger_messages,
        "arg_truncation_trigger": arg_truncation_trigger,
        "arg_truncation_keep": arg_truncation_keep,
        "arg_max_length": arg_max_length,
    }


def add_skills(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    skills: Any,
) -> None:
    """Append SkillMiddleware if skill config is provided."""
    if skills is None:
        return

    from arion_agent.skills.middleware import SkillMiddleware as SkillMW

    if isinstance(skills, SkillMW):
        skills.set_identity_dir(ctx.identity_dir)
        skills.set_workspace_dir(ctx.workspace_dir)
        mw_stack.append(skills)


def add_subagents(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    subagents: Any,
    user_tools: Sequence[BaseTool] | None,
    soul: Any,
    deep_memory: Any = None,
    shallow_memory: Any = None,
    skills: Any = None,
    mounts: Sequence | None = None,
    max_recursion_depth: int | None,
    checkpointer: Any,
) -> None:
    """Append SubagentMiddleware if subagent specs are provided."""
    if subagents is None:
        return

    effective_depth = max_recursion_depth
    if effective_depth is not None and effective_depth <= 0:
        return

    from arion_agent.subagenting.middleware import SubagentMiddleware as SubMW

    mw_stack.append(SubMW(
        specs=subagents,
        parent_agent_id=ctx.agent_id,
        parent_model=ctx.default_model_spec,
        parent_workspace=str(ctx.workspace_dir),
        parent_tools=list(user_tools or []),
        parent_soul=soul,
        parent_deep_memory=deep_memory,
        parent_shallow_memory=shallow_memory,
        parent_skills=skills,
        parent_mounts=mounts,
        max_recursion_depth=effective_depth,
        checkpointer=checkpointer,
    ))


def add_routing(
    ctx: AgentContext,
    mw_stack: list[ArionMiddleware],
    *,
    routing: Any,
) -> Any | None:
    """Insert RoutingMiddleware if routing config is provided.

    Returns the RoutingMiddleware instance (needed by the graph builder
    for model selection), or None if routing is disabled.
    """
    if routing is None or routing is False:
        return None

    from arion_agent.routing.config import RoutingConfig
    from arion_agent.routing.middleware import RoutingMiddleware

    if not isinstance(routing, RoutingConfig):
        routing = RoutingConfig()

    if not routing.enabled or not routing.weak_model:
        return None

    mw = RoutingMiddleware(routing)
    mw_stack.append(mw)
    return mw


def ensure_patch_tool_calls(mw_stack: list[ArionMiddleware]) -> None:
    """Append PatchToolCallsMiddleware if not already present."""
    if not any(isinstance(m, PatchToolCallsMiddleware) for m in mw_stack):
        mw_stack.append(PatchToolCallsMiddleware())


def collect_tools(
    mw_stack: list[ArionMiddleware],
    user_tools: Sequence[BaseTool] | None,
) -> tuple[list[BaseTool], dict[str, BaseTool]]:
    """Gather all tools from middleware + user tools. Returns (all_tools, tool_map)."""
    mw_tools: list[BaseTool] = []
    for mw in mw_stack:
        mw_tools.extend(mw.tools)

    tool_map: dict[str, BaseTool] = {t.name: t for t in mw_tools + list(user_tools or [])}
    return list(tool_map.values()), tool_map

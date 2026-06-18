# ArionAgent

A modular Python agent library with ReAct loop, multi-provider model support, persistent sessions, agent identity, summarization, subagenting, skills, and workspace-confined file/shell environments.

## Install

```
pip install -e .
# Optional extras: pip install -e ".[browser]" for browser tools
```

## Quick Start

```python
import asyncio
from arion_agent import create_arion_agent
from arion_agent.identity import STANDARD_SOUL, STANDARD_DEEPMEMORY

async def main():
    agent = create_arion_agent(
        model="openai:gpt-4o-mini",
        workspace_dir="./my_workspace",
        agent_id="my-agent",
        soul=STANDARD_SOUL,
        deep_memory=STANDARD_DEEPMEMORY,
    )
    result = await agent.ainvoke(
        {"messages": [("user", "Hello, what is your name?")]},
        config={"configurable": {"thread_id": "my-session"}},
    )
    print(result["messages"][-1].content)
    print(agent.stats.summary())

asyncio.run(main())
```

## Project Structure

```
arion_agent/                        53 Python files, 15 modules
    __init__.py                     Public API: create_arion_agent, estimate_tokens
    graph.py                        ReAct loop (LangGraph StateGraph)
    session.py                      SQLite checkpointer for persistent sessions

    providers/
        resolver.py                 Model string -> BaseChatModel resolution
        moonshot.py                 ChatMoonshot adapter (reasoning_content round-trip)

    tool_manager/
        executor.py                 Central tool executor: timeout, truncation, error handling

    middleware/
        base.py                     ArionMiddleware base class (4 lifecycle hooks + tools)
        patch_tool_calls.py         Patches dangling tool calls, orphaned tool messages, tool_call_id sanitization
        stats.py                    StatsMiddleware (token tracking, session logging)

    identity/                       Phase 3: Agent identity
        config.py                   SoulConfig, MemoryConfig, ShallowMemoryConfig
        templates.py                STANDARD_SOUL, TASK_SOUL, STANDARD_DEEPMEMORY
        middleware.py               IdentityMiddleware (self-awareness, file-based identity)

    summarization/                  Phase 4: History compression
        config.py                   SummarizationPolicy, PolicyDecision, events
        policies.py                 STANDARD_POLICY, AGGRESSIVE_POLICY
        prompts.py                  TASK/PERPETUAL prompt templates and wrappers
        middleware.py               SummarizationMiddleware

    subagenting/                    Phase 5: Child agent spawning
        config.py                   SubAgentSpec, SubagentEvent
        templates.py                SELF_CLONE, SELF_INFERTILE_CLONE, TASK_SUBAGENT
        prompts.py                  System prompt sections for fertile agents
        middleware.py               SubagentMiddleware (task tool, fire-and-forget)
        managed.py                  ManagedSubagentMiddleware (spawn/send/read/dismiss)

    skills/                         Phase 6: Progressive disclosure skills
        config.py                   SkillMetadata, SKILL.md parsing (YAML + XML)
        prompts.py                  DEFAULT_SKILL_INSTRUCTIONS
        middleware.py               SkillMiddleware (scan, catalog, inject)

    environments/
        agentic_core/               Agent reasoning/lifecycle tools (3 tools)
            config.py               PlanConfig, default plan templates and prompts
        signal/                     Signal environment (2 tools, optional)
            config.py               SignalConfig, SignalHub
            store.py                SignalStore (memory + JSONL + archival)
        heartbeat/                  Heartbeat environment (optional, no tools)
            config.py               HeartbeatConfig, EventTrigger, HibernationTrigger, FieldHandler
            middleware.py           HeartbeatEnvironment (system prompt pointer)
            scheduler.py            HeartbeatScheduler (process-level orchestrator)
            effectors.py            SyntheticPromptEffector, SpawnAgent, FileOp, Callback, Composite
            parser.py               HEARTBEAT_SCHEDULE.md parser
        file/                       File operations (7 tools)
        shell/                      Shell execution + terminals (9 tools)
        _sandbox/                   Path confinement and workspace config

    util/
        tokens.py                   Token estimation (English + CJK)
        persistence.py              seed_file, append_jsonl, load_jsonl, read_last_jsonl_record
        runtime.py                  Runtime detection (is_container for Docker-safe SQLite)
        stats.py                    AgentStats, SessionLogger
        timezone.py                 AgentClock (cross-cutting timezone utility)
```

## create_arion_agent Parameters

```python
create_arion_agent(
    model,                          # Required. "provider:model" or BaseChatModel
    workspace_dir,                  # Required. Workspace root directory
    *,
    agent_id=None,                  # Stable ID for resumable agents (None=auto)
    soul=None,                      # SoulConfig, string, or None
    deep_memory=None,               # MemoryConfig, string, or None
    shallow_memory=None,            # ShallowMemoryConfig or None
    pinned_instructions=None,       # Non-editable guardrails
    tools=None,                     # Custom tools (list[BaseTool])
    middleware=None,                # Custom middleware (list[ArionMiddleware])
    subagents=None,                 # SubAgentSpec list (None=infertile)
    skills=None,                    # SkillMiddleware instance (None=no skills)
    summarization=None,             # SummarizationMiddleware (None=default, False=off)
    planning=None,                  # PlanConfig (None=default, False=off)
    enable_status=False,            # get_running_status tool (session metrics)
    signals=None,                   # SignalConfig (None=off)
    heartbeat=None,                 # HeartbeatConfig (None=off)
    timezone="UTC",                 # Agent-wide IANA timezone
    session_log=False,              # JSONL session logging
    checkpointer=True,              # True=SQLite, False=stateless
    tool_executor=None,             # Custom ToolExecutor
    recursion_limit=200,            # Max graph steps
    max_recursion_depth=None,       # Max subagent nesting (None=unlimited)
)
```

Returns a compiled LangGraph with `.stats` (AgentStats), `.agent_id`, and optionally `.heartbeat_scheduler` attached.

## Model Resolution

Three ways to specify the model:

```python
# 1. Provider:model string (auto-routes to correct SDK)
create_arion_agent(model="openai:gpt-5-mini", ...)
create_arion_agent(model="anthropic:claude-sonnet-4-5", ...)
create_arion_agent(model="moonshot:kimi-k2.5", ...)
create_arion_agent(model="deepseek:deepseek_v4_flash", ...)

# 2. Plain model name (auto-detect provider)
create_arion_agent(model="gpt-5-mini", ...)       # -> openai
create_arion_agent(model="claude-sonnet-4-5", ...) # -> anthropic
create_arion_agent(model="kimi-k2.5", ...)         # -> moonshot

# 3. Pre-built BaseChatModel (full control)
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="gpt-5-mini",
    base_url="https://my-proxy.com/v1",
    api_key="sk-...",
    max_tokens=4096,
)
create_arion_agent(model=model, ...)
```

Use option 3 when you need custom LLM configuration that string shortcuts
don't cover: proxy chains, custom output parsers, non-standard providers,
or model-specific parameters. The pre-built instance is used as-is with
no modification.

`**model_kwargs` (e.g. `temperature=0.7`) are forwarded to `init_chat_model`
for string-based model specs only. They are ignored for pre-built instances.

Hot-switch mid-session: `config={"configurable": {"model": "anthropic:claude-sonnet-4-5"}}`

### Supported Providers

| Provider | Env var | Pattern | Adapter |
|---|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-*`, `o1-*`, `chatgpt*` | `init_chat_model` (langchain-openai) |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-*` | `init_chat_model` (langchain-anthropic) |
| Google GenAI | `GOOGLE_API_KEY` | `gemini-*` | `init_chat_model` (langchain-google-genai) |
| Moonshot (Kimi) | `MOONSHOT_API_KEY` | `kimi-*`, `moonshot-*` | `ChatMoonshot` (built-in) |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-*`, `deepseek_v4_*` | `ChatDeepSeek` (built-in) |

Moonshot/Kimi uses a dedicated `ChatMoonshot` adapter (not generic `ChatOpenAI`) to
preserve `reasoning_content` during tool-calling round-trips. Kimi K2.5's thinking
mode is enabled by default at full capability.

DeepSeek V4 Pro / Flash uses `ChatDeepSeek` with the same reasoning round-trip.
Model specs accept underscores: `deepseek:deepseek_v4_flash` maps to `deepseek-v4-flash`.
Thinking mode is enabled by default for DeepSeek models.

Adding another OpenAI-compatible provider: add an entry to `_OPENAI_COMPATIBLE`
in `resolver.py` and optionally a custom adapter in `providers/`.

## Agent Identity (Phase 3)

```python
from arion_agent.identity import STANDARD_SOUL, STANDARD_DEEPMEMORY, SoulConfig

create_arion_agent(soul=STANDARD_SOUL, deep_memory=STANDARD_DEEPMEMORY, ...)
create_arion_agent(soul="You are a Python expert.", ...)
create_arion_agent(soul=SoulConfig(initial_template="...", instructions="..."), ...)
```

System prompt uses XML tags: `<role>`, `<agent_identity>`, `<soul>`, `<deep_memory>`, `<pinned_instructions>`. Agent knows its own agent_id and identity directory.

## Summarization (Phase 4 + Phase 11 Persistence)

Default on with STANDARD_POLICY (80 messages or 85% context window).

```python
from arion_agent.summarization import SummarizationMiddleware, SummarizationPolicy

create_arion_agent(summarization=False, ...)                 # disable
create_arion_agent(summarization=SummarizationMiddleware(     # custom
    policy=SummarizationPolicy(trigger_messages=40, keep_messages=10),
    summary_budget=800,
), ...)
```

Two prompt categories: TASK_SUMMARY_PROMPT (session-based) and PERPETUAL_SUMMARY_PROMPT (operational). Budget defaults scale with context window: 20% of `max_tokens` for summary, 30% for trim. Falls back to fixed 600/690 tokens when `max_tokens` is not set.

**Persistence and eviction**: After summarization, evicted messages are removed from the active state via `RemoveMessage` but retained in prior checkpoint snapshots. The active checkpoint stays bounded regardless of conversation length. Agent-facing markdown transcripts are written to `conversation_history/` for the agent to reference evicted content.

**Interruption safety**: Persistent writes (JSONL record, store archival) are deferred until `drain_state_updates()`, which runs just before the checkpoint saves. If the process dies during the LLM call, no persistent side effects exist -- the agent recovers on restart.

**Recovery**: On restart, if the checkpoint has accumulated too many messages (eviction failed or was never applied), recovery summarization triggers automatically with a WARNING. Corrupted JSONL raises an explicit error and routes into recovery.

## Subagenting (Phase 5)

```python
from arion_agent.subagenting import SubAgentSpec, TASK_SUBAGENT

create_arion_agent(
    subagents=[
        SubAgentSpec(name="researcher", description="...", soul="...", tier="important"),
        TASK_SUBAGENT,
    ],
    max_recursion_depth=2,
    ...
)
```

Two schemes: **task tool** (fire-and-forget, default) and **managed** (spawn/send/read/dismiss).

## Skills (Phase 6)

```python
from arion_agent.skills import SkillMiddleware

create_arion_agent(
    skills=SkillMiddleware(important_skills=["web-research", "code-review"]),
    ...
)
```

Skills follow [Agent Skills open standard](https://agentskills.io). YAML or XML frontmatter. Two tiers: important (in system prompt) and generic (catalog file). Skills define agent classes: same environments, different skills = different specializations.

## Work Planning (Phase 9)

Default on. File-based planning with two scopes and five sections.

```python
from arion_agent.environments.agentic_core import PlanConfig

# Defaults (no argument needed): personal + project plans enabled
agent = create_arion_agent(...)

# Disable planning
agent = create_arion_agent(..., planning=False)

# Enable plan guard (opt-in): nudge the agent to finish pending items
agent = create_arion_agent(..., planning=PlanConfig(max_nudges=3))

# Enable session status tool (default off)
agent = create_arion_agent(..., enable_status=True)
```

Structured plan items stored as JSON at `.arion/agents/{id}/plan.json`. Each item has: id, description, status (pending, in_progress, completed, deprioritized). Plan enforcement is opt-in via `PlanConfig.max_nudges` (default `0`, guard disabled). When set to a positive integer, the agent stopping with incomplete items triggers a synthetic system message nudging it to continue or explicitly deprioritize, up to `max_nudges` times per user turn. Enforcement wording is added to the tool description and system prompt only when enabled; custom `tool_description` / `system_instructions` are used verbatim.

## Signal Environment (Phase 10)

Optional, default off. Structured message-passing for human-in-the-loop, agent-driven hooks, and cross-agent coordination.

```python
from arion_agent.environments.signal import SignalConfig, SignalHub

# Basic: per-agent signals (no cross-agent relay)
agent = create_arion_agent(..., signals=SignalConfig())

# With hub: cross-agent relay in the same process
hub = SignalHub(registry_path="./workspace/.arion/signal_hub.json")
agent_a = create_arion_agent(..., agent_id="alice", signals=SignalConfig(hub=hub))
agent_b = create_arion_agent(..., agent_id="bob", signals=SignalConfig(hub=hub))
# alice's signal_send -> automatically relayed to bob's signal_check

# Custom retention
agent = create_arion_agent(..., signals=SignalConfig(max_signals_per_channel=50))
```

Two tools: **signal_send** (post to a channel) and **signal_check** (read recent from a channel). Signals are always file-persistent at `.arion/agents/{id}/signals/{channel}.jsonl`. Old signals are archived (not discarded) when the file exceeds `2 * max_signals_per_channel` on startup. SignalHub maintains a persistent JSON registry so it can relay to agents not yet instantiated after a script restart.

## Heartbeat Environment (Phase 13, optional)

Optional, default off. Periodic, event-driven, and lifecycle triggers for perpetual agents.

```python
from arion_agent.environments.heartbeat import HeartbeatConfig, HibernationTrigger, SyntheticPromptEffector

agent = create_arion_agent(
    ...,
    heartbeat=HeartbeatConfig(timezone="America/New_York"),
    timezone="America/New_York",
)

# Scheduler is attached but not auto-started
await agent.heartbeat_scheduler.start()
# ... agent runs perpetually ...
await agent.heartbeat_scheduler.stop()
```

Three trigger types:
- **Periodic** (time-driven): defined in `HEARTBEAT_SCHEDULE.md` using cron syntax. Agent-editable — the agent can add, remove, or modify entries using standard file tools. The schedule file is self-documenting with a management section.
- **Event** (condition-driven): registered programmatically — `signal_received`, `file_changed`, `custom` poll function.
- **Hibernation** (lifecycle-driven): fires on every agent creation or script restart, before the tick loop.

```python
from arion_agent.environments.heartbeat import (
    HeartbeatConfig, EventTrigger, HibernationTrigger,
    SyntheticPromptEffector, CallbackEffector, FieldHandler,
)

config = HeartbeatConfig(
    timezone="America/New_York",
    tick_interval=60,
    hibernation_triggers=[
        HibernationTrigger(
            name="startup",
            effector=SyntheticPromptEffector(
                prepend="[Startup at {timestamp}. Last active: {last_active}. First run: {is_first_run}]",
                body="Review pending tasks and catch up.",
                append="[Don't drop existing todos. Integrate new work into existing obligations.]",
            ),
        ),
    ],
    event_triggers=[
        EventTrigger(
            name="config_watch",
            type="file_changed",
            watch_paths=["config.yaml"],
            effector=SyntheticPromptEffector(
                prepend="[File changed: {changed_path} at {timestamp}]",
                body="Review the configuration change.",
            ),
        ),
    ],
)
```

Extensible via `FieldHandler`: developers register custom field handlers (e.g., `skip_if_holiday`, `max_runs_per_day`) that gate or enrich periodic triggers. The schedule file's management section auto-documents available extensions.

## Statistics and Logging (Phase 7)

```python
agent = create_arion_agent(..., session_log=True)
result = await agent.ainvoke(...)
print(agent.stats.summary())     # model calls, tool calls, tokens
# Session log at: .arion/agents/{id}/session_logs/session.jsonl
```

## Tools (19 built-in + 2 optional signal + 10 optional browser)

**File** (7): read_file, write_file, edit_file, delete_file, move_file, list_files, set_directory

**Shell** (9): execute_python, execute_shell_inline, terminal_create/input/read/interrupt/reset/close/list

**Agentic Core** (3): maintenance_tool, update_plan (default on, with plan enforcement), get_running_status (default off)

**Signal** (2, optional): signal_send, signal_check

**Browser** (10, optional): browser_action, browser_snapshot, browser_screenshot, browser_wait_for_human, browser_console, browser_eval_js, http_request, browser_status, browser_reconnect, browser_close

## Browser Environment (Phase 8, optional)

```python
from arion_agent.environments.browser import BrowserEnvironment, BrowserConfig, get_browser_skill_names
from arion_agent.skills import SkillMiddleware

browser = BrowserEnvironment(
    BrowserConfig(headless=False, stealth=True, humanize=True),
    workspace_dir="./workspace",
)

agent = create_arion_agent(
    ...,
    middleware=[browser],
    skills=SkillMiddleware(important_skills=get_browser_skill_names()["important"]),
)
```

Requires:
- Python: `pip install arion-agent[browser]` (or `pip install playwright playwright-stealth aiohttp`)
- Browsers: `playwright install chromium`
- Linux system deps (libglib, libnss3, etc.): `playwright install-deps` or `playwright install --with-deps chromium`. If install-deps fails, install manually, e.g. `apt-get update && apt-get install -y libglib2.0-0` (and other deps as Playwright reports)

### Docker / Remote Server Deployment

When the agent runs inside a Docker container (or any remote server), a locally launched browser is invisible to the host user. Use CDP or WebSocket mode to connect to a browser running on the host machine.

**Recommended pattern: CDP with a host-side persistent browser**

1. On the host, launch a persistent Chromium with CDP enabled:
```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        user_data_dir="./browser_data",
        headless=False,
        args=["--remote-debugging-port=9222"],
    )
    # Browser stays open until this script exits
    input("Press Enter to close browser...")
```

2. In the agent's config, connect via CDP:
```python
browser = BrowserEnvironment(
    BrowserConfig(cdp_endpoint="http://host.docker.internal:9222"),
    workspace_dir="./workspace",
)
```

3. In `docker-compose.yml`, pass the endpoint:
```yaml
environment:
  - CDP_ENDPOINT=http://host.docker.internal:9222
```

**Browser resilience (three levels):**
- Tab-level: if the user closes a tab, the agent recreates it on next use.
- Connection-level: if the CDP connection drops (network blip, container restart) but the browser is still running, `browser_reconnect` revives the connection and recovers open tabs with their URLs and state intact.
- Process-level: if the browser process is killed (user close, crash, taskkill), the CDP endpoint disappears. The agent's `browser_reconnect` will fail until the browser is relaunched. For production deployments, run a supervisor that auto-relaunches the browser on exit. The agent then reconnects transparently on the next `browser_reconnect` call. Login state persists via `user_data_dir`.

**Recommended: supervised browser for production.**
Wrap the browser launch in a process supervisor (systemd, a Python watchdog, or your app's service manager) that restarts the browser on crash/exit. The agent handles reconnection automatically -- it does not need to know the browser restarted. This makes the browser effectively immortal from the agent's perspective.

**Process safety:** `reconnect()` properly closes the old browser process (local mode) before launching a new one, preventing orphaned Chromium process accumulation. In CDP mode, `reconnect()` only disconnects the Playwright client without killing the remote browser.

### SQLite on Docker Bind Mounts (Windows)

ArionAgent auto-detects Docker container environments and uses a safe SQLite configuration that works on bind-mounted host directories. No user configuration needed.

When running inside a container, the checkpointer uses the `unix-none` VFS (disables file locking) with `journal_mode=DELETE` and `mmap_size=0`. This avoids the `flock()`/`fcntl()` calls that grpcfuse/9p do not support, while keeping the database file on the bind mount (visible on the host, portable via zip).

This is safe because ArionAgent guarantees single-process access per checkpoint file.

## Custom Middleware and Environments

Custom environments can bundle skills that teach the agent how to use
the environment's tools effectively. Use `seed_file` and
`get_browser_skill_names()` as a reference pattern.

```python
from arion_agent.middleware import ArionMiddleware

class MyMiddleware(ArionMiddleware):
    @property
    def tools(self): return [my_tool]
    def before_agent(self, state): ...
    def wrap_model_call(self, messages, tools, **kw): ...
    def wrap_tool_call(self, name, args, result): ...
    def drain_state_updates(self): ...    # return [RemoveMessage, ...] for state mutations
    def after_agent(self, state): ...
```

## Workspace Layout

```
workspace/
  .arion/
    plan.md                          # project plan (shared)
    signal_hub.json                  # hub registry (if using SignalHub)
    terminals/                       # shared across all agents
    agents/{agent_id}/               # per-agent state
      checkpoints.sqlite             # LangGraph checkpoint (per-agent, bounded)
      SOUL.md, DEEPMEMORY.md         # identity files (seed-if-absent)
      plan.md                        # personal plan
      skills/important/              # skill folders
      skills/catalog.md              # generic skill index
      conversation_history/          # agent-facing markdown transcripts
        {thread_id}/*.md             #   one file per summarization event
      session_logs/                  # JSONL telemetry
      signals/                       # signal channels (JSONL)
        archive/                     # archived signal batches
      HEARTBEAT_SCHEDULE.md          # periodic heartbeat schedule (agent-editable)
      heartbeat_log.jsonl            # heartbeat execution log
  (user files)                       # agent's working area
```

**Scope distinction**: `checkpoints.sqlite` is per-agent (under `agents/{agent_id}/`). Everything under that directory is agent-specific (identity, checkpoints, transcripts, signals). Workspace-level shared resources (plan.md, signal_hub.json, terminals/) live directly under `.arion/`.

## Workspace Portability

The workspace directory is designed to be self-contained for all persistent state. To clone or migrate an agent swarm:

```
zip -r workspace_backup.zip workspace/
# on another machine:
unzip workspace_backup.zip
# re-create agents with the same create_arion_agent() parameters
```

**What the workspace captures** (complete, file-based):
- Agent identity (SOUL.md, DEEPMEMORY.md, SHALLOW_MEMORY.md)
- Checkpoint state (checkpoints.sqlite per agent, bounded)
- Agent-facing transcripts (conversation_history/*.md)
- Work plans (plan.md at both agent and project level)
- Signals (signals/*.jsonl + archive)
- Session logs (session_logs/*.jsonl)
- Skills (skills/ folder tree)
- Terminal state (terminals/)

**What the workspace does NOT capture** (code-level, must be provided at construction):
- Model spec (which LLM to use)
- Custom tools and middleware (Python objects)
- Summarization policy parameters (trigger/keep thresholds)
- Subagent specs (SubAgentSpec roster)
- API keys
- The ArionAgent library itself

On restart or clone, `create_arion_agent()` must be called with compatible parameters. Identity files are preserved (seed-if-absent contract), and the agent resumes from its checkpointed state.

## Running Tests

```bash
pip install -e .

# Mock tests (no API calls, fast)
python -m pytest tests/test_agentic_core.py tests/test_compression.py tests/test_docker_sqlite.py tests/test_skills.py tests/test_signal.py tests/test_heartbeat.py -v -k mock

# Unit tests (no API calls)
python -m pytest tests/test_docker_sqlite.py tests/test_agentic_core.py tests/test_signal.py tests/test_heartbeat.py -v

# Full test suite (requires API keys)
python -m pytest tests/test_identity.py tests/test_compression.py tests/test_subagenting.py tests/test_skills.py tests/test_agentic_core.py tests/test_signal.py tests/test_heartbeat.py -v

# Hot-switch test: cross-provider model switching with persistent session
python -m tests.test_hot_switch --quick             # 1 permutation of [gemini, gpt, claude, kimi]
python -m tests.test_hot_switch                     # all 24 permutations
python -m tests.test_hot_switch --order kimi,gpt    # custom subset
```

## Playground

```bash
cd playground
python simulate_user_session.py              # non-interactive multi-phase test (quick, reduced triggers)
python final_exam.py                         # full-capability stress test: builds a poker game
python test_summarization_persistence.py     # summarization persistence + eviction + message store
python compare_deepagent_tokens.py           # compare tool token overhead vs Deep Agents
```

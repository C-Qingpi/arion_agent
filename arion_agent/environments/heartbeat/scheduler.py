"""HeartbeatScheduler: process-level orchestrator for heartbeat triggers.

Runs independently of the ReAct loop. Monitors periodic schedules (from
HEARTBEAT_SCHEDULE.md), event triggers, and hibernation triggers. Fires
effectors when conditions are met.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from croniter import croniter

from arion_agent.environments.heartbeat.config import (
    BaseEffector,
    EventTrigger,
    HeartbeatConfig,
    TriggerContext,
)
from arion_agent.environments.heartbeat.effectors import SyntheticPromptEffector
from arion_agent.environments.heartbeat.parser import (
    normalize_cron,
    parse_interval_seconds,
    parse_schedule_full,
)
from arion_agent.util.persistence import append_jsonl, load_jsonl
from arion_agent.util.timezone import AgentClock

logger = logging.getLogger(__name__)


class HeartbeatScheduler:
    """Process-level heartbeat orchestrator.

    Lifecycle:
    1. start() fires hibernation triggers, then begins the tick loop.
    2. Each tick: parse schedule, evaluate periodic + event triggers.
    3. stop() cancels the loop.
    """

    def __init__(
        self,
        agent: Any,
        config: HeartbeatConfig,
        identity_dir: Path,
        workspace_dir: Path,
        clock: AgentClock | None = None,
    ) -> None:
        self._agent = agent
        self._config = config
        self._identity_dir = identity_dir
        self._workspace_dir = workspace_dir
        self._clock = clock or AgentClock(config.timezone)
        self._self_heal = config.self_heal

        self._schedule_path = identity_dir / "HEARTBEAT_SCHEDULE.md"
        self._log_path = identity_dir / "heartbeat_log.jsonl"

        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._lock = asyncio.Lock()

        self._activity_path = identity_dir / "last_activity.json"

        self._last_fired: dict[str, datetime] = {}
        self._event_last_fired: dict[str, datetime] = {}
        self._file_mtimes: dict[str, float] = {}
        self._last_heal_errors: set[tuple[str, str]] = set()

        self._load_last_fired()

    async def start(self) -> None:
        """Fire hibernation triggers, then start the tick loop."""
        if self._running:
            return

        self._running = True

        await self._fire_hibernation_triggers()

        self._task = asyncio.create_task(self._tick_loop())
        logger.info(
            "HeartbeatScheduler started (tick=%ds, tz=%s)",
            self._config.tick_interval,
            self._clock.timezone_name,
        )

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._log_execution("shutdown", "lifecycle", self._clock.now())
        logger.info("HeartbeatScheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    async def _tick_loop(self) -> None:
        """Main scheduler loop. Ticks at config.tick_interval."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("HeartbeatScheduler tick error")
            try:
                await asyncio.sleep(self._config.tick_interval)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        """Evaluate all triggers for the current tick."""
        now = self._clock.now()

        await self._evaluate_periodic_triggers(now)
        await self._evaluate_event_triggers(now)

    async def _evaluate_periodic_triggers(self, now: datetime) -> None:
        """Parse schedule file and fire any due periodic triggers.

        If parse errors are detected, fires a self-heal prompt to the agent
        so it can fix its own schedule file.
        """
        from arion_agent.util.persistence import file_exists, read_file_text
        if not file_exists(self._schedule_path):
            return

        try:
            text = read_file_text(self._schedule_path)
        except OSError:
            logger.warning("Failed to read heartbeat schedule at %s", self._schedule_path)
            return

        result = parse_schedule_full(text)
        all_errors: list[tuple[str, str]] = list(result.errors)

        for block in result.blocks:
            name = block["name"]
            cron_expr = block.get("cron", "")

            interval_seconds = parse_interval_seconds(cron_expr)
            if interval_seconds is not None:
                if self._should_fire_interval(name, interval_seconds, now):
                    await self._fire_periodic(name, block, now)
                continue

            normalized = normalize_cron(cron_expr)
            if normalized is None:
                continue

            try:
                if not croniter.match(normalized, now):
                    continue
            except (ValueError, KeyError) as e:
                all_errors.append((name, f"invalid cron expression '{cron_expr}': {e}"))
                continue

            if self._was_recently_fired(name, now):
                continue

            suppressed = False
            context = self._build_context(name, "periodic", now, block)
            for field_name, handler in self._config.field_handlers.items():
                if field_name in block:
                    if not handler.should_fire(block[field_name], block, context):
                        suppressed = True
                        break
            if suppressed:
                continue

            for field_name, handler in self._config.field_handlers.items():
                if field_name in block:
                    handler.enrich_context(block[field_name], block, context)

            await self._fire_periodic(name, block, now, context)

        if all_errors and self._self_heal:
            await self._fire_self_heal(all_errors, now)

    async def _fire_periodic(
        self,
        name: str,
        block: dict[str, str],
        now: datetime,
        context: TriggerContext | None = None,
    ) -> None:
        """Fire a periodic trigger's effector."""
        if context is None:
            context = self._build_context(name, "periodic", now, block)

        effector = self._resolve_effector(block)
        if effector is None:
            logger.warning("Unknown effector type '%s' for '%s'", block.get("effector"), name)
            return

        if self._config.allow_concurrent:
            await self._execute_effector(effector, context)
        else:
            async with self._lock:
                await self._execute_effector(effector, context)

        self._last_fired[name] = now
        self._log_execution(name, "periodic", now)

    async def _fire_self_heal(
        self,
        errors: list[tuple[str, str]],
        now: datetime,
    ) -> None:
        """Send a synthetic prompt to the agent about schedule parse errors.

        Deduplicates: only fires when errors change from the last time.
        """
        error_set = set(errors)
        if error_set == self._last_heal_errors:
            return
        self._last_heal_errors = error_set

        from arion_agent.util.persistence import workspace_relative_path

        rel_path = workspace_relative_path(self._schedule_path, self._workspace_dir)
        error_lines = "\n".join(f"  - {name}: {reason}" for name, reason in errors)

        prompt = (
            f"[Heartbeat self-heal at {self._clock.format_iso(now)}]\n"
            f"Your heartbeat schedule file ({rel_path}) has entries that the "
            f"scheduler cannot process:\n{error_lines}\n"
            f"These entries are being skipped. Read the file and fix the "
            f"malformed entries so they can fire correctly. Refer to the "
            f"'How to manage this schedule' section in the file for the "
            f"required format."
        )

        thread_id = self._config.thread_id or f"heartbeat-self-heal-{now.strftime('%Y-%m-%d')}"
        logger.info("Heartbeat self-heal: notifying agent of %d parse error(s)", len(errors))

        try:
            if self._config.allow_concurrent:
                await self._agent.ainvoke(
                    {"messages": [("user", prompt)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
            else:
                async with self._lock:
                    await self._agent.ainvoke(
                        {"messages": [("user", prompt)]},
                        config={"configurable": {"thread_id": thread_id}},
                    )
        except Exception:
            logger.exception("Heartbeat self-heal invocation failed")

    async def _evaluate_event_triggers(self, now: datetime) -> None:
        """Poll all registered event triggers."""
        for trigger in self._config.event_triggers:
            if trigger.cooldown > 0:
                last = self._event_last_fired.get(trigger.name)
                if last and (now - last).total_seconds() < trigger.cooldown:
                    continue

            fired = False
            context = self._build_context(trigger.name, "event", now)

            if trigger.type == "file_changed":
                fired = self._check_file_changed(trigger, context)
            elif trigger.type == "signal_received":
                fired = await self._check_signal_received(trigger, context)
            elif trigger.type == "custom" and trigger.poll_fn:
                event_data = await trigger.poll_fn()
                if event_data is not None:
                    fired = True
                    for k, v in event_data.items():
                        context.set(k, str(v))

            if fired:
                if self._config.allow_concurrent:
                    await self._execute_effector(trigger.effector, context)
                else:
                    async with self._lock:
                        await self._execute_effector(trigger.effector, context)

                self._event_last_fired[trigger.name] = now
                self._log_execution(trigger.name, f"event:{trigger.type}", now)

    async def _fire_hibernation_triggers(self) -> None:
        """Fire all hibernation triggers once during start()."""
        now = self._clock.now()
        last_active = self._get_last_active_time()

        for trigger in self._config.hibernation_triggers:
            delta = (now - last_active).total_seconds() if last_active is not None else None

            if trigger.condition is not None and not trigger.condition(delta):
                logger.debug(
                    "Hibernation trigger '%s' suppressed by condition (offline=%s)",
                    trigger.name, f"{delta:.0f}s" if delta is not None else "first-run",
                )
                self._log_execution(trigger.name, "hibernation", now)
                continue

            context = self._build_context(trigger.name, "hibernation", now)

            if last_active is not None:
                context.set("last_active", last_active.isoformat())
                context.set("offline_duration", self._clock.format_duration(delta))
                context.set("is_first_run", "false")
            else:
                context.set("last_active", "never")
                context.set("offline_duration", "first run")
                context.set("is_first_run", "true")

            await self._execute_effector(trigger.effector, context)
            self._log_execution(trigger.name, "hibernation", now)

    def _build_context(
        self,
        trigger_name: str,
        trigger_type: str,
        now: datetime,
        block: dict[str, str] | None = None,
    ) -> TriggerContext:
        ctx = TriggerContext(
            timestamp=self._clock.format_iso(now),
            timestamp_utc=now.astimezone(timezone.utc).isoformat(),
            weekday=now.strftime("%A"),
            trigger_name=trigger_name,
            trigger_type=trigger_type,
            agent_id=getattr(self._agent, "agent_id", "unknown"),
        )
        if block:
            for k, v in block.items():
                if k not in ("name", "cron", "effector") and not k.startswith("_"):
                    ctx.set(k, v)
        return ctx

    def _resolve_effector(self, block: dict[str, str]) -> BaseEffector | None:
        """Resolve an effector from a parsed schedule block.

        Resolution order:
        1. Built-in type ("synthetic_prompt")
        2. Developer-provided effector_factory (for custom types)

        Thread_id resolution: per-trigger field > config-level > auto-generated.
        EffectorDefaults: applied when the per-trigger prepend/append are empty.
        """
        effector_type = block.get("effector", "").strip().lower()

        if effector_type == "synthetic_prompt":
            prepend = block.get("prompt_prepend", "")
            body = block.get("prompt_body", "")
            append = block.get("prompt_append", "")

            defaults = self._config.effector_defaults
            if defaults:
                if not prepend and defaults.prepend:
                    prepend = defaults.prepend
                if not append and defaults.append:
                    append = defaults.append

            thread_id = block.get("thread_id") or self._config.thread_id

            return SyntheticPromptEffector(
                prepend=prepend,
                body=body,
                append=append,
                thread_id=thread_id,
            )

        if self._config.effector_factory is not None:
            return self._config.effector_factory(block)

        return None

    async def _execute_effector(self, effector: BaseEffector, context: TriggerContext) -> None:
        try:
            if self._config.before_invoke is not None:
                gate = await self._config.before_invoke(effector, context)
                if gate is None:
                    logger.debug(
                        "before_invoke suppressed trigger '%s'",
                        context.get("trigger_name"),
                    )
                    return

            result = await effector.execute(context, agent=self._agent)
            self.record_activity()

            if self._config.after_invoke is not None:
                await self._config.after_invoke(result, context)
        except Exception as exc:
            logger.exception(
                "Effector execution failed for trigger '%s'",
                context.get("trigger_name"),
            )
            if self._config.on_error is not None:
                try:
                    await self._config.on_error(exc, context)
                except Exception:
                    logger.exception("on_error callback failed")

    def _should_fire_interval(self, name: str, interval_seconds: int, now: datetime) -> bool:
        last = self._last_fired.get(name)
        if last is None:
            return True
        elapsed = (now - last).total_seconds()
        return elapsed >= interval_seconds

    def _was_recently_fired(self, name: str, now: datetime) -> bool:
        """Prevent double-firing within the same tick window (tick_interval)."""
        last = self._last_fired.get(name)
        if last is None:
            return False
        return (now - last).total_seconds() < self._config.tick_interval

    def _check_file_changed(self, trigger: EventTrigger, context: TriggerContext) -> bool:
        """Check if any watched files have been modified since last check."""
        if not trigger.watch_paths:
            return False

        for rel_path in trigger.watch_paths:
            full_path = self._workspace_dir / rel_path
            if not full_path.exists():
                continue
            try:
                mtime = full_path.stat().st_mtime
            except OSError:
                continue

            cache_key = f"{trigger.name}:{rel_path}"
            prev_mtime = self._file_mtimes.get(cache_key)
            self._file_mtimes[cache_key] = mtime

            if prev_mtime is not None and mtime > prev_mtime:
                context.set("changed_path", rel_path)
                context.set("change_type", "modified")
                return True

        return False

    async def _check_signal_received(self, trigger: EventTrigger, context: TriggerContext) -> bool:
        """Check if new signals arrived on the watched channel."""
        if not trigger.channel:
            return False

        signal_dir = self._identity_dir / "signals"
        channel_file = signal_dir / f"{trigger.channel}.jsonl"
        if not channel_file.exists():
            return False

        cache_key = f"signal:{trigger.name}"
        try:
            mtime = channel_file.stat().st_mtime
        except OSError:
            return False

        prev_mtime = self._file_mtimes.get(cache_key)
        self._file_mtimes[cache_key] = mtime

        if prev_mtime is not None and mtime > prev_mtime:
            records = load_jsonl(channel_file)
            if records:
                latest = records[-1]
                context.set("channel", trigger.channel)
                context.set("sender", latest.get("sender", ""))
                context.set("signal_type", latest.get("type", ""))
                context.set("signal_content", latest.get("content", ""))
                return True

        return False

    def _log_execution(self, trigger_name: str, trigger_type: str, now: datetime) -> None:
        record = {
            "trigger_name": trigger_name,
            "trigger_type": trigger_type,
            "timestamp": self._clock.format_iso(now),
            "timestamp_utc": now.astimezone(timezone.utc).isoformat(),
        }
        append_jsonl(self._log_path, record)
        self._trim_log()

    def _trim_log(self) -> None:
        """Cap heartbeat_log.jsonl at max_log_entries."""
        from arion_agent.util.persistence import file_exists, rewrite_jsonl
        if not file_exists(self._log_path):
            return
        records = load_jsonl(self._log_path)
        if len(records) <= self._config.max_log_entries:
            return
        keep = records[-self._config.max_log_entries:]
        rewrite_jsonl(self._log_path, keep)

    def _load_last_fired(self) -> None:
        """Load last-fired timestamps from heartbeat_log.jsonl for dedup."""
        records = load_jsonl(self._log_path)
        for r in records:
            name = r.get("trigger_name")
            ts_str = r.get("timestamp")
            if name and ts_str:
                try:
                    self._last_fired[name] = self._clock.parse(ts_str)
                except (ValueError, OSError):
                    pass

    def record_activity(self, now: datetime | None = None) -> None:
        """Record the current time as the last agent interaction.

        Called internally after effector execution (heartbeat-driven model
        calls) and externally by the host after user-initiated model calls.
        """
        ts = now or self._clock.now()
        try:
            self._activity_path.write_text(
                json.dumps({"timestamp": self._clock.format_iso(ts)}),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to write last activity timestamp")

    def _get_last_active_time(self) -> datetime | None:
        """Get the timestamp of the last agent interaction.

        Reads from last_activity.json (written after every model call).
        Falls back to the last heartbeat log entry for backward compat.
        """
        if self._activity_path.exists():
            try:
                data = json.loads(
                    self._activity_path.read_text(encoding="utf-8")
                )
                if "timestamp" in data:
                    return self._clock.parse(data["timestamp"])
            except (json.JSONDecodeError, ValueError, OSError):
                pass
        if not self._log_path.exists():
            return None
        from arion_agent.util.persistence import read_last_jsonl_record
        last = read_last_jsonl_record(self._log_path)
        if last and "timestamp" in last:
            try:
                return self._clock.parse(last["timestamp"])
            except (ValueError, OSError):
                pass
        return None

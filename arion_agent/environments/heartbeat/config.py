"""Heartbeat environment configuration.

Defines the config dataclasses for triggers (periodic, event, hibernation),
the FieldHandler extension interface, and HeartbeatConfig.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


class TriggerContext:
    """Mutable context dict populated during trigger evaluation.

    Holds template variables available to effectors. Core fields are set
    by the scheduler; field handlers and event sources add custom entries
    via enrich_context().
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, str] = {k: str(v) for k, v in kwargs.items()}

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(key, default)

    def as_dict(self) -> dict[str, str]:
        return dict(self._data)

    def format_template(self, template: str) -> str:
        """Apply {variable} substitution. Unknown variables are left as-is."""
        result = template
        for k, v in self._data.items():
            result = result.replace("{" + k + "}", v)
        return result


class FieldHandler(ABC):
    """Extension interface for developer-defined schedule fields.

    Registered in HeartbeatConfig.field_handlers keyed by field name.
    The scheduler calls these during periodic trigger evaluation.
    """

    def should_fire(self, field_value: str, trigger: dict[str, str], context: TriggerContext) -> bool:
        """Return False to suppress this trigger for the current tick."""
        return True

    def enrich_context(self, field_value: str, trigger: dict[str, str], context: TriggerContext) -> None:
        """Add custom template variables to context before the effector runs."""

    def management_doc(self) -> str:
        """Return documentation text for this field, included in the
        'How to manage this schedule' section when seeding the file."""
        return ""


class BaseEffector(ABC):
    """Base class for all heartbeat effectors."""

    @abstractmethod
    async def execute(self, context: TriggerContext, agent: Any = None) -> None:
        """Execute the effector action.

        Args:
            context: Trigger context with template variables.
            agent: The compiled agent graph (for effectors that invoke it).
        """


@dataclass
class EventTrigger:
    """Configuration for a condition-driven event trigger.

    Attributes:
        name: Unique identifier for this trigger.
        type: Built-in type: 'signal_received', 'file_changed', or 'custom'.
        effector: What to do when the event fires.
        channel: For signal_received: which signal channel to watch.
        watch_paths: For file_changed: list of relative paths to watch.
        poll_fn: For custom: async callable returning event data or None.
        cooldown: Minimum seconds between consecutive firings.
    """

    name: str
    type: str
    effector: BaseEffector
    channel: str | None = None
    watch_paths: list[str] | None = None
    poll_fn: Callable[[], Awaitable[dict[str, Any] | None]] | None = None
    cooldown: float = 0.0


@dataclass
class HibernationTrigger:
    """Configuration for a lifecycle trigger that fires on startup.

    Fires once during HeartbeatScheduler.start(), before the tick loop.
    When condition is set, it receives the offline duration in seconds
    (None on first-ever run) and must return True to allow firing.
    """

    name: str
    effector: BaseEffector
    condition: Callable[[float | None], bool] | None = None


@dataclass
class EffectorDefaults:
    """Default prepend/append for synthetic prompt effectors.

    Applied when the effector's own prepend/append is empty.
    """

    prepend: str = ""
    append: str = ""


@dataclass
class HeartbeatConfig:
    """Configuration for the heartbeat environment.

    Attributes:
        timezone: IANA timezone name for the agent (e.g. 'America/New_York').
        tick_interval: Seconds between scheduler evaluation ticks.
        allow_concurrent: Allow parallel heartbeat invocations.
        self_heal: When True (default), parse errors in the schedule file are
            sent to the agent as a synthetic prompt so it can fix its own file.
        thread_id: Default thread_id for all heartbeat-triggered invocations.
            None (default) = auto-generate per trigger per day (isolated threads).
            A string = all heartbeat prompts share this thread. Supports
            {variable} templates. Use "main" or "default" to inject heartbeats
            into the agent's primary conversation thread.
        before_invoke: Async callable intercepting invocations before agent.ainvoke.
            Signature: async (prompt: str, thread_id: str, context: TriggerContext)
                -> (prompt, thread_id) to proceed, or None to suppress.
        after_invoke: Async callable called after agent.ainvoke completes.
            Signature: async (result: dict, context: TriggerContext) -> None
        on_error: Async callable called when an effector raises an exception.
            Signature: async (error: Exception, context: TriggerContext) -> None
        effector_factory: Callable that resolves custom effector types from a
            parsed schedule block. Signature: (block: dict[str, str]) -> BaseEffector | None.
            Called when the built-in type mapping returns None.
        coalesce_window_multiplier: Missed-beat catch-up window as multiple of interval.
        max_log_entries: Cap on heartbeat_log.jsonl entries.
        initial_schedule: Seed content for HEARTBEAT_SCHEDULE.md. Empty = auto-assembled.
        event_triggers: Developer-registered event triggers.
        hibernation_triggers: Developer-registered startup triggers.
        field_handlers: Extension fields for periodic trigger evaluation.
        effector_defaults: Default prepend/append for synthetic prompts.
            Applied when the per-trigger fields are empty.
    """

    timezone: str = "UTC"
    tick_interval: int = 60
    allow_concurrent: bool = False
    self_heal: bool = True
    thread_id: str | None = None
    before_invoke: Callable[..., Awaitable[Any]] | None = None
    after_invoke: Callable[..., Awaitable[Any]] | None = None
    on_error: Callable[..., Awaitable[Any]] | None = None
    effector_factory: Callable[..., Any] | None = None
    coalesce_window_multiplier: float = 2.0
    max_log_entries: int = 500
    initial_schedule: str = ""
    event_triggers: list[EventTrigger] = field(default_factory=list)
    hibernation_triggers: list[HibernationTrigger] = field(default_factory=list)
    field_handlers: dict[str, FieldHandler] = field(default_factory=dict)
    effector_defaults: EffectorDefaults | None = None

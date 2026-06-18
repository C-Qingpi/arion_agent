"""Heartbeat environment middleware.

Seeds HEARTBEAT_SCHEDULE.md and injects a lightweight pointer into the
system prompt. No dedicated tools — the agent uses existing file tools.
The schedule document is self-documenting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from arion_agent.environments.heartbeat.config import HeartbeatConfig
from arion_agent.middleware.base import ArionMiddleware
from arion_agent.util.persistence import ensure_directory, seed_file
from arion_agent.util.timezone import AgentClock

logger = logging.getLogger(__name__)

BASE_MANAGEMENT_TEMPLATE = """\
# Heartbeat Schedule

## How to manage this schedule
This file controls your periodic heartbeat triggers. The scheduler reads it
and fires entries when their cron time arrives. You own this file -- add,
remove, or edit entries as needed.

Format for each entry (one block per trigger):
  ## periodic: <unique_name>
  cron: <5-field cron expression>    (required)
  effector: <effector type>          (required)
  description: <what this heartbeat does>
  prompt_prepend: <context frame injected before the body>
  prompt_body: <the core instruction>
  prompt_append: <post-instruction guidance>
  thread_id: <conversation thread to use> (optional, omit for default)
  (any additional key: value lines are passed to registered handlers)

Comments: lines starting with // or > are notes. The scheduler preserves
them but does not act on them. Use comments for context, rationale, or
instructions to your future self.
  // This is a comment
  > This is also a comment

Cron syntax: minute hour day-of-month month day-of-week
  Examples: 0 9 * * 1-5 (weekdays 9 AM), 0 */2 * * * (every 2 hours)
  Shortcuts: @hourly, @daily, @weekly, @monthly, @yearly
  Intervals: every 15m, every 2h, every 30s

Template variables available in prompt fields:
  {timestamp} - fire time in agent timezone
  {trigger_name} - name of this entry
  {agent_id} - your agent ID

{extension_fields_doc}To add a new periodic task: copy an existing block, give it a unique name.
To remove: delete the entire ## periodic: block.
To pause: comment out lines with # or remove the block temporarily.
To adjust timing: change the cron expression.

---
"""


class HeartbeatEnvironment(ArionMiddleware):
    """Middleware providing heartbeat schedule awareness to the agent.

    Seeds HEARTBEAT_SCHEDULE.md on construction (seed-if-absent).
    Injects a minimal system prompt section with file location and current time.
    No tools — the agent uses existing file tools.
    """

    def __init__(
        self,
        agent_id: str,
        identity_dir: Path,
        workspace_dir: Path,
        clock: AgentClock,
        config: HeartbeatConfig | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._identity_dir = identity_dir
        self._workspace_dir = workspace_dir
        self._clock = clock
        self._config = config or HeartbeatConfig()

        self._schedule_path = identity_dir / "HEARTBEAT_SCHEDULE.md"
        self._log_path = identity_dir / "heartbeat_log.jsonl"

        self._seed_files()

    @property
    def schedule_path(self) -> Path:
        return self._schedule_path

    @property
    def log_path(self) -> Path:
        return self._log_path

    def _seed_files(self) -> None:
        ensure_directory(self._identity_dir)
        content = self._config.initial_schedule
        if not content:
            content = self._assemble_management_template()
        seed_file(self._schedule_path, content)

    def _assemble_management_template(self) -> str:
        """Build schedule seed content from base template + field handler docs."""
        extension_docs = []
        for field_name, handler in self._config.field_handlers.items():
            doc = handler.management_doc()
            if doc:
                extension_docs.append(f"  {field_name}: {doc}")

        if extension_docs:
            ext_section = "Available extension fields:\n" + "\n".join(extension_docs) + "\n\n"
        else:
            ext_section = ""

        return BASE_MANAGEMENT_TEMPLATE.replace("{extension_fields_doc}", ext_section)

    def wrap_system_message(self, parts: list[str], **kwargs: Any) -> list[str]:
        from arion_agent.util.persistence import workspace_relative_path

        rel_schedule = workspace_relative_path(self._schedule_path, self._workspace_dir)
        rel_log = workspace_relative_path(self._log_path, self._workspace_dir)
        now_str = self._clock.format_iso()

        parts.append(
            f"<heartbeat>\n"
            f"Heartbeat schedule: {rel_schedule}\n"
            f"Heartbeat log: {rel_log}\n"
            f"Timezone: {self._clock.timezone_name}\n"
            f"Current time: {now_str}\n"
            f"</heartbeat>"
        )
        return parts

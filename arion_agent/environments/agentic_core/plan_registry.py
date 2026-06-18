"""Structured work plan registry for programmatic plan enforcement.

The plan has five semantic sections:
  - deliverables:  what to deliver, quality/acceptance criteria
  - methodology:   binding constraints on how work is carried out
  - context:       reference material (task background + working refs)
  - items:         tactical work items (the only enforced section)
  - confirmation:  self-audit notes before reporting completion

Only the items array is programmatically enforced (nudge on premature
stop). The other four sections are narrative text preserved alongside
items for semantic guidance.

Thread isolation:
  Plan state is tracked per thread_id. The agent may serve multiple
  conversation threads via config["configurable"]["thread_id"]. The
  registry uses a swap-on-switch pattern: when set_active_thread is
  called, the current thread's state is cached in memory and the
  target thread's state is restored (from cache or disk). All public
  methods operate on the active thread's data.

Lifecycle:
  - Created by AgenticCoreEnvironment at agent construction time
  - set_active_thread called at the start of each ainvoke (patched_ainvoke)
  - before_agent resets nudge count for the active thread
  - Written to by the update_plan tool
  - Read by the graph's plan_guard node to decide whether to nudge
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "deprioritized"})
ACTIVE_STATUSES = frozenset({"pending", "in_progress"})


@dataclass
class PlanItem:
    """A single structured plan item."""

    id: str
    description: str
    status: str = "pending"


class PlanRegistry:
    """In-memory registry of a structured work plan with nudge tracking.

    Stores both narrative sections (deliverables, methodology, context,
    confirmation) and an enforced items array. Plan state is isolated
    per thread_id using a swap-on-switch pattern.
    """

    def __init__(
        self,
        max_nudges: int = 3,
        plans_dir: Path | None = None,
    ) -> None:
        self.max_nudges: int = max_nudges
        self._plans_dir: Path | None = plans_dir
        self._thread_cache: dict[str, dict[str, Any]] = {}
        self._active_thread: str = "default"

        # Current-thread mutable state (swapped on thread switch)
        self.items: list[PlanItem] = []
        self.deliverables: str = ""
        self.methodology: str = ""
        self.context: str = ""
        self.confirmation: str = ""
        self.nudge_count: int = 0

        if plans_dir is not None:
            self._try_load_from_disk()

    # ---- Thread management ----

    @property
    def active_thread(self) -> str:
        return self._active_thread

    def set_active_thread(self, thread_id: str) -> None:
        """Switch to a different thread's plan state.

        Saves current thread to cache, restores or loads the target
        thread. No-op if already on the requested thread.
        """
        if thread_id == self._active_thread:
            return

        self._thread_cache[self._active_thread] = self._snapshot()
        self._active_thread = thread_id

        if thread_id in self._thread_cache:
            self._restore(self._thread_cache[thread_id])
        elif self._plans_dir is not None:
            self._try_load_from_disk()
        else:
            self._reset_plan_data()

    def get_persist_path(self) -> Path | None:
        """Return the file path for the active thread's plan, or None."""
        if self._plans_dir is None:
            return None
        return self._plans_dir / f"{self._active_thread}.json"

    def _snapshot(self) -> dict[str, Any]:
        """Capture current-thread state as a dict."""
        return {
            "items": list(self.items),
            "deliverables": self.deliverables,
            "methodology": self.methodology,
            "context": self.context,
            "confirmation": self.confirmation,
            "nudge_count": self.nudge_count,
        }

    def _restore(self, snap: dict[str, Any]) -> None:
        """Restore current-thread state from a snapshot."""
        self.items = snap["items"]
        self.deliverables = snap["deliverables"]
        self.methodology = snap["methodology"]
        self.context = snap["context"]
        self.confirmation = snap["confirmation"]
        self.nudge_count = snap["nudge_count"]

    def _reset_plan_data(self) -> None:
        """Clear all plan data for a fresh thread."""
        self.items = []
        self.deliverables = ""
        self.methodology = ""
        self.context = ""
        self.confirmation = ""
        self.nudge_count = 0

    def _try_load_from_disk(self) -> None:
        """Try to load the active thread's plan from disk."""
        if self._plans_dir is None:
            return
        path = self._plans_dir / f"{self._active_thread}.json"
        if not path.exists():
            self._reset_plan_data()
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.set_plan(data)
            elif isinstance(data, list):
                self._reset_plan_data()
                self.set_items(data)
            else:
                self._reset_plan_data()
            logger.debug(
                "Loaded plan (%d items) for thread '%s' from %s",
                len(self.items), self._active_thread, path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load plan from %s: %s", path, exc)
            self._reset_plan_data()

    # ---- Mutators ----

    def set_items(self, raw_items: list[dict[str, Any]]) -> list[PlanItem]:
        """Validate and replace all plan items. Returns the validated items."""
        validated: list[PlanItem] = []
        for raw in raw_items:
            status = raw.get("status", "pending")
            if status not in VALID_STATUSES:
                status = "pending"
            validated.append(PlanItem(
                id=str(raw.get("id", "unnamed")),
                description=str(raw.get("description", "")),
                status=status,
            ))
        self.items = validated
        return validated

    def set_plan(self, raw: dict[str, Any]) -> list[PlanItem]:
        """Set all plan sections from a parsed JSON object.

        Accepts a dict with optional keys: deliverables, methodology,
        context, items (array), confirmation. Missing keys are left
        unchanged (allows partial updates).
        """
        if "deliverables" in raw:
            self.deliverables = str(raw["deliverables"])
        if "methodology" in raw:
            self.methodology = str(raw["methodology"])
        if "context" in raw:
            self.context = str(raw["context"])
        if "confirmation" in raw:
            self.confirmation = str(raw["confirmation"])
        raw_items = raw.get("items", [])
        if isinstance(raw_items, list):
            return self.set_items(raw_items)
        return list(self.items)

    # ---- Queries ----

    def has_pending_work(self) -> bool:
        """True if any item is pending or in_progress."""
        return any(i.status in ACTIVE_STATUSES for i in self.items)

    def should_nudge(self) -> bool:
        """True if there is pending work and nudge budget remains."""
        return self.has_pending_work() and self.nudge_count < self.max_nudges

    def reset_nudge_count(self) -> None:
        """Reset nudge counter. Called at the start of each user turn."""
        self.nudge_count = 0

    def format_nudge_message(self) -> str:
        """Build a synthetic user message nudging the agent to continue.

        Increments the nudge counter. Returns a clearly-labeled system
        message with the list of incomplete items and an explicit call
        to action. Includes a deliverables reminder when available to
        keep the agent focused on the goal.
        """
        active = [i for i in self.items if i.status in ACTIVE_STATUSES]
        remaining = self.max_nudges - self.nudge_count - 1
        lines = [
            "[SYSTEM - Plan Enforcement]",
            "",
            "Your plan has items not yet completed or deprioritized:",
        ]
        for item in active:
            lines.append(f"  - [{item.status}] {item.id}: {item.description}")
        if self.deliverables:
            snippet = self.deliverables[:300]
            if len(self.deliverables) > 300:
                snippet += "..."
            lines.append("")
            lines.append(f"Deliverables: {snippet}")
        lines.append("")
        lines.append(
            "Either continue working on these items, or call update_plan "
            "to explicitly mark them as deprioritized."
        )
        if remaining <= 1:
            lines.append(
                "This is your final reminder. Unaddressed items will be "
                "left incomplete when the session ends."
            )
        self.nudge_count += 1
        return "\n".join(lines)

    def pending_summary(self) -> str:
        """Short summary of incomplete items for tool feedback."""
        active = [i for i in self.items if i.status in ACTIVE_STATUSES]
        if not active:
            return "All items completed or deprioritized."
        parts = [f"[{i.status}] {i.id}" for i in active]
        return f"{len(active)} incomplete: {', '.join(parts)}"

    # ---- Serialization ----

    def to_dict(self) -> dict[str, Any]:
        """Full plan as a dict for serialization."""
        return {
            "deliverables": self.deliverables,
            "methodology": self.methodology,
            "context": self.context,
            "items": [asdict(i) for i in self.items],
            "confirmation": self.confirmation,
        }

    def to_json(self) -> str:
        """Serialize full plan to JSON for file persistence."""
        return json.dumps(self.to_dict(), indent=2)

    # ---- Factory methods ----

    @classmethod
    def load_from_file(cls, path: Path, max_nudges: int = 3) -> PlanRegistry:
        """Create a registry pre-populated from a single JSON file.

        Convenience for testing or single-thread usage. For per-thread
        persistence, use the plans_dir constructor parameter instead.
        """
        registry = cls(max_nudges=max_nudges)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    registry.set_plan(data)
                elif isinstance(data, list):
                    registry.set_items(data)
                logger.debug(
                    "Loaded plan (%d items) from %s", len(registry.items), path,
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load plan from %s: %s", path, exc)
        return registry

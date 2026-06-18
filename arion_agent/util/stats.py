"""Agent session statistics: token usage, call counts, and optional JSONL logging."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arion_agent.util.persistence import append_jsonl

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    """Cumulative statistics for an agent session."""

    model_calls: int = 0
    tool_calls: int = 0
    summarization_events: int = 0
    subagent_spawns: int = 0
    input_tokens_estimated: int = 0
    output_tokens_estimated: int = 0
    input_tokens_actual: int = 0
    output_tokens_actual: int = 0
    total_messages: int = 0
    session_start: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.session_start

    def summary(self) -> str:
        lines = [
            f"Model calls: {self.model_calls}",
            f"Tool calls: {self.tool_calls}",
            f"Summarizations: {self.summarization_events}",
            f"Subagent spawns: {self.subagent_spawns}",
            f"Tokens (estimated): input={self.input_tokens_estimated}, output={self.output_tokens_estimated}",
        ]
        if self.input_tokens_actual:
            lines.append(
                f"Tokens (actual): input={self.input_tokens_actual}, output={self.output_tokens_actual}"
            )
        lines.append(f"Elapsed: {self.elapsed_seconds:.1f}s")
        return "\n".join(lines)


class SessionLogger:
    """Append-only JSONL session logger for structured event tracking."""

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data: Any) -> None:
        record = {
            "event": event,
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **data,
        }
        try:
            append_jsonl(self._path, record)
        except OSError:
            logger.warning("Failed to write session log to %s", self._path)

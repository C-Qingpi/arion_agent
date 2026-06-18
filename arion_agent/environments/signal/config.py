"""Signal environment configuration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arion_agent.environments.signal.store import SignalStore

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for the signal environment.

    Args:
        hub: Optional SignalHub for cross-agent relay.
        max_signals_per_channel: In-memory cap and post-archival target.
        archive_threshold: File record count that triggers archival on load.
            Defaults to 2 * max_signals_per_channel.
    """

    hub: SignalHub | None = None
    max_signals_per_channel: int = 100
    archive_threshold: int | None = None

    def effective_archive_threshold(self) -> int:
        if self.archive_threshold is not None:
            return self.archive_threshold
        return 2 * self.max_signals_per_channel


class SignalHub:
    """Cross-agent signal relay with persistent registry.

    Maintains a JSON file mapping agent_id -> signal_dir so it can relay
    signals to agents that are not yet instantiated in the current process.
    """

    def __init__(self, registry_path: Path | str) -> None:
        self._registry_path = Path(registry_path)
        self._stores: dict[str, SignalStore] = {}
        self._registry: dict[str, str] = {}
        self._load_registry()

    def register(self, agent_id: str, store: SignalStore) -> None:
        """Register an agent's live store. Persists agent_id -> signal_dir."""
        self._stores[agent_id] = store
        self._registry[agent_id] = str(store.signal_dir)
        self._save_registry()

    def relay(self, signal: dict[str, Any], from_agent: str) -> None:
        """Relay a signal to all registered agents except the sender.

        For live agents: calls store.receive() (memory + JSONL).
        For non-live agents: appends directly to their JSONL file.
        """
        from arion_agent.util.persistence import append_jsonl

        channel = signal["channel"]
        for agent_id in list(self._registry):
            if agent_id == from_agent:
                continue
            if agent_id in self._stores:
                self._stores[agent_id].receive(signal)
            else:
                signal_dir = Path(self._registry[agent_id])
                channel_file = signal_dir / f"{channel}.jsonl"
                append_jsonl(channel_file, signal)

    def _load_registry(self) -> None:
        from arion_agent.util.persistence import file_exists, read_file_text
        if file_exists(self._registry_path):
            try:
                self._registry = json.loads(read_file_text(self._registry_path))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load signal hub registry at %s", self._registry_path)
                self._registry = {}

    def _save_registry(self) -> None:
        from arion_agent.util.persistence import write_file as persistence_write
        content = json.dumps(self._registry, indent=2, ensure_ascii=False)
        persistence_write(self._registry_path, content)
            raise

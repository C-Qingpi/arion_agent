"""Signal environment middleware: contributes signal_send and signal_check tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from arion_agent.environments.signal.config import SignalConfig
from arion_agent.environments.signal.store import SignalStore
from arion_agent.environments.signal.tools import create_signal_tools
from arion_agent.middleware.base import ArionMiddleware


class SignalEnvironment(ArionMiddleware):
    """Middleware providing signal-based coordination tools.

    Stores signals per-agent at workspace/.arion/agents/{agent_id}/signals/.
    If a SignalHub is configured, registers with it for cross-agent relay.
    """

    def __init__(
        self,
        agent_id: str,
        workspace_dir: Path,
        config: SignalConfig | None = None,
        clock: Any | None = None,
    ) -> None:
        cfg = config or SignalConfig()
        signal_dir = Path(workspace_dir) / ".arion" / "agents" / agent_id / "signals"

        self._store = SignalStore(
            signal_dir=signal_dir,
            max_per_channel=cfg.max_signals_per_channel,
            archive_threshold=cfg.effective_archive_threshold(),
            hub=cfg.hub,
            clock=clock,
        )

        if cfg.hub is not None:
            cfg.hub.register(agent_id, self._store)

        self._tools = create_signal_tools(agent_id, self._store)

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    @property
    def store(self) -> SignalStore:
        return self._store

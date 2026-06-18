"""Signal storage: in-memory cache backed by per-channel JSONL files."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arion_agent.util.persistence import append_jsonl, ensure_directory, load_jsonl

logger = logging.getLogger(__name__)


class SignalStore:
    """Per-agent signal storage with JSONL backing and archival.

    Each channel maps to one JSONL file: {signal_dir}/{channel}.jsonl
    Archived overflow goes to: {signal_dir}/archive/{channel}/{timestamp}.jsonl
    """

    def __init__(
        self,
        signal_dir: Path,
        max_per_channel: int = 100,
        archive_threshold: int = 200,
        hub: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        self._signal_dir = signal_dir
        self._max = max_per_channel
        self._archive_threshold = archive_threshold
        self._hub = hub
        self._clock = clock
        self._channels: dict[str, list[dict[str, Any]]] = {}
        self._counters: dict[str, int] = {}

        ensure_directory(signal_dir)
        self._load_existing()

    @property
    def signal_dir(self) -> Path:
        return self._signal_dir

    def send(
        self,
        channel: str,
        sender: str,
        signal_type: str,
        content: str,
    ) -> dict[str, Any]:
        """Create and persist a signal. Relays via hub if present."""
        sig_id = self._next_id(channel)
        if self._clock is not None:
            ts = self._clock.format_iso()
        else:
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        signal = {
            "id": sig_id,
            "timestamp": ts,
            "sender": sender,
            "channel": channel,
            "type": signal_type,
            "content": content,
        }

        self._append(channel, signal)
        self._append_to_file(channel, signal)

        if self._hub is not None:
            self._hub.relay(signal, from_agent=sender)

        return signal

    def receive(self, signal: dict[str, Any]) -> None:
        """Accept a relayed signal from the hub. No further relay."""
        channel = signal["channel"]
        self._append(channel, signal)
        self._append_to_file(channel, signal)
        sig_num = self._parse_id_num(signal.get("id", "sig-0"))
        if sig_num >= self._counters.get(channel, 1):
            self._counters[channel] = sig_num + 1

    def check(self, channel: str, last_n: int = 10) -> list[dict[str, Any]]:
        """Return the last N signals from a channel (from memory)."""
        signals = self._channels.get(channel, [])
        return signals[-last_n:]

    def _append(self, channel: str, signal: dict[str, Any]) -> None:
        if channel not in self._channels:
            self._channels[channel] = []
        self._channels[channel].append(signal)
        if len(self._channels[channel]) > self._max:
            self._channels[channel] = self._channels[channel][-self._max :]

    def _append_to_file(self, channel: str, signal: dict[str, Any]) -> None:
        path = self._signal_dir / f"{channel}.jsonl"
        append_jsonl(path, signal)

    def _next_id(self, channel: str) -> str:
        num = self._counters.get(channel, 1)
        self._counters[channel] = num + 1
        return f"sig-{num:03d}"

    @staticmethod
    def _parse_id_num(sig_id: str) -> int:
        try:
            return int(sig_id.split("-", 1)[1])
        except (IndexError, ValueError):
            return 0

    def _load_existing(self) -> None:
        """Scan signal_dir for *.jsonl, load into memory, archive if needed."""
        from arion_agent.util.persistence import file_exists, glob_files
        if not file_exists(self._signal_dir):
            return
        for jsonl_file in glob_files(self._signal_dir, "*.jsonl"):
            channel = jsonl_file.stem
            records = load_jsonl(jsonl_file)
            if not records:
                continue

            if len(records) > self._archive_threshold:
                keep = records[-self._max :]
                archive = records[: len(records) - self._max]
                self._archive_records(channel, archive)
                self._rewrite_active(channel, keep)
                records = keep

            tail = records[-self._max :]
            self._channels[channel] = tail

            max_num = 0
            for r in records:
                num = self._parse_id_num(r.get("id", "sig-0"))
                if num > max_num:
                    max_num = num
            self._counters[channel] = max_num + 1

    def _archive_records(
        self, channel: str, records: list[dict[str, Any]]
    ) -> None:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        archive_dir = self._signal_dir / "archive" / channel
        ensure_directory(archive_dir)
        archive_path = archive_dir / f"{ts}.jsonl"
        for r in records:
            append_jsonl(archive_path, r)
        logger.debug(
            "Archived %d signals for channel '%s' to %s",
            len(records), channel, archive_path,
        )

    def _rewrite_active(
        self, channel: str, keep: list[dict[str, Any]]
    ) -> None:
        from arion_agent.util.persistence import rewrite_jsonl
        path = self._signal_dir / f"{channel}.jsonl"
        rewrite_jsonl(path, keep)

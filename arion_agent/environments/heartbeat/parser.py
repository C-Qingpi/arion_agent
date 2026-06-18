"""HEARTBEAT_SCHEDULE.md parser.

Parses `## periodic: <name>` blocks into flat key-value dicts.
All other content (management section, markdown headers, blank lines) is ignored.
Unknown fields are preserved in the dict -- never rejected.

Comment lines (starting with // or >) inside a block are preserved in the
parsed dict under '_comments' so they survive read-modify-write cycles.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

PERIODIC_HEADER_RE = re.compile(r"^##\s+periodic:\s*(.+)$", re.IGNORECASE)
KV_RE = re.compile(r"^([a-z_][a-z0-9_]*):\s*(.+)$", re.IGNORECASE)
COMMENT_RE = re.compile(r"^(//|>)\s?(.*)")

INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*([smhd])$", re.IGNORECASE)

INTERVAL_UNIT_MAP = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class ParseResult:
    """Result of parsing HEARTBEAT_SCHEDULE.md.

    Attributes:
        blocks: Valid periodic trigger dicts.
        errors: List of (entry_name, reason) for malformed entries.
    """

    __slots__ = ("blocks", "errors")

    def __init__(
        self,
        blocks: list[dict[str, str]],
        errors: list[tuple[str, str]],
    ) -> None:
        self.blocks = blocks
        self.errors = errors


def parse_schedule(text: str) -> list[dict[str, str]]:
    """Parse HEARTBEAT_SCHEDULE.md into a list of valid periodic trigger dicts.

    Convenience wrapper around parse_schedule_full that drops error info.
    """
    return parse_schedule_full(text).blocks


def parse_schedule_full(text: str) -> ParseResult:
    """Parse HEARTBEAT_SCHEDULE.md into valid blocks and error reports.

    Each valid dict has at minimum 'name' and whatever key-value fields
    are in the block. Malformed blocks (missing 'cron' or 'effector')
    are reported in errors and skipped.

    Comment lines (// or >) inside a block are collected into '_comments'.
    """
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    current_comments: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            header_match = PERIODIC_HEADER_RE.match(stripped)
            if header_match:
                if current is not None:
                    if current_comments:
                        current["_comments"] = "\n".join(current_comments)
                    blocks.append(current)
                current = {"name": header_match.group(1).strip()}
                current_comments = []
            elif current is not None and stripped.startswith("## "):
                if current_comments:
                    current["_comments"] = "\n".join(current_comments)
                blocks.append(current)
                current = None
                current_comments = []
            continue

        if current is None:
            continue

        if not stripped:
            continue

        comment_match = COMMENT_RE.match(stripped)
        if comment_match:
            current_comments.append(stripped)
            continue

        kv_match = KV_RE.match(stripped)
        if kv_match:
            key = kv_match.group(1).lower()
            value = kv_match.group(2).strip()
            current[key] = value

    if current is not None:
        if current_comments:
            current["_comments"] = "\n".join(current_comments)
        blocks.append(current)

    valid: list[dict[str, str]] = []
    errors: list[tuple[str, str]] = []
    for block in blocks:
        name = block.get("name", "<unnamed>")
        if "cron" not in block:
            reason = f"missing required 'cron' field"
            logger.warning("Heartbeat schedule: skipping '%s' — %s", name, reason)
            errors.append((name, reason))
            continue
        _try_extract_interval(block)
        if "effector" not in block:
            reason = f"missing required 'effector' field"
            logger.warning("Heartbeat schedule: skipping '%s' — %s", name, reason)
            errors.append((name, reason))
            continue
        valid.append(block)

    return ParseResult(blocks=valid, errors=errors)


def normalize_cron(cron_expr: str) -> str | None:
    """Normalize cron expression, handling shortcuts and interval syntax.

    Returns a standard 5-field cron string, or None if the expression
    is an interval (which needs special handling in the scheduler).
    """
    shortcuts = {
        "@yearly": "0 0 1 1 *",
        "@annually": "0 0 1 1 *",
        "@monthly": "0 0 1 * *",
        "@weekly": "0 0 * * 0",
        "@daily": "0 0 * * *",
        "@midnight": "0 0 * * *",
        "@hourly": "0 * * * *",
    }
    expr = cron_expr.strip()
    if expr.lower() in shortcuts:
        return shortcuts[expr.lower()]
    if INTERVAL_RE.match(expr):
        return None
    return expr


def parse_interval_seconds(expr: str) -> int | None:
    """Parse 'every Xm/Xh/Xs/Xd' into seconds. Returns None if not interval syntax."""
    match = INTERVAL_RE.match(expr.strip())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    return amount * INTERVAL_UNIT_MAP[unit]


def _try_extract_interval(block: dict[str, str]) -> bool:
    """Check if the block's cron field uses interval syntax. Normalize if so."""
    cron = block.get("cron", "")
    seconds = parse_interval_seconds(cron)
    if seconds is not None:
        block["_interval_seconds"] = str(seconds)
        return True
    return False

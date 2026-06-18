"""Shared multimodal utilities for converting images to LangChain messages.

Image tools (read_file for images, browser_screenshot) use the IMAGE_BLOCK_SENTINEL
format with a file-based payload. This module converts file:// references to
proper LangChain multimodal ToolMessages. Base64 in tool output is never decoded;
images are always loaded from file to avoid truncation and padding errors.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote

logger = logging.getLogger(__name__)

IMAGE_BLOCK_SENTINEL = "__IMAGE_BLOCK__"

FILE_PREFIX = "file://"


def convert_image_block(msg: Any) -> Any:
    """Convert __IMAGE_BLOCK__:mime:file://path sentinel to a multimodal ToolMessage.

    Reads image from file and builds a data URL. Never decodes base64 from tool
    output. On any error (missing file, read failure, etc.) returns a safe
    fallback ToolMessage so the agentic cycle continues (tool-failure style).
    """
    from langchain_core.messages import ToolMessage

    if not isinstance(msg, ToolMessage):
        return msg
    content = msg.content
    if not isinstance(content, str) or not content.startswith(IMAGE_BLOCK_SENTINEL):
        return msg

    sentinel_line = content.split("\n", 1)[0]
    parts = sentinel_line.split(":", 2)
    if len(parts) != 3:
        return msg

    _, mime_type, payload = parts

    extra_text = content[len(sentinel_line):].strip()

    try:
        if payload.startswith(FILE_PREFIX):
            path = Path(unquote(payload[len(FILE_PREFIX) :].lstrip("/")))
            if not path.is_absolute():
                path = path.resolve()
            if not path.exists():
                return _fallback_tool_message(msg, f"Image file not found: {path}")
            raw = path.read_bytes()
            size_bytes = len(raw)
            b64 = base64.b64encode(raw).decode("ascii")
        else:
            b64 = payload
            size_bytes = None
        size_text = f"{size_bytes} bytes" if size_bytes is not None else "unknown size"
        text_parts = [f"[Image loaded: {mime_type}, {size_text}. If you cannot see the image, use execute_python to analyze it programmatically.]"]
        if extra_text:
            text_parts.append(extra_text)
        return ToolMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                {"type": "text", "text": "\n".join(text_parts)},
            ],
            name=msg.name,
            tool_call_id=msg.tool_call_id,
        )
    except Exception as exc:
        logger.warning("convert_image_block failed: %s", exc, exc_info=True)
        return _fallback_tool_message(msg, str(exc))


def _fallback_tool_message(msg: Any, reason: str) -> Any:
    """Return a ToolMessage that does not break the agentic cycle."""
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content=f"[Image could not be loaded: {reason}. Continue without the image or retry.]",
        name=msg.name,
        tool_call_id=msg.tool_call_id,
    )

"""Test image reading across GPT, Claude, Gemini, Kimi via CloseAI proxy."""

from __future__ import annotations

import asyncio
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: E402, F401

from arion_agent import create_arion_agent


def _create_red_png(path: Path, width: int = 32, height: int = 32) -> None:
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    compressed = zlib.compress(raw)

    def _chunk(ct: bytes, data: bytes) -> bytes:
        c = ct + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", compressed))
        f.write(_chunk(b"IEND", b""))


MODELS = [
    ("GPT", "openai:gpt-5-mini"),
    ("Claude", "anthropic:claude-sonnet-4-5"),
    ("Gemini3", conftest.get_test_model("gemini-3-flash-preview")),
    ("Kimi", "moonshot:kimi-k2.5"),
]


async def _test_model(label: str, model_spec):
    print(f"\n{'='*60}")
    print(f"Image test: {label}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as ws:
        _create_red_png(Path(ws) / "red.png")
        agent = create_arion_agent(
            model=model_spec,
            soul="Answer briefly. When you see an image, describe its color.",
            workspace_dir=ws,
            checkpointer=False,
        )
        result = await agent.ainvoke({"messages": [("user",
            "Read red.png and tell me the color. One word answer."
        )]})

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))
            if "image_url" in content or IMAGE_BLOCK in content:
                print(f"  [{role}] [IMAGE sent to model]")
            else:
                print(f"  [{role}] {content[:200]}")
            tc = getattr(msg, "tool_calls", None)
            if tc:
                print(f"    tools: {[t['name'] for t in tc]}")

        last_ai = [m for m in result["messages"] if getattr(m, "type", "") == "ai"][-1]
        answer = str(getattr(last_ai, "content", "")).lower()
        assert "red" in answer, f"{label}: expected 'red', got: {answer[:100]}"
        print(f"  >> {label} PASSED")


IMAGE_BLOCK = "__IMAGE_BLOCK__"


async def main():
    for label, model_spec in MODELS:
        try:
            await _test_model(label, model_spec)
        except Exception as exc:
            print(f"  >> {label} FAILED: {type(exc).__name__}: {exc}")

    print(f"\n{'='*60}")
    print("CROSS-PROVIDER IMAGE TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

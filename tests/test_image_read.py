"""Test read_file with an image: create a pure red PNG and ask the agent to identify the color."""

from __future__ import annotations

import asyncio
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from arion_agent import create_arion_agent


def _create_red_png(path: Path, width: int = 64, height: int = 64) -> None:
    """Create a minimal pure red PNG without any image library."""
    raw_rows = []
    for _ in range(height):
        row = b"\x00"  # filter byte: None
        row += b"\xff\x00\x00" * width  # RGB: pure red
        raw_rows.append(row)
    raw_data = b"".join(raw_rows)
    compressed = zlib.compress(raw_data)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr_data))
        f.write(_chunk(b"IDAT", compressed))
        f.write(_chunk(b"IEND", b""))


async def main():
    with tempfile.TemporaryDirectory() as workspace:
        img_path = Path(workspace) / "red_square.png"
        _create_red_png(img_path)
        print(f"Created {img_path} ({img_path.stat().st_size} bytes)")

        agent = create_arion_agent(
            model="gpt-5-mini",
            soul="You are a test agent. Answer questions about images you see.",
            workspace_dir=workspace,
            checkpointer=False,
        )

        print("\n" + "=" * 60)
        print("Test: Read red PNG image and ask agent to identify the color")
        print("=" * 60)

        result = await agent.ainvoke({"messages": [("user",
            "Read the file red_square.png and tell me what color the image is. "
            "Answer with just the color name."
        )]})

        for msg in result["messages"]:
            role = getattr(msg, "type", "?")
            content = str(getattr(msg, "content", ""))
            tc = getattr(msg, "tool_calls", None)
            if role == "tool" and "__IMAGE_BLOCK__" in content:
                print(f"  [{role}] [IMAGE_BLOCK: {content[:60]}...]")
            else:
                print(f"  [{role}] {content[:300]}")
            if tc:
                print(f"    tools: {[t['name'] for t in tc]}")

        last_ai = [m for m in result["messages"] if getattr(m, "type", "") == "ai"][-1]
        answer = str(getattr(last_ai, "content", "")).lower()
        assert "red" in answer, f"Expected 'red' in answer, got: {answer}"
        print("\n  >> IMAGE READ TEST PASSED: Agent correctly identified red")

    print(f"\n{'=' * 60}")
    print("IMAGE READ TEST COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if not os.environ.get("CLOSEAI_API_KEY"):
        print("Set CLOSEAI_API_KEY to run this test.")
        sys.exit(1)
    asyncio.run(main())

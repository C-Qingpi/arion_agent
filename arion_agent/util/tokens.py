"""Token estimation utilities.

Used across the agent: summarization triggers, executor output budgeting,
context window management, etc.
"""

from __future__ import annotations

import re
from typing import Any

_CJK_RANGE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff"
    r"\uac00-\ud7af\uff00-\uffef]"
)

IMAGE_TOKEN_ESTIMATE = 1000


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    English/Latin: ~3.3 chars per token.
    CJK (Chinese, Japanese, Korean): ~1.5 chars per token.
    Uses a weighted blend based on CJK character ratio.
    """
    if not text:
        return 0
    total_chars = len(text)
    cjk_chars = len(_CJK_RANGE.findall(text))
    latin_chars = total_chars - cjk_chars

    tokens = latin_chars / 3.3 + cjk_chars / 1.5
    return max(1, int(tokens))


def estimate_message_tokens(content: str | list | Any) -> int:
    """Estimate tokens for a message's content, handling multimodal blocks.

    Multimodal ToolMessages store content as a list of dicts (image_url, text).
    Stringifying the whole list would count the raw base64 as text tokens,
    massively inflating estimates. Instead we sum text-block tokens and use
    a fixed per-image estimate (most APIs charge ~85-1000 tokens per image).
    """
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        tokens = 0
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    tokens += estimate_tokens(block.get("text", ""))
                elif btype == "image_url":
                    tokens += IMAGE_TOKEN_ESTIMATE
                else:
                    tokens += estimate_tokens(str(block))
            elif isinstance(block, str):
                tokens += estimate_tokens(block)
            else:
                tokens += estimate_tokens(str(block))
        return tokens
    return estimate_tokens(str(content))

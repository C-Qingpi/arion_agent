"""Summarization: compresses old messages into structured summaries.

Phase 15: compression is a graph node, not middleware.
The compress module provides make_compress_node and make_should_compress
factory functions.
"""

from arion_agent.summarization.compress import (
    PrefetchRegistry,
    find_safe_cutoff,
    make_compress_node,
    make_must_compress,
    make_prefetch_node,
    make_route_compression,
    make_should_compress,
    raw_cutoff,
    truncate_args,
    walk_back_past_tool_messages,
)
from arion_agent.summarization.config import (
    OPTIONAL_SECTIONS,
    RECOVERY_THRESHOLD,
    CutoffFn,
    PolicyDecision,
    SummarizationCallback,
    SummarizationEvent,
    SummarizationPolicy,
    SummaryFileState,
    SummaryRecord,
)
from arion_agent.summarization.policies import (
    AGGRESSIVE_POLICY,
    STANDARD_POLICY,
    STANDARD_PREFETCH_POLICY,
)
from arion_agent.summarization.prompts import (
    PERPETUAL_SUMMARY_PROMPT,
    PERPETUAL_WRAPPER,
    PERPETUAL_WRAPPER_NO_PATH,
    TASK_SUMMARY_PROMPT,
    TASK_WRAPPER,
    TASK_WRAPPER_NO_PATH,
)

__all__ = [
    "AGGRESSIVE_POLICY",
    "CutoffFn",
    "PERPETUAL_SUMMARY_PROMPT",
    "PERPETUAL_WRAPPER",
    "PERPETUAL_WRAPPER_NO_PATH",
    "PolicyDecision",
    "RECOVERY_THRESHOLD",
    "PrefetchRegistry",
    "STANDARD_POLICY",
    "STANDARD_PREFETCH_POLICY",
    "SummarizationCallback",
    "SummarizationEvent",
    "SummarizationPolicy",
    "SummaryFileState",
    "SummaryRecord",
    "TASK_SUMMARY_PROMPT",
    "TASK_WRAPPER",
    "TASK_WRAPPER_NO_PATH",
    "find_safe_cutoff",
    "make_compress_node",
    "make_must_compress",
    "make_prefetch_node",
    "make_route_compression",
    "make_should_compress",
    "raw_cutoff",
    "truncate_args",
    "walk_back_past_tool_messages",
]

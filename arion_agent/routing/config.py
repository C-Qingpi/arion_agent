"""Routing configuration for intra-ReAct-loop model switching.

Checkpoint / thread persistence (often confused with "serde strategy"):
  Per-thread graph state is stored under ``configurable["thread_id"]`` via
  LangGraph's SQLite checkpointer. Serialization uses
  ``langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`` — see
  ``arion_agent.session.create_checkpointer`` and ``graph._setup_checkpointer``.
  Jackio enables the checkpointer from ``jackio_agent.create_jackio(..., checkpointer=True)``.

Two signal mechanisms, used together:

1. signal_next_step tool: the model calls this alongside its other tool
   calls to indicate what the NEXT step requires. This is the primary
   mechanism because tool-calling responses have empty text content.

2. [ROUTING: X] text tag: fallback for pure-text responses (final answers,
   mid-loop reasoning without tool calls). Stripped before checkpointing.

Categories:
  think   - complex reasoning, planning, decisions  -> strong
  read    - processing tool results, exploration     -> weak
  write   - generating code, composing content       -> strong
  operate - simple tool dispatch, command execution   -> weak
  digest  - synthesizing/summarizing findings         -> strong

Fallback: when no signal is present, the strong model is used.
Signals from both strong and weak models are trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


WEAK_CATEGORIES = frozenset({"read", "operate"})
STRONG_CATEGORIES = frozenset({"think", "write", "digest"})
ALL_CATEGORIES = WEAK_CATEGORIES | STRONG_CATEGORIES

ROUTING_INSTRUCTION = """\
Step routing: when calling tools, also call signal_next_step to indicate \
what type of processing the NEXT step will require. For text-only responses, \
append [ROUTING: X] on its own line instead.
  think - complex reasoning, planning, or important decisions
  read - processing tool results, gathering information, or exploration
  write - generating code, composing text, or producing file content
  operate - running commands, simple tool dispatch, or navigation
  digest - synthesizing findings, summarizing results, or drawing conclusions
This optimizes which model handles each step."""


@dataclass
class RoutingConfig:
    """Configuration for model routing within the ReAct loop.

    Attributes:
        weak_model: Model spec for fast/cheap steps (read, operate).
            Same format as the default model spec ("provider:model_id").
        instruction: System prompt section explaining the routing protocol.
        enabled: Master switch. Set False to disable routing without
            removing the config.
    """

    weak_model: str = ""
    instruction: str = field(default=ROUTING_INSTRUCTION)
    enabled: bool = True

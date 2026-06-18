"""Configuration for structured work planning in the agentic core.

PlanConfig controls the update_plan tool behavior, system prompt
injection, and plan enforcement parameters (max nudges before the
agent is allowed to stop with incomplete items).

Plan enforcement (the "plan guard") is opt-in. When max_nudges is 0
(the default), the graph never injects continuation nudges and the
tool description / system prompt omit enforcement wording. Set
max_nudges to a positive integer to enable nudging.

The plan is stored as structured JSON items (not markdown). Status
labels and enforcement logic live in plan_registry.py.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PLAN_TOOL_DESCRIPTION = """\
Update the structured work plan, replacing all existing content.

Pass a JSON object with the following sections:
  deliverables - what to deliver, quality and acceptance criteria
  methodology - binding constraints on how work is carried out \
(process, technical, quality gates, domain conventions). All items \
must comply with these.
  context - important reference material for the session. Two kinds: \
(1) task background -- why, user intent, prior decisions, constraints; \
(2) working references -- code, aliases, paths, patterns discovered
  items - JSON array of tactical work items. Each has: \
id (short identifier), description (what to do), \
status (pending / in_progress / completed / deprioritized)
  confirmation - self-audit notes before reporting completion. \
Verify each deliverable is met and methodology was followed.

All sections are optional. A bare JSON array is also accepted as \
items-only shorthand.

Use for complex multi-step work to track progress across turns. \
Update item status as work progresses. Mark completed immediately \
when done. Mark deprioritized (not just leave pending) if an item \
is no longer relevant."""

_PLAN_TOOL_ENFORCEMENT_NOTE = (
    " The system enforces item completion -- incomplete items trigger "
    "a continuation prompt when you attempt to stop."
)

DEFAULT_PLAN_TOOL_DESCRIPTION_WITH_ENFORCEMENT = (
    DEFAULT_PLAN_TOOL_DESCRIPTION + _PLAN_TOOL_ENFORCEMENT_NOTE
)


_PLAN_SYSTEM_PROMPT_BODY = """\
<planning>
You have a structured work plan at {plan_path}.
Use update_plan to manage your plan as a JSON object with five sections:
1) deliverables - what to deliver, acceptance criteria
2) methodology - binding process and quality constraints that govern
   how all work is carried out
3) context - task background and working references (code, paths,
   patterns, environment details). Keeps critical information visible
   across turns. Keep concise, prune when stale.
4) items - array of tactical work items, each with id, description,
   status (pending, in_progress, completed, deprioritized)
5) confirmation - self-audit before reporting completion. Verify each
   deliverable is met and each methodology item was followed.

When to use:
1) Complex multi-step work (3+ steps)
2) When tracking progress across turns

Management rules:
1) Update item status as work progresses
2) Mark items completed immediately when done
3) Add new items discovered during work
4) Mark items deprioritized (not pending) if no longer relevant
5) Keep at least one item in_progress unless all work is done
6) Keep context section updated with discovered references
7) Populate confirmation before declaring completion\
"""

_PLAN_SYSTEM_PROMPT_ENFORCEMENT = """

The system monitors your plan items. If you stop with incomplete
items, you will be prompted to continue or explicitly deprioritize.\
"""

_PLAN_SYSTEM_PROMPT_CLOSE = "\n</planning>"

DEFAULT_PLAN_SYSTEM_PROMPT = _PLAN_SYSTEM_PROMPT_BODY + _PLAN_SYSTEM_PROMPT_CLOSE

DEFAULT_PLAN_SYSTEM_PROMPT_WITH_ENFORCEMENT = (
    _PLAN_SYSTEM_PROMPT_BODY
    + _PLAN_SYSTEM_PROMPT_ENFORCEMENT
    + _PLAN_SYSTEM_PROMPT_CLOSE
)


@dataclass
class PlanConfig:
    """Configuration for the structured work planning tool.

    Customize tool_description or system_instructions to change how
    the tool and planning guidance are presented to the LLM.

    max_nudges controls how many times the system will prompt the
    agent to continue before allowing it to stop with pending items.
    Defaults to 0 (plan guard disabled). Set to a positive integer
    to opt in; when enabled with the default descriptions, enforcement
    wording is automatically appended to both the tool description
    and the system prompt. Custom descriptions are used as-is.
    """

    max_nudges: int = 0
    tool_description: str = DEFAULT_PLAN_TOOL_DESCRIPTION
    system_instructions: str = DEFAULT_PLAN_SYSTEM_PROMPT

    def effective_tool_description(self) -> str:
        """Tool description actually presented to the LLM.

        Returns the enforcement-augmented default when plan guard is
        enabled and the default description is in use. Overrides are
        returned unchanged so callers stay in full control.
        """
        if self.max_nudges > 0 and self.tool_description is DEFAULT_PLAN_TOOL_DESCRIPTION:
            return DEFAULT_PLAN_TOOL_DESCRIPTION_WITH_ENFORCEMENT
        return self.tool_description

    def effective_system_instructions(self) -> str:
        """System prompt section contributed to the system message.

        Returns the enforcement-augmented default when plan guard is
        enabled and the default prompt is in use. Overrides are
        returned unchanged so callers stay in full control.
        """
        if self.max_nudges > 0 and self.system_instructions is DEFAULT_PLAN_SYSTEM_PROMPT:
            return DEFAULT_PLAN_SYSTEM_PROMPT_WITH_ENFORCEMENT
        return self.system_instructions

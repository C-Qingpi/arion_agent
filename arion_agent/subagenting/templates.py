"""Standard subagent spec templates."""

from arion_agent.identity.templates import TASK_SOUL
from arion_agent.subagenting.config import SubAgentSpec

SELF_CLONE = SubAgentSpec(
    name="self-clone",
    description="A clone of the parent agent with full capabilities.",
    soul=None,
    model=None,
    tools=None,
    fertile=True,
    inherit_identity=True,
)

SELF_INFERTILE_CLONE = SubAgentSpec(
    name="self-infertile-clone",
    description="A clone of the parent agent without subagenting ability.",
    soul=None,
    model=None,
    tools=None,
    fertile=False,
    inherit_identity=True,
)

TASK_SUBAGENT = SubAgentSpec(
    name="task-agent",
    description="A minimal task agent for isolated, focused work.",
    soul=TASK_SOUL,
    model=None,
    tools=None,
    fertile=False,
    max_turns=30,
)

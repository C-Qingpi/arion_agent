"""System prompt sections for subagent-capable agents."""

DEFAULT_SUBAGENT_INSTRUCTIONS = (
    "You can delegate tasks to subagents using the task tool. "
    "Use subagents for complex, multi-step tasks that benefit from "
    "isolated context. Do not delegate trivial tasks. "
    "Subagents cannot see your conversation history; provide all "
    "necessary context in the task description. "
    "Subagent results are not visible to the user; summarize results "
    "in your response."
)

TASK_TOOL_DESCRIPTION = """\
Spawn a subagent to handle a task in an isolated context.

The subagent runs independently with its own conversation. It cannot see \
your history. Provide all necessary context in the task description. The \
subagent returns a single result when done.

Use for complex, multi-step tasks. Do not use for trivial one-tool tasks. \
Launch multiple subagents in parallel when tasks are independent.

Available subagent classes:
{available_classes}"""

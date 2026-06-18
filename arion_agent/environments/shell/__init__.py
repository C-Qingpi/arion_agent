"""Shell environment: quick Python execution and recoverable background CLI jobs."""

from arion_agent.environments.shell.backend import (
    LocalShellBackend,
    RemoteShellBackend,
    ShellBackend,
)
from arion_agent.environments.shell.jobs import JobRegistry
from arion_agent.environments.shell.middleware import ShellEnvironment

__all__ = [
    "ShellEnvironment",
    "ShellBackend",
    "LocalShellBackend",
    "RemoteShellBackend",
    "JobRegistry",
]

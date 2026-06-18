"""Agent identity system: SOUL, DEEPMEMORY, SHALLOW_MEMORY configs and middleware."""

from arion_agent.identity.config import (
    MemoryConfig,
    ShallowMemoryConfig,
    SoulConfig,
)
from arion_agent.identity.middleware import IdentityMiddleware
from arion_agent.identity.templates import (
    STANDARD_DEEPMEMORY,
    STANDARD_SHALLOW_MEMORY,
    STANDARD_SOUL,
    TASK_SOUL,
)

__all__ = [
    "IdentityMiddleware",
    "MemoryConfig",
    "ShallowMemoryConfig",
    "SoulConfig",
    "STANDARD_DEEPMEMORY",
    "STANDARD_SHALLOW_MEMORY",
    "STANDARD_SOUL",
    "TASK_SOUL",
]

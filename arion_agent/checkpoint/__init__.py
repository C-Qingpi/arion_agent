"""Checkpoint backends for ArionAgent.

AsyncProxySaver: HTTP-based checkpoint saver for Docker deployments
where direct SQLite access degrades on grpcfuse bind mounts (Phase 17+).
"""

from arion_agent.checkpoint.proxy import AsyncProxySaver

__all__ = ["AsyncProxySaver"]

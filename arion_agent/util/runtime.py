"""Runtime environment detection utilities."""

from __future__ import annotations

import os
import sys

_is_container: bool | None = None


def is_container() -> bool:
    """Detect whether we are running inside a Docker or OCI container.

    Checks for /.dockerenv (Docker) and /run/.containerenv (Podman/Buildah).
    Only returns True on Linux, since containers run a Linux userspace even
    when the host is Windows or macOS.

    Result is cached after first call.
    """
    global _is_container
    if _is_container is None:
        _is_container = (
            sys.platform.startswith("linux")
            and (
                os.path.exists("/.dockerenv")
                or os.path.exists("/run/.containerenv")
            )
        )
    return _is_container

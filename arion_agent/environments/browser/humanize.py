"""Human-like mouse path generation using cubic Bezier curves.

Generates waypoints that mimic natural hand movement: curved paths,
acceleration/deceleration, and micro-jitter. Same mathematical approach
as pyclick/ghost-cursor, without the OS-level mouse dependencies.
"""

from __future__ import annotations

import math
import random


def human_curve(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    curvature: float = 0.3,
    jitter: float = 1.0,
    target_points: int | None = None,
) -> list[tuple[float, float]]:
    """Generate waypoints along a human-like curved path.

    Args:
        start: (x, y) origin.
        end: (x, y) destination.
        curvature: how far control points deviate perpendicular to the
            straight line, as a fraction of distance (0 = straight).
        jitter: standard deviation of per-point micro-noise (pixels).
        target_points: number of waypoints. Defaults to distance-scaled
            value (min 10, roughly 1 point per 10px).

    Returns:
        List of (x, y) tuples from start to end.
    """
    sx, sy = start
    ex, ey = end
    dist = math.hypot(ex - sx, ey - sy)

    if dist < 2:
        return [start, end]

    if target_points is None:
        target_points = max(10, min(80, int(dist / 10)))

    dx, dy = ex - sx, ey - sy
    nx, ny = -dy / dist, dx / dist  # perpendicular unit vector
    spread = dist * curvature

    # Two control points at ~1/3 and ~2/3 along the path,
    # offset perpendicular by a random amount.
    cp1 = (
        sx + dx * random.uniform(0.25, 0.40) + nx * random.gauss(0, spread),
        sy + dy * random.uniform(0.25, 0.40) + ny * random.gauss(0, spread),
    )
    cp2 = (
        sx + dx * random.uniform(0.60, 0.75) + nx * random.gauss(0, spread),
        sy + dy * random.uniform(0.60, 0.75) + ny * random.gauss(0, spread),
    )

    points: list[tuple[float, float]] = []
    last = target_points - 1
    for i in range(target_points):
        t_raw = i / last
        t = _ease_out_quad(t_raw)

        u = 1.0 - t
        x = u**3 * sx + 3 * u**2 * t * cp1[0] + 3 * u * t**2 * cp2[0] + t**3 * ex
        y = u**3 * sy + 3 * u**2 * t * cp1[1] + 3 * u * t**2 * cp2[1] + t**3 * ey

        if 0 < i < last and jitter > 0:
            x += random.gauss(0, jitter)
            y += random.gauss(0, jitter)

        points.append((x, y))

    return points


def _ease_out_quad(t: float) -> float:
    """Deceleration curve — fast start, slow approach to target."""
    return 1.0 - (1.0 - t) ** 2

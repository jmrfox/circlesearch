"""Geometric circle representation and intersection computation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Circle:
    """A circle in the plane defined by center (x, y) and radius r."""

    x: float
    y: float
    r: float

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"Radius must be positive, got {self.r}")

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: Circle) -> float:
        """Euclidean distance between centers."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def intersection_points(self, other: Circle) -> list[tuple[float, float]]:
        """Compute the intersection points with another circle.

        Returns a list of 0 or 2 (x, y) tuples. Tangential cases (1 point)
        are excluded from this problem, but we return an empty list for them.
        """
        d = self.distance_to(other)

        # No intersection: too far apart or one inside the other
        if d >= self.r + other.r or d <= abs(self.r - other.r):
            return []

        # Tangent case (excluded by problem definition)
        if math.isclose(d, self.r + other.r) or math.isclose(d, abs(self.r - other.r)):
            return []

        # Standard two-point intersection
        a = (self.r**2 - other.r**2 + d**2) / (2 * d)
        h = math.sqrt(self.r**2 - a**2)

        # Unit vector from self.center to other.center
        dx = (other.x - self.x) / d
        dy = (other.y - self.y) / d

        # Midpoint along the line between centers
        mx = self.x + a * dx
        my = self.y + a * dy

        # Two intersection points
        p1 = (mx + h * dy, my - h * dx)
        p2 = (mx - h * dy, my + h * dx)
        return [p1, p2]

    def contains_point(self, px: float, py: float) -> bool:
        """Check if a point is strictly inside this circle."""
        return math.hypot(px - self.x, py - self.y) < self.r

    def angle_of(self, px: float, py: float) -> float:
        """Angle (in radians, -pi to pi) of point (px, py) relative to center."""
        return math.atan2(py - self.y, px - self.x)

    def intersects(self, other: Circle) -> bool:
        """Check if this circle has a two-point intersection with another."""
        return len(self.intersection_points(other)) == 2

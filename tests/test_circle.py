"""Tests for circlesearch.circle.Circle."""

import math

import pytest

from circlesearch.circle import Circle


# ── Construction ──────────────────────────────────────────────────


class TestCircleConstruction:
    def test_basic_creation(self):
        c = Circle(1.0, 2.0, 3.0)
        assert c.x == 1.0
        assert c.y == 2.0
        assert c.r == 3.0

    def test_center_property(self):
        c = Circle(1.0, 2.0, 3.0)
        assert c.center == (1.0, 2.0)

    def test_frozen(self):
        c = Circle(0, 0, 1)
        with pytest.raises(AttributeError):
            c.x = 5

    def test_zero_radius_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Circle(0, 0, 0)

    def test_negative_radius_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Circle(0, 0, -1)

    def test_small_positive_radius(self):
        c = Circle(0, 0, 1e-10)
        assert c.r == 1e-10

    def test_equality(self):
        assert Circle(1, 2, 3) == Circle(1, 2, 3)
        assert Circle(1, 2, 3) != Circle(1, 2, 4)

    def test_hashable(self):
        s = {Circle(0, 0, 1), Circle(0, 0, 1), Circle(1, 0, 1)}
        assert len(s) == 2


# ── distance_to ──────────────────────────────────────────────────


class TestDistanceTo:
    def test_same_center(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(0, 0, 2)
        assert c1.distance_to(c2) == 0.0

    def test_horizontal(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(3, 0, 1)
        assert c1.distance_to(c2) == pytest.approx(3.0)

    def test_diagonal(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(3, 4, 1)
        assert c1.distance_to(c2) == pytest.approx(5.0)

    def test_symmetric(self):
        c1 = Circle(1, 2, 1)
        c2 = Circle(4, 6, 1)
        assert c1.distance_to(c2) == pytest.approx(c2.distance_to(c1))


# ── intersection_points ──────────────────────────────────────────


class TestIntersectionPoints:
    def test_disjoint_far(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(5, 0, 1)
        assert c1.intersection_points(c2) == []

    def test_disjoint_nested(self):
        c1 = Circle(0, 0, 3)
        c2 = Circle(0, 0, 1)
        assert c1.intersection_points(c2) == []

    def test_external_tangent_excluded(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(2, 0, 1)
        assert c1.intersection_points(c2) == []

    def test_internal_tangent_excluded(self):
        c1 = Circle(0, 0, 2)
        c2 = Circle(1, 0, 1)
        assert c1.intersection_points(c2) == []

    def test_two_point_intersection(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(1, 0, 1)
        pts = c1.intersection_points(c2)
        assert len(pts) == 2

    def test_intersection_points_lie_on_both_circles(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(1, 0, 1)
        for px, py in c1.intersection_points(c2):
            d1 = math.hypot(px - c1.x, py - c1.y)
            d2 = math.hypot(px - c2.x, py - c2.y)
            assert d1 == pytest.approx(c1.r, abs=1e-10)
            assert d2 == pytest.approx(c2.r, abs=1e-10)

    def test_symmetric_result(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(1, 0, 1)
        pts_12 = c1.intersection_points(c2)
        pts_21 = c2.intersection_points(c1)
        assert len(pts_12) == len(pts_21) == 2
        # Same set of points (order may differ)
        s12 = {(round(x, 10), round(y, 10)) for x, y in pts_12}
        s21 = {(round(x, 10), round(y, 10)) for x, y in pts_21}
        assert s12 == s21

    def test_different_radii(self):
        c1 = Circle(0, 0, 2)
        c2 = Circle(2, 0, 1.5)
        pts = c1.intersection_points(c2)
        assert len(pts) == 2
        for px, py in pts:
            assert math.hypot(px, py) == pytest.approx(2.0, abs=1e-10)
            assert math.hypot(px - 2, py) == pytest.approx(1.5, abs=1e-10)

    def test_nearly_tangent_but_overlapping(self):
        # Just barely overlapping (not tangent)
        c1 = Circle(0, 0, 1)
        c2 = Circle(1.99, 0, 1)
        pts = c1.intersection_points(c2)
        assert len(pts) == 2

    def test_concentric_no_intersection(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(0, 0, 2)
        assert c1.intersection_points(c2) == []

    def test_off_axis_intersection(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(0.5, 0.866, 1)
        pts = c1.intersection_points(c2)
        assert len(pts) == 2


# ── intersects ────────────────────────────────────────────────────


class TestIntersects:
    def test_overlapping(self):
        assert Circle(0, 0, 1).intersects(Circle(1, 0, 1))

    def test_disjoint(self):
        assert not Circle(0, 0, 1).intersects(Circle(5, 0, 1))

    def test_nested(self):
        assert not Circle(0, 0, 3).intersects(Circle(0, 0, 1))

    def test_tangent(self):
        assert not Circle(0, 0, 1).intersects(Circle(2, 0, 1))

    def test_symmetric(self):
        c1 = Circle(0, 0, 1)
        c2 = Circle(1, 0, 1)
        assert c1.intersects(c2) == c2.intersects(c1)


# ── contains_point ────────────────────────────────────────────────


class TestContainsPoint:
    def test_center_inside(self):
        assert Circle(0, 0, 1).contains_point(0, 0)

    def test_outside(self):
        assert not Circle(0, 0, 1).contains_point(2, 0)

    def test_on_boundary_not_inside(self):
        # Strictly inside, so boundary point should be False
        assert not Circle(0, 0, 1).contains_point(1, 0)

    def test_just_inside(self):
        assert Circle(0, 0, 1).contains_point(0.999, 0)


# ── angle_of ──────────────────────────────────────────────────────


class TestAngleOf:
    def test_right(self):
        c = Circle(0, 0, 1)
        assert c.angle_of(1, 0) == pytest.approx(0.0)

    def test_up(self):
        c = Circle(0, 0, 1)
        assert c.angle_of(0, 1) == pytest.approx(math.pi / 2)

    def test_left(self):
        c = Circle(0, 0, 1)
        assert abs(c.angle_of(-1, 0)) == pytest.approx(math.pi)

    def test_down(self):
        c = Circle(0, 0, 1)
        assert c.angle_of(0, -1) == pytest.approx(-math.pi / 2)

    def test_offset_center(self):
        c = Circle(3, 4, 1)
        assert c.angle_of(4, 4) == pytest.approx(0.0)
        assert c.angle_of(3, 5) == pytest.approx(math.pi / 2)

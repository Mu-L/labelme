from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from PyQt5 import QtCore

from labelme import _shape
from labelme._shape import Shape


def _make_point_shape(x: float, y: float) -> Shape:
    shape = Shape(shape_type="point")
    shape.add_point(QtCore.QPointF(x, y))
    return shape


def _make_mask_shape(
    mask: NDArray[np.bool_], origin_x: float, origin_y: float
) -> Shape:
    shape = Shape(shape_type="mask")
    h, w = mask.shape
    shape.add_point(QtCore.QPointF(origin_x, origin_y))
    shape.add_point(QtCore.QPointF(origin_x + w, origin_y + h))
    shape.mask = mask
    return shape


def test_point_shape_contains_center() -> None:
    shape = _make_point_shape(100.0, 200.0)
    assert (
        _shape.contains_point(shape=shape, point=QtCore.QPointF(100.0, 200.0)) is True
    )


def test_point_shape_contains_within_radius() -> None:
    shape = _make_point_shape(100.0, 200.0)
    # point_size defaults to 8, so radius = 4. A point 3px away should hit.
    assert (
        _shape.contains_point(shape=shape, point=QtCore.QPointF(103.0, 200.0)) is True
    )


def test_point_shape_at_exact_boundary() -> None:
    shape = _make_point_shape(100.0, 200.0)
    # point_size defaults to 8, so radius = 4. Exactly 4px away should hit.
    assert (
        _shape.contains_point(shape=shape, point=QtCore.QPointF(104.0, 200.0)) is True
    )


def test_point_shape_outside_radius() -> None:
    shape = _make_point_shape(100.0, 200.0)
    assert (
        _shape.contains_point(shape=shape, point=QtCore.QPointF(110.0, 200.0)) is False
    )


def test_point_shape_empty_points() -> None:
    shape = Shape(shape_type="point")
    assert _shape.contains_point(shape=shape, point=QtCore.QPointF(0.0, 0.0)) is False


@pytest.mark.parametrize(
    "point, expected",
    [
        ((4, 3), True),  # inside True region
        ((4, 4), True),  # last valid row/column
        ((5, 2), False),  # one pixel past right boundary
        ((2, 5), False),  # one pixel past bottom boundary
        ((5, 5), False),  # past both boundaries
    ],
)
def test_mask_contains_point(point: tuple[int, int], expected: bool) -> None:
    mask = np.ones((5, 5), dtype=bool)
    shape = _make_mask_shape(mask, origin_x=0, origin_y=0)
    assert _shape.contains_point(shape=shape, point=QtCore.QPointF(*point)) is expected


@pytest.mark.parametrize(
    "point, expected",
    [
        (QtCore.QPointF(5.5, 0.2), 1),  # near bottom-edge midpoint (5, 0)
        (QtCore.QPointF(-0.1, 1.9), 0),  # near left-edge midpoint (0, 2)
        (QtCore.QPointF(100.0, 100.0), None),  # too far for epsilon
    ],
)
def test_oriented_rectangle_nearest_rotation_point(
    axis_aligned_oriented_rectangle: Shape,
    point: QtCore.QPointF,
    expected: int | None,
) -> None:
    assert (
        _shape.nearest_rotation_point_index(
            shape=axis_aligned_oriented_rectangle, point=point, epsilon=5.0
        )
        == expected
    )


@pytest.mark.parametrize(
    "point, expected",
    [
        (QtCore.QPointF(5.0, 2.0), True),
        (QtCore.QPointF(20.0, 20.0), False),
    ],
)
def test_oriented_rectangle_contains_point(
    axis_aligned_oriented_rectangle: Shape,
    point: QtCore.QPointF,
    expected: bool,
) -> None:
    assert (
        _shape.contains_point(shape=axis_aligned_oriented_rectangle, point=point)
        is expected
    )


def test_non_oriented_shape_nearest_rotation_point_returns_none(
    closed_axis_aligned_rectangle: Shape,
) -> None:
    assert (
        _shape.nearest_rotation_point_index(
            shape=closed_axis_aligned_rectangle,
            point=QtCore.QPointF(5.0, 0.0),
            epsilon=5.0,
        )
        is None
    )


def test_rotate_oriented_rectangle_around_origin(
    axis_aligned_oriented_rectangle: Shape,
) -> None:
    _shape.rotate(
        shape=axis_aligned_oriented_rectangle,
        center=QtCore.QPointF(0.0, 0.0),
        angle=math.pi / 2,
    )
    expected = [(0.0, 0.0), (0.0, 10.0), (-4.0, 10.0), (-4.0, 0.0)]
    for i, (x, y) in enumerate(expected):
        assert (
            axis_aligned_oriented_rectangle.points[i].x(),
            axis_aligned_oriented_rectangle.points[i].y(),
        ) == pytest.approx((x, y))


def test_rotate_non_oriented_rectangle_raises(
    closed_axis_aligned_rectangle: Shape,
) -> None:
    with pytest.raises(ValueError):
        _shape.rotate(
            shape=closed_axis_aligned_rectangle,
            center=QtCore.QPointF(0.0, 0.0),
            angle=math.pi / 2,
        )


def test_oriented_rectangle_cannot_add_point(
    axis_aligned_oriented_rectangle: Shape,
) -> None:
    assert axis_aligned_oriented_rectangle.can_add_point() is False


def test_oriented_rectangle_cannot_remove_point(
    axis_aligned_oriented_rectangle: Shape,
) -> None:
    assert axis_aligned_oriented_rectangle.can_remove_point() is False

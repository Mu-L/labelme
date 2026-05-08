from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from PyQt5 import QtCore

from labelme import _shape
from labelme.shape import Shape


def _make_point_shape(x: float, y: float) -> Shape:
    """Create a point shape with a single point at (x, y)."""
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
    """Clicking exactly on a point shape should return True."""
    shape = _make_point_shape(100.0, 200.0)
    assert (
        _shape.contains_point(shape=shape, point=QtCore.QPointF(100.0, 200.0)) is True
    )


def test_point_shape_contains_within_radius() -> None:
    """Clicking within point_size/2 of the center should return True."""
    shape = _make_point_shape(100.0, 200.0)
    # point_size defaults to 8, so radius = 4. A point 3px away should hit.
    assert (
        _shape.contains_point(shape=shape, point=QtCore.QPointF(103.0, 200.0)) is True
    )


def test_point_shape_at_exact_boundary() -> None:
    """Clicking exactly at point_size/2 distance should return True (inclusive)."""
    shape = _make_point_shape(100.0, 200.0)
    # point_size defaults to 8, so radius = 4. Exactly 4px away should hit.
    assert (
        _shape.contains_point(shape=shape, point=QtCore.QPointF(104.0, 200.0)) is True
    )


def test_point_shape_outside_radius() -> None:
    """Clicking more than point_size/2 away should return False."""
    shape = _make_point_shape(100.0, 200.0)
    # 10px away, well outside the radius of 4
    assert (
        _shape.contains_point(shape=shape, point=QtCore.QPointF(110.0, 200.0)) is False
    )


def test_point_shape_empty_points() -> None:
    """A point shape with no points should return False, not raise."""
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

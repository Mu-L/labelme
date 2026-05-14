from __future__ import annotations

import math

import numpy as np
import pytest
from PyQt5 import QtCore

from labelme import _shape
from labelme._shape import Shape


def _make_axis_aligned_oriented_rectangle() -> Shape:
    shape = Shape(shape_type="oriented_rectangle")
    for x, y in [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]:
        shape.add_point(QtCore.QPointF(x, y))
    shape.close()
    return shape


def test_rotate_oriented_rectangle_around_origin() -> None:
    shape = _make_axis_aligned_oriented_rectangle()

    _shape.rotate(
        shape=shape,
        center=QtCore.QPointF(0.0, 0.0),
        angle=math.pi / 2,
    )

    expected = [(0.0, 0.0), (0.0, 10.0), (-4.0, 10.0), (-4.0, 0.0)]
    for i, (x, y) in enumerate(expected):
        assert (shape.points[i].x(), shape.points[i].y()) == pytest.approx((x, y))


def test_rotate_non_oriented_rectangle_raises() -> None:
    shape = Shape(shape_type="rectangle")
    shape.add_point(QtCore.QPointF(0.0, 0.0))
    shape.add_point(QtCore.QPointF(10.0, 5.0))
    shape.close()

    with pytest.raises(ValueError):
        _shape.rotate(
            shape=shape,
            center=QtCore.QPointF(0.0, 0.0),
            angle=math.pi / 2,
        )


def test_nearest_vertex_index_returns_none_for_mask() -> None:
    # Mask bbox is anchored to the bitmap; exposing draggable vertices would
    # desync the rectangle from the mask.
    shape = Shape(shape_type="mask", mask=np.ones((4, 4), dtype=bool))
    shape.add_point(QtCore.QPointF(0.0, 0.0))
    shape.add_point(QtCore.QPointF(3.0, 3.0))
    shape.close()

    for corner in shape.points:
        assert (
            _shape.nearest_vertex_index(shape=shape, point=corner, epsilon=10.0) is None
        )

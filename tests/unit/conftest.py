from __future__ import annotations

import pytest
from PyQt5 import QtCore

from labelme._shape import Shape


@pytest.fixture
def axis_aligned_oriented_rectangle() -> Shape:
    shape = Shape(shape_type="oriented_rectangle")
    for x, y in [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]:
        shape.add_point(QtCore.QPointF(x, y))
    shape.close()
    return shape


@pytest.fixture
def closed_axis_aligned_rectangle() -> Shape:
    shape = Shape(shape_type="rectangle")
    shape.add_point(QtCore.QPointF(0.0, 0.0))
    shape.add_point(QtCore.QPointF(10.0, 5.0))
    shape.close()
    return shape

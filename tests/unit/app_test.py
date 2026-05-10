from __future__ import annotations

import math

import pytest
from PyQt5 import QtCore

from labelme._shape import Shape
from labelme.app import _shape_to_xyxy_bbox


def test_shape_to_xyxy_bbox_circle() -> None:
    shape = Shape(shape_type="circle")
    shape.add_point(QtCore.QPointF(50, 40))
    shape.add_point(QtCore.QPointF(53, 44))

    bbox = _shape_to_xyxy_bbox(shape)

    radius = math.sqrt((53 - 50) ** 2 + (44 - 40) ** 2)
    assert bbox.tolist() == pytest.approx(
        [50 - radius, 40 - radius, 50 + radius, 40 + radius]
    )


def test_shape_to_xyxy_bbox_raises_on_unsupported_shape_type() -> None:
    shape = Shape(shape_type="point")
    shape.add_point(QtCore.QPointF(1, 2))

    with pytest.raises(ValueError, match="Unsupported shape_type"):
        _shape_to_xyxy_bbox(shape)

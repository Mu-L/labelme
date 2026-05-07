from __future__ import annotations

from collections.abc import Iterable

from PyQt5 import QtCore
from PyQt5 import QtGui

from . import utils
from .shape import Shape


def _argmin(values: Iterable[float]) -> tuple[int, float] | None:
    return min(enumerate(values), key=lambda item: item[1], default=None)


def nearest_vertex_index(
    *,
    shape: Shape,
    point: QtCore.QPointF,
    epsilon: float,
) -> int | None:
    scaled_point = point * shape.scale
    nearest = _argmin(
        utils.distance(p * shape.scale - scaled_point) for p in shape.points
    )
    if nearest is None or nearest[1] > epsilon:
        return None
    return nearest[0]


def nearest_edge_index(
    *,
    shape: Shape,
    point: QtCore.QPointF,
    epsilon: float,
) -> int | None:
    scaled_point = point * shape.scale
    scaled_points = [p * shape.scale for p in shape.points]
    nearest = _argmin(
        utils.distance_to_line(scaled_point, (scaled_points[i - 1], scaled_points[i]))
        for i in range(len(shape.points))
    )
    if nearest is None or nearest[1] > epsilon:
        return None
    return nearest[0]


def contains_point(*, shape: Shape, point: QtCore.QPointF) -> bool:
    if shape.shape_type in ("line", "linestrip", "points"):
        return False
    if shape.shape_type == "point":
        if not shape.points:
            return False
        return utils.distance(point - shape.points[0]) <= shape.point_size / 2
    if shape.mask is not None:
        raw_y = int(round(point.y() - shape.points[0].y()))
        raw_x = int(round(point.x() - shape.points[0].x()))
        if (
            raw_y < 0
            or raw_y >= shape.mask.shape[0]
            or raw_x < 0
            or raw_x >= shape.mask.shape[1]
        ):
            return False
        return bool(shape.mask[raw_y, raw_x])
    return _build_path(shape=shape).contains(point)


def bounds(*, shape: Shape) -> QtCore.QRectF:
    return _build_path(shape=shape).boundingRect()


def _build_path_rectangle(*, points: list[QtCore.QPointF]) -> QtGui.QPainterPath:
    out = QtGui.QPainterPath()
    if len(points) == 2:
        out.addRect(QtCore.QRectF(points[0], points[1]))
    return out


def _build_path_circle(*, points: list[QtCore.QPointF]) -> QtGui.QPainterPath:
    out = QtGui.QPainterPath()
    if len(points) == 2:
        radius = utils.distance(points[0] - points[1])
        out.addEllipse(points[0], radius, radius)
    return out


def _build_path_polyline(*, points: list[QtCore.QPointF]) -> QtGui.QPainterPath:
    out = QtGui.QPainterPath()
    if not points:
        return out
    out.moveTo(points[0])
    for vertex in points[1:]:
        out.lineTo(vertex)
    return out


def _build_path(*, shape: Shape) -> QtGui.QPainterPath:
    build_path_fn = {
        "rectangle": _build_path_rectangle,
        "mask": _build_path_rectangle,
        "circle": _build_path_circle,
    }.get(shape.shape_type, _build_path_polyline)
    return build_path_fn(points=shape.points)

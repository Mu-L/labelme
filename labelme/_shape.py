from __future__ import annotations

from collections.abc import Iterable

from PyQt5 import QtCore

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

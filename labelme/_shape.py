from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import skimage.measure
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


def paint(*, shape: Shape, painter: QtGui.QPainter) -> None:
    if shape.mask is None and not shape.points:
        return

    color = shape.select_line_color if shape.selected else shape.line_color
    pen = QtGui.QPen(color)
    pen.setWidth(shape.PEN_WIDTH)
    painter.setPen(pen)

    if shape.shape_type == "mask" and shape.mask is not None:
        _paint_shape_mask(painter=painter, shape=shape)

    if shape.points:
        _paint_shape_points(painter=painter, shape=shape)


def _paint_shape_mask(*, painter: QtGui.QPainter, shape: Shape) -> None:
    assert shape.mask is not None
    fill = shape.select_fill_color if shape.selected else shape.fill_color
    image_to_draw = np.zeros(shape.mask.shape + (4,), dtype=np.uint8)
    image_to_draw[shape.mask] = fill.getRgb()
    qimage = QtGui.QImage.fromData(utils.img_arr_to_data(image_to_draw))
    origin = shape.points[0]
    target_top_left = origin * shape.scale
    target_rect = QtCore.QRectF(
        target_top_left.x(),
        target_top_left.y(),
        qimage.width() * shape.scale,
        qimage.height() * shape.scale,
    )
    painter.drawImage(target_rect, qimage)

    path = QtGui.QPainterPath()
    _build_shape_mask_contour_path(
        path=path, mask=shape.mask, origin=origin, scale=shape.scale
    )
    painter.drawPath(path)


def _paint_shape_points(*, painter: QtGui.QPainter, shape: Shape) -> None:
    path_line = QtGui.QPainterPath()
    path_vertices = QtGui.QPainterPath()
    path_negative_vertices = QtGui.QPainterPath()

    _build_shape_points_paths(
        shape=shape,
        path_line=path_line,
        path_vertices=path_vertices,
        path_negative_vertices=path_negative_vertices,
    )

    painter.drawPath(path_line)
    if path_vertices.length() > 0:
        vertex_fill = (
            shape.vertex_fill_color
            if shape.highlight is None
            else shape.hvertex_fill_color
        )
        painter.drawPath(path_vertices)
        painter.fillPath(path_vertices, vertex_fill)
    if shape.fill and shape.shape_type not in ["line", "linestrip", "points", "mask"]:
        fill = shape.select_fill_color if shape.selected else shape.fill_color
        painter.fillPath(path_line, fill)

    neg_color = QtGui.QColor(255, 0, 0, 255)
    neg_pen = QtGui.QPen(neg_color)
    neg_pen.setWidth(shape.PEN_WIDTH)
    painter.setPen(neg_pen)
    painter.drawPath(path_negative_vertices)
    painter.fillPath(path_negative_vertices, neg_color)


def _build_shape_mask_contour_path(
    *,
    path: QtGui.QPainterPath,
    mask: npt.NDArray[np.bool_],
    origin: QtCore.QPointF,
    scale: float,
) -> None:
    contours = skimage.measure.find_contours(np.pad(mask, pad_width=1))
    for contour in contours:
        contour = contour + [origin.y(), origin.x()]
        path.moveTo(QtCore.QPointF(contour[0, 1], contour[0, 0]) * scale)
        for point in contour[1:]:
            path.lineTo(QtCore.QPointF(point[1], point[0]) * scale)


def _build_shape_point_path(
    *, path: QtGui.QPainterPath, shape: Shape, vertex_index: int
) -> None:
    highlight = shape.highlight
    if highlight is not None and highlight.index == vertex_index:
        size = shape.point_size * highlight.size_factor
        point_type = highlight.point_type
    else:
        size = shape.point_size
        point_type = shape.point_type

    pos = shape.points[vertex_index] * shape.scale

    half = size / 2.0
    if point_type == "square":
        path.addRect(pos.x() - half, pos.y() - half, size, size)
    elif point_type == "round":
        path.addEllipse(pos, half, half)
    else:
        raise ValueError(f"Unsupported vertex shape: {point_type}")


def _build_shape_points_paths(
    *,
    shape: Shape,
    path_line: QtGui.QPainterPath,
    path_vertices: QtGui.QPainterPath,
    path_negative_vertices: QtGui.QPainterPath,
) -> None:
    if shape.shape_type in ["rectangle", "mask"]:
        assert len(shape.points) in [1, 2]
        if len(shape.points) == 2:
            path_line.addRect(
                QtCore.QRectF(
                    shape.points[0] * shape.scale,
                    shape.points[1] * shape.scale,
                )
            )
        if shape.shape_type == "rectangle":
            for i in range(len(shape.points)):
                _build_shape_point_path(path=path_vertices, shape=shape, vertex_index=i)
    elif shape.shape_type == "circle":
        assert len(shape.points) in [1, 2]
        if len(shape.points) == 2:
            radius = utils.distance((shape.points[0] - shape.points[1]) * shape.scale)
            path_line.addEllipse(shape.points[0] * shape.scale, radius, radius)
        for i in range(len(shape.points)):
            _build_shape_point_path(path=path_vertices, shape=shape, vertex_index=i)
    elif shape.shape_type == "linestrip":
        path_line.moveTo(shape.points[0] * shape.scale)
        for i, p in enumerate(shape.points):
            path_line.lineTo(p * shape.scale)
            _build_shape_point_path(path=path_vertices, shape=shape, vertex_index=i)
    elif shape.shape_type == "points":
        assert len(shape.points) == len(shape.point_labels)
        for i, point_label in enumerate(shape.point_labels):
            path = path_vertices if point_label == 1 else path_negative_vertices
            _build_shape_point_path(path=path, shape=shape, vertex_index=i)
    else:
        path_line.moveTo(shape.points[0] * shape.scale)
        for i, p in enumerate(shape.points):
            path_line.lineTo(p * shape.scale)
            _build_shape_point_path(path=path_vertices, shape=shape, vertex_index=i)
        if shape.is_closed():
            path_line.lineTo(shape.points[0] * shape.scale)

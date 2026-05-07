from __future__ import annotations

import copy
import dataclasses
from typing import Any
from typing import Final
from typing import Literal

import numpy as np
import numpy.typing as npt
import skimage.measure
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui

import labelme.utils

_P_SQUARE: Final[int] = 0
_P_ROUND: Final[int] = 1

POLYLINE_SHAPE_TYPES: Final[tuple[str, ...]] = ("polygon", "linestrip")


@dataclasses.dataclass(frozen=True)
class _VertexHighlight:
    index: int
    mode: Literal["move", "near"]

    @property
    def size_factor(self) -> float:
        return {
            "move": 1.5,
            "near": 4.0,
        }[self.mode]

    @property
    def point_type(self) -> int:
        return {
            "move": _P_SQUARE,
            "near": _P_ROUND,
        }[self.mode]


class Shape:
    PEN_WIDTH: Final[int] = 2

    line_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 128)
    fill_color: QtGui.QColor = QtGui.QColor(0, 0, 0, 64)
    vertex_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 255)
    select_line_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)
    select_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 64)
    hvertex_fill_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)

    point_type: int = _P_ROUND
    point_size: int = 8
    scale: float = 1.0

    def __init__(
        self,
        label: str | None = None,
        line_color: QtGui.QColor | None = None,
        shape_type: str | None = None,
        flags: dict[str, bool] | None = None,
        group_id: int | None = None,
        description: str | None = None,
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        self.label = label
        self.group_id = group_id
        self.points: list[QtCore.QPointF] = []
        self.point_labels: list[int] = []
        self.shape_type = shape_type
        self.fill = False
        self.selected = False
        self.visible = True
        self.flags = flags
        self.description = description
        self.other_data: dict[str, Any] = {}
        self.mask = mask
        self._closed = False
        self.highlight: _VertexHighlight | None = None

        if line_color is not None:
            # Per-instance line color override (used for the pending line).
            self.line_color = line_color

    @property
    def shape_type(self) -> str:
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value: str | None) -> None:
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "points",
            "mask",
        ]:
            raise ValueError(f"Unexpected shape_type: {value}")
        self._shape_type = value

    def close(self) -> None:
        self._closed = True

    def add_point(
        self, point: QtCore.QPointF, label: int = 1, *, autoclose: bool = False
    ) -> None:
        if autoclose and self.points and self.points[0] == point:
            self.close()
            return
        self.points.append(point)
        self.point_labels.append(label)

    def can_add_point(self) -> bool:
        return self.shape_type in POLYLINE_SHAPE_TYPES

    def pop_point(self) -> QtCore.QPointF | None:
        if not self.points:
            return None
        if self.point_labels:
            self.point_labels.pop()
        return self.points.pop()

    def insert_point(self, i: int, point: QtCore.QPointF, label: int = 1) -> None:
        self.points.insert(i, point)
        self.point_labels.insert(i, label)

    def can_remove_point(self) -> bool:
        if not self.can_add_point():
            return False

        if self.shape_type == "polygon" and len(self.points) <= 3:
            return False

        if self.shape_type == "linestrip" and len(self.points) <= 2:
            return False

        return True

    def remove_point(self, i: int) -> None:
        if not self.can_remove_point():
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        self.points.pop(i)
        self.point_labels.pop(i)

    def is_closed(self) -> bool:
        return self._closed

    def open(self) -> None:
        self._closed = False

    def paint(self, painter: QtGui.QPainter) -> None:
        _paint_shape(painter=painter, shape=self)

    def move_vertex(self, i: int, pos: QtCore.QPointF) -> None:
        self.points[i] = pos

    def translate(self, offset: QtCore.QPointF) -> None:
        for i, point in enumerate(self.points):
            self.points[i] = point + offset

    def highlight_vertex(self, index: int, mode: Literal["move", "near"]) -> None:
        self.highlight = _VertexHighlight(index=index, mode=mode)

    def clear_highlight(self) -> None:
        self.highlight = None

    def copy(self) -> Shape:
        return copy.deepcopy(self)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, key: int) -> QtCore.QPointF:
        return self.points[key]

    def __setitem__(self, key: int, value: QtCore.QPointF) -> None:
        self.points[key] = value


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
    if point_type == _P_SQUARE:
        path.addRect(pos.x() - half, pos.y() - half, size, size)
    elif point_type == _P_ROUND:
        path.addEllipse(pos, half, half)
    else:
        raise ValueError(f"Unsupported vertex shape: {point_type}")


def _paint_shape(*, painter: QtGui.QPainter, shape: Shape) -> None:
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
    qimage = QtGui.QImage.fromData(labelme.utils.img_arr_to_data(image_to_draw))
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
            radius = labelme.utils.distance(
                (shape.points[0] - shape.points[1]) * shape.scale
            )
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

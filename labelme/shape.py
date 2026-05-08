from __future__ import annotations

import copy
import dataclasses
import typing
from typing import Any
from typing import Final
from typing import Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from PyQt5 import QtCore
from PyQt5 import QtGui

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
    def point_type(self) -> Literal["square", "round"]:
        match self.mode:
            case "move":
                return "square"
            case "near":
                return "round"
            case _:
                typing.assert_never(self.mode)


class Shape:
    PEN_WIDTH: Final[int] = 2

    line_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 128)
    fill_color: QtGui.QColor = QtGui.QColor(0, 0, 0, 64)
    vertex_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 255)
    select_line_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)
    select_fill_color: QtGui.QColor = QtGui.QColor(0, 255, 0, 64)
    hvertex_fill_color: QtGui.QColor = QtGui.QColor(255, 255, 255, 255)

    point_type: Literal["square", "round"] = "round"
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

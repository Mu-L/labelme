from __future__ import annotations

import math
from typing import Final

import pytest
from PyQt5 import QtGui
from PyQt5.QtCore import QPointF
from pytestqt.qtbot import QtBot

from labelme._shape import Shape
from labelme.widgets.canvas import Canvas
from labelme.widgets.canvas import _compute_intersection_edges_image
from labelme.widgets.canvas import _normalize_bbox_points
from labelme.widgets.canvas import _opposite_corner_in_parallelogram
from labelme.widgets.canvas import _project_oriented_rectangle_corners

_WIDTH: Final = 100
_HEIGHT: Final = 50


@pytest.fixture()
def canvas(qtbot: QtBot) -> Canvas:
    canvas = Canvas()
    canvas.pixmap = QtGui.QPixmap(_WIDTH, _HEIGHT)
    qtbot.addWidget(canvas)
    return canvas


@pytest.mark.gui
@pytest.mark.parametrize(
    ("point", "is_outside"),
    [
        (QPointF(_WIDTH / 2, _HEIGHT / 2), False),
        (QPointF(0, 0), False),
        (QPointF(_WIDTH, _HEIGHT), False),
        (QPointF(_WIDTH, _HEIGHT / 2), False),
        (QPointF(_WIDTH / 2, _HEIGHT), False),
        (QPointF(_WIDTH + 0.1, _HEIGHT / 2), True),
        (QPointF(_WIDTH / 2, _HEIGHT + 0.1), True),
        (QPointF(-0.1, _HEIGHT / 2), True),
        (QPointF(_WIDTH / 2, -0.1), True),
    ],
)
def test_is_out_of_pixmap(canvas: Canvas, point: QPointF, is_outside: bool) -> None:
    assert canvas.is_out_of_pixmap(point) is is_outside


@pytest.mark.gui
@pytest.mark.parametrize(
    ("p1", "p2", "pt_intersection"),
    [
        (
            pt_center := QPointF(_WIDTH / 2, _HEIGHT / 2),
            QPointF(_WIDTH + 50, _HEIGHT / 2),  # to the right
            QPointF(_WIDTH, _HEIGHT / 2),  # right edge
        ),
        (
            pt_center,
            QPointF(_WIDTH / 2, -10),  # to the top
            QPointF(_WIDTH / 2, 0),  # top edge
        ),
        (
            pt_center,
            QPointF(-10, _HEIGHT / 2),  # to the left
            QPointF(0, _HEIGHT / 2),  # left edge
        ),
        (
            pt_center,
            QPointF(_WIDTH / 2, _HEIGHT + 30),  # to the bottom
            QPointF(_WIDTH / 2, _HEIGHT),  # bottom edge
        ),
        (
            QPointF(0, _HEIGHT / 2),  # on left edge
            QPointF(-5, _HEIGHT / 2),  # further left
            QPointF(0, _HEIGHT / 2),  # stays on left edge
        ),
        (
            QPointF(_WIDTH / 2, 0),  # on top edge
            QPointF(_WIDTH / 2, -5),  # further up
            QPointF(_WIDTH / 2, 0),  # stays on top edge
        ),
        (
            QPointF(0, _HEIGHT / 2),  # on left edge
            QPointF(-5, _HEIGHT / 2 + 10),  # further left and down
            QPointF(0, _HEIGHT / 2 + 10),  # slides down along left edge
        ),
        (
            QPointF(0, 0),  # top-left corner
            QPointF(-5, -5),  # diagonally out
            QPointF(0, 0),  # stays at corner
        ),
        (
            QPointF(_WIDTH, _HEIGHT),  # bottom-right corner
            QPointF(_WIDTH + 5, _HEIGHT + 5),  # diagonally out
            QPointF(_WIDTH, _HEIGHT),  # stays at corner
        ),
    ],
)
def test_intersectionPoint(
    canvas: Canvas, p1: QPointF, p2: QPointF, pt_intersection: QPointF
) -> None:
    assert (
        _compute_intersection_edges_image(p1, p2, image_size=canvas.pixmap.size())
        == pt_intersection
    )


@pytest.mark.parametrize(
    ("p1", "p2"),
    [
        (QPointF(10, 20), QPointF(30, 40)),  # top-left -> bottom-right
        (QPointF(30, 40), QPointF(10, 20)),  # bottom-right -> top-left
        (QPointF(30, 20), QPointF(10, 40)),  # top-right -> bottom-left
        (QPointF(10, 40), QPointF(30, 20)),  # bottom-left -> top-right
    ],
)
def test_normalize_bbox_points(p1: QPointF, p2: QPointF) -> None:
    assert _normalize_bbox_points(bbox_points=[p1, p2]) == [
        QPointF(10, 20),
        QPointF(30, 40),
    ]


def test_normalize_bbox_points_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="Expected 2 points"):
        _normalize_bbox_points(bbox_points=[QPointF(0, 0)])


def test_opposite_corner_in_parallelogram() -> None:
    # Square corners (0,0), (10,0), (10,10), (0,10).
    # Opposite of (0,0) given neighbors (10,0) and (0,10) is (10,10).
    assert _opposite_corner_in_parallelogram(
        opposite_to=QPointF(0, 0),
        neighbor1=QPointF(10, 0),
        neighbor2=QPointF(0, 10),
    ) == QPointF(10, 10)


def test_project_oriented_rectangle_corners_axis_aligned() -> None:
    # Rectangle with anchor at origin, edge along +x at y=0; moving cursor at (10, 4).
    # Expected: perp = (0, 4), para = (10, 0).
    perp, para = _project_oriented_rectangle_corners(
        anchor=QPointF(0, 0),
        edge_axis=QPointF(10, 0),
        moving=QPointF(10, 4),
    )
    assert (perp.x(), perp.y()) == pytest.approx((0.0, 4.0))
    assert (para.x(), para.y()) == pytest.approx((10.0, 0.0))


def _make_oriented_rectangle(corners: list[tuple[float, float]]) -> Shape:
    shape = Shape(shape_type="oriented_rectangle")
    for x, y in corners:
        shape.add_point(QPointF(x, y))
    shape.close()
    return shape


@pytest.mark.gui
def test_bounded_move_oriented_rectangle_vertex_in_bounds_preserves_parallelogram(
    canvas: Canvas,
) -> None:
    # Axis-aligned rect; drag the opposite corner to a fully-inside pos.
    # Anchor stays put; adjacents project to preserve right angles.
    shape = _make_oriented_rectangle(corners=[(10, 10), (40, 10), (40, 30), (10, 30)])

    canvas._bounded_move_oriented_rectangle_vertex(
        shape=shape, vertex_index=2, pos=QPointF(50, 25)
    )

    expected = [(10, 10), (50, 10), (50, 25), (10, 25)]
    for i, (x, y) in enumerate(expected):
        assert (shape.points[i].x(), shape.points[i].y()) == pytest.approx((x, y))


@pytest.mark.gui
def test_bounded_move_oriented_rectangle_vertex_clips_when_moving_outside(
    canvas: Canvas,
) -> None:
    # 45° rotated square; dragging vertex 2 above the pixmap top keeps both
    # adjacent corners inside the pixmap, exercising the moving-out branch.
    shape = _make_oriented_rectangle(corners=[(50, 30), (60, 20), (50, 10), (40, 20)])

    canvas._bounded_move_oriented_rectangle_vertex(
        shape=shape, vertex_index=2, pos=QPointF(50, -10)
    )

    expected = [(50, 30), (65, 15), (50, 0), (35, 15)]
    for i, (x, y) in enumerate(expected):
        assert (shape.points[i].x(), shape.points[i].y()) == pytest.approx((x, y))


@pytest.mark.gui
def test_bounded_move_oriented_rectangle_vertex_clips_when_perpendicular_corner_outside(
    canvas: Canvas,
) -> None:
    # Tilted parallelogram chosen so that dragging vertex 2 to (95, 5) keeps
    # the moving corner inside the pixmap but pushes the perpendicular
    # adjacent corner above y=0, isolating the perpendicular-clip branch.
    shape = _make_oriented_rectangle(corners=[(50, 30), (60, 35), (65, 25), (55, 20)])

    canvas._bounded_move_oriented_rectangle_vertex(
        shape=shape, vertex_index=2, pos=QPointF(95, 5)
    )

    expected = [(50, 30), (76, 43), (91, 13), (65, 0)]
    for i, (x, y) in enumerate(expected):
        assert (shape.points[i].x(), shape.points[i].y()) == pytest.approx((x, y))


@pytest.mark.gui
def test_bounded_move_oriented_rectangle_vertex_clips_when_parallel_corner_outside(
    canvas: Canvas,
) -> None:
    # Same shape; dragging vertex 2 to (95, 45) keeps the moving and
    # perpendicular adjacent inside but pushes the parallel adjacent corner
    # below y=50, isolating the parallel-clip branch.
    shape = _make_oriented_rectangle(corners=[(50, 30), (60, 35), (65, 25), (55, 20)])

    canvas._bounded_move_oriented_rectangle_vertex(
        shape=shape, vertex_index=2, pos=QPointF(95, 45)
    )

    expected = [(50, 30), (90, 50), (93, 44), (53, 24)]
    for i, (x, y) in enumerate(expected):
        assert (shape.points[i].x(), shape.points[i].y()) == pytest.approx((x, y))


@pytest.mark.gui
def test_drag_hovered_rotation_point_rotates_around_center(canvas: Canvas) -> None:
    # Centred 40x30 rect; corners are >epsilon away from edge midpoints so
    # hover resolves to the rotation handle, not a vertex handle.
    shape = _make_oriented_rectangle(corners=[(30, 10), (70, 10), (70, 40), (30, 40)])
    canvas.load_shapes(shapes=[shape])

    canvas._refresh_hover_state(pos=QPointF(50, 10))
    assert canvas._hovered_rotation == 1
    canvas._capture_rotation_anchors()

    # Drag the top-edge midpoint (50, 10) -> (35, 25); with flip_y this is a
    # 90° screen-CCW rotation around the centre (50, 25). After rotation the
    # original top edge becomes the new left edge.
    canvas._drag_hovered_rotation_point(pos=QPointF(35, 25))

    expected = [(35, 45), (35, 5), (65, 5), (65, 45)]
    for i, (x, y) in enumerate(expected):
        assert (
            canvas.shapes[0].points[i].x(),
            canvas.shapes[0].points[i].y(),
        ) == pytest.approx((x, y))


@pytest.mark.gui
def test_drag_hovered_rotation_point_does_not_drift_on_repeated_drags(
    canvas: Canvas,
) -> None:
    # Rotate a shape through many small steps, then back to the start.
    # Without drift-free rotation, accumulated floating-point error from
    # composed rotation matrices would leave residual offset on the corners.
    original: list[tuple[float, float]] = [
        (30, 10),
        (70, 10),
        (70, 40),
        (30, 40),
    ]
    shape = _make_oriented_rectangle(corners=original)
    canvas.load_shapes(shapes=[shape])

    canvas._refresh_hover_state(pos=QPointF(50, 10))
    assert canvas._hovered_rotation == 1
    canvas._capture_rotation_anchors()

    center_x, center_y = 50.0, 25.0
    radius = 15.0
    steps = 200
    for step in range(1, steps + 1):
        theta = -math.pi / 2 + 2 * math.pi * step / steps
        pos = QPointF(
            center_x + radius * math.cos(theta), center_y + radius * math.sin(theta)
        )
        canvas._drag_hovered_rotation_point(pos=pos)

    for i, (x, y) in enumerate(original):
        assert canvas.shapes[0].points[i].x() == pytest.approx(x)
        assert canvas.shapes[0].points[i].y() == pytest.approx(y)


@pytest.mark.gui
def test_shape_visibility_survives_backup_and_restore(canvas: Canvas) -> None:
    shape = Shape(label="a", shape_type="rectangle")
    shape.add_point(QPointF(0, 0))
    shape.add_point(QPointF(10, 10))
    shape.close()
    canvas.load_shapes([shape])

    canvas.set_shape_visible(canvas.shapes[0], False)
    canvas.backup_shapes()
    canvas.load_shapes([shape.copy()])
    assert canvas.shapes[0].visible is False

    canvas.restore_last_shape()
    assert canvas.shapes[0].visible is False

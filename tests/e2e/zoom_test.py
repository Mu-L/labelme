from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from PyQt5 import QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import Qt
from pytestqt.qtbot import QtBot

from labelme.app import MainWindow
from labelme.app import _ZoomMode

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata

_TEST_FILE_NAME: Final[str] = "annotated/2011_000003.json"


@pytest.fixture()
def _win(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> MainWindow:
    win = main_win(
        file_or_dir=str(data_path / _TEST_FILE_NAME),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    return win


@pytest.mark.gui
def test_zoom_fit_window(
    qtbot: QtBot,
    _win: MainWindow,
    pause: bool,
) -> None:
    _win.set_fit_window_mode(True)

    zoom_value = _win._canvas_widgets.zoom_widget.value()
    assert zoom_value != 100
    assert zoom_value > 0
    assert _win._zoom_mode == _ZoomMode.FIT_WINDOW

    close_or_pause(qtbot=qtbot, widget=_win, pause=pause)


@pytest.mark.gui
def test_zoom_fit_width(
    qtbot: QtBot,
    _win: MainWindow,
    pause: bool,
) -> None:
    _win.set_fit_window_mode(True)
    _win.set_fit_width_mode(True)

    fit_width_zoom = _win._canvas_widgets.zoom_widget.value()
    assert fit_width_zoom > 0
    assert _win._zoom_mode == _ZoomMode.FIT_WIDTH

    close_or_pause(qtbot=qtbot, widget=_win, pause=pause)


@pytest.mark.gui
def test_zoom_to_original(
    qtbot: QtBot,
    _win: MainWindow,
    pause: bool,
) -> None:
    _win.set_fit_window_mode(True)
    assert _win._canvas_widgets.zoom_widget.value() != 100

    _win._set_zoom_to_original()

    assert _win._canvas_widgets.zoom_widget.value() == 100
    assert _win._zoom_mode == _ZoomMode.MANUAL_ZOOM

    close_or_pause(qtbot=qtbot, widget=_win, pause=pause)


def _make_wheel_event(
    pos: QPointF,
    angle_delta: QPoint,
    modifiers: Qt.KeyboardModifiers,
) -> QtGui.QWheelEvent:
    # PyQt5's QWheelEvent has overlapping qt4 and modern overloads; the
    # all-positional 8-arg form is the only one that disambiguates without
    # forcing a particular `KeyboardModifier`/`KeyboardModifiers` type.
    return QtGui.QWheelEvent(
        pos,
        pos,
        QPoint(0, 0),
        angle_delta,
        Qt.NoButton,
        modifiers,
        Qt.NoScrollPhase,
        False,
    )


@pytest.mark.gui
@pytest.mark.parametrize(
    ("modifiers", "angle_delta", "signal_attr", "expected_orientation"),
    [
        pytest.param(
            Qt.ControlModifier, QPoint(0, 120), "zoom_request", None, id="ctrl_zoom"
        ),
        pytest.param(
            Qt.NoModifier,
            QPoint(0, 120),
            "scroll_request",
            Qt.Vertical,
            id="plain_scroll",
        ),
        pytest.param(
            Qt.ShiftModifier,
            QPoint(0, 120),
            "scroll_request",
            Qt.Horizontal,
            id="shift_horizontal_scroll",
        ),
    ],
)
def test_canvas_wheel_event_dispatches_signal(
    qtbot: QtBot,
    _win: MainWindow,
    pause: bool,
    modifiers: Qt.KeyboardModifiers,
    angle_delta: QPoint,
    signal_attr: str,
    expected_orientation: Qt.Orientation | None,
) -> None:
    canvas = _win._canvas_widgets.canvas
    captured: list[tuple[object, ...]] = []
    signal = getattr(canvas, signal_attr)
    signal.connect(lambda *args: captured.append(args))

    canvas.wheelEvent(
        _make_wheel_event(
            pos=QPointF(canvas.width() / 2, canvas.height() / 2),
            angle_delta=angle_delta,
            modifiers=modifiers,
        )
    )

    assert captured, f"{signal_attr} was not emitted"
    if expected_orientation is not None:
        # The non-zero-delta emission must be on the expected axis: the plain
        # case also emits an empty horizontal step (delta.x() == 0) before the
        # vertical one, so filter to non-zero deltas before asserting.
        non_zero = [args for args in captured if args[0] != 0]
        assert non_zero, f"{signal_attr} emitted no non-zero deltas"
        assert all(args[1] == expected_orientation for args in non_zero)

    close_or_pause(qtbot=qtbot, widget=_win, pause=pause)

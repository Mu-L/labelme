from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5 import QtWidgets
from pytestqt.qtbot import QtBot

from labelme.app import MainWindow

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata


@pytest.fixture()
def paths(data_path: Path, tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        annotated_dir=data_path / "annotated",
        next_image=data_path / "raw" / "2011_000006.jpg",
        save_path=tmp_path / "saved.json",
        new_output_dir=tmp_path / "alt_output",
    )


@pytest.fixture()
def loaded_win(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> MainWindow:
    # All four scenarios need an image already loaded so the action methods
    # have a populated `_image_path` / `image_list` to work with.
    win = main_win(file_or_dir=str(data_path / "raw" / "2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    return win


@pytest.mark.gui
@pytest.mark.parametrize(
    ("dialog_method", "dialog_return", "trigger", "verify"),
    [
        pytest.param(
            "getOpenFileName",
            lambda paths: (str(paths.next_image), ""),
            lambda win: win._open_file_with_dialog(),
            lambda win, paths: Path(win._image_path).resolve()
            == paths.next_image.resolve(),
            id="open_file",
        ),
        pytest.param(
            "getExistingDirectory",
            lambda paths: str(paths.annotated_dir),
            lambda win: win._open_dir_with_dialog(),
            lambda win, paths: win._docks.file_list.count() > 0
            and Path(win._image_path).parent.resolve() == paths.annotated_dir.resolve(),
            id="open_dir",
        ),
        pytest.param(
            "getSaveFileName",
            lambda paths: (str(paths.save_path), ""),
            lambda win: win._save_label_file(save_as=True),
            lambda win, paths: paths.save_path.exists(),
            id="save_as",
        ),
        pytest.param(
            "getExistingDirectory",
            lambda paths: str(paths.new_output_dir),
            lambda win: win.prompt_output_dir(),
            lambda win, paths: win._output_dir == paths.new_output_dir,
            id="change_output_dir",
        ),
    ],
)
def test_action_via_qfile_dialog(
    qtbot: QtBot,
    loaded_win: MainWindow,
    paths: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    pause: bool,
    dialog_method: str,
    dialog_return: Callable[[SimpleNamespace], object],
    trigger: Callable[[MainWindow], None],
    verify: Callable[[MainWindow, SimpleNamespace], bool],
) -> None:
    if (output_dir := getattr(paths, "new_output_dir", None)) is not None:
        output_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        dialog_method,
        lambda *args, **kwargs: dialog_return(paths),
    )

    trigger(loaded_win)
    qtbot.wait(100)

    assert verify(loaded_win, paths)

    close_or_pause(qtbot=qtbot, widget=loaded_win, pause=pause)

import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from swarm_rescue.simulation.reporting.bombs_io import (
    COMPARISON_HEADER_HEIGHT,
    WALL_DESTRUCTION_LEGEND_HEIGHT,
    _COMPARISON_COLOR_WHITE,
    _WALL_COLOR_BLAST_OVERLAY,
    _WALL_COLOR_BOMB,
    _WALL_COLOR_DESTROYED_BOUNDARY,
    _WALL_COLOR_DESTROYED_INTERIOR,
    _WALL_COLOR_INTACT_BOUNDARY,
    _WALL_COLOR_INTACT_INTERIOR,
    save_wall_destruction_image,
    wall_destruction_image_output_path,
)
from swarm_rescue.simulation.reporting.evaluation import EvalConfig


def _wall_grid() -> np.ndarray:
    """40x40 grid: boundary ring + interior bar (world coords via row=20-y)."""
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[0, :] = 1
    truth[-1, :] = 1
    truth[:, 0] = 1
    truth[:, -1] = 1
    truth[10:30, 20:22] = 1
    return truth


def _pixel(image: np.ndarray, row: int, col: int) -> tuple:
    return tuple(
        int(v)
        for v in image[COMPARISON_HEADER_HEIGHT + row, col]
    )


def _render(
    truth: np.ndarray,
    bombs: list,
    *,
    radius: float = 5.0,
    boundary_weight: float = 0.5,
    tmp: Path,
) -> np.ndarray:
    out = tmp / "wall.png"
    save_wall_destruction_image(
        out, truth, bombs,
        blast_radius=radius,
        boundary_wall_weight=boundary_weight,
        score_wall_percent=30.0,
    )
    image = cv2.imread(str(out))
    assert image is not None
    return image


def test_wall_destruction_image_output_path_naming():
    team_info = SimpleNamespace(team_number_str_padded="007", team_number_str="7")
    eval_config = EvalConfig(map_name="Map04", zones_config=())

    path = wall_destruction_image_output_path(
        "/tmp/results", team_info, eval_config, 3
    )

    assert path is not None
    assert Path(path).name == "team007_Map04_none_rd3_wall_destruction.png"


def test_wall_destruction_image_output_path_none_without_result_path():
    team_info = SimpleNamespace(team_number_str_padded="007", team_number_str="7")
    eval_config = EvalConfig(map_name="Map04", zones_config=())

    assert wall_destruction_image_output_path(
        None, team_info, eval_config, 1
    ) is None


def test_save_wall_destruction_image_is_single_panel():
    truth = _wall_grid()
    with tempfile.TemporaryDirectory() as tmp:
        image = _render(truth, [(0.0, 0.0)], tmp=Path(tmp))

    h, w = truth.shape
    assert image.shape == (
        COMPARISON_HEADER_HEIGHT + h + WALL_DESTRUCTION_LEGEND_HEIGHT,
        w,
        3,
    )


def test_walls_coloured_by_state_and_bombs_marked():
    truth = _wall_grid()
    with tempfile.TemporaryDirectory() as tmp:
        image = _render(truth, [(0.0, 0.0)], radius=5.0, tmp=Path(tmp))

    # (25,20) interior wall, within blast radius 5 but outside the marker.
    assert _pixel(image, 25, 20) == _WALL_COLOR_DESTROYED_INTERIOR
    # (12,20) interior wall too far from the bomb -> intact.
    assert _pixel(image, 12, 20) == _WALL_COLOR_INTACT_INTERIOR
    # (0,0) boundary frame, untouched -> intact boundary.
    assert _pixel(image, 0, 0) == _WALL_COLOR_INTACT_BOUNDARY
    # Bomb at world (0,0) -> grid (20,20); its marker covers the center.
    assert _pixel(image, 20, 20) == _WALL_COLOR_BOMB


def test_blast_coverage_tints_free_space():
    truth = _wall_grid()
    with tempfile.TemporaryDirectory() as tmp:
        image = _render(truth, [(0.0, 0.0)], radius=5.0, tmp=Path(tmp))

    # (20,15) free space at the blast boundary (radius 5, off the marker).
    assert _pixel(image, 20, 15) == _WALL_COLOR_BLAST_OVERLAY
    # Free space outside the blast stays white.
    assert _pixel(image, 20, 10) == _COMPARISON_COLOR_WHITE


def test_no_bombs_renders_all_walls_intact():
    truth = _wall_grid()
    with tempfile.TemporaryDirectory() as tmp:
        image = _render(truth, [], tmp=Path(tmp))

    assert _pixel(image, 20, 20) == _WALL_COLOR_INTACT_INTERIOR
    assert _pixel(image, 0, 0) == _WALL_COLOR_INTACT_BOUNDARY
    # No blast -> no blue coverage fill.
    assert _pixel(image, 20, 16) == _COMPARISON_COLOR_WHITE


def test_destroyed_boundary_walls_are_distinguished():
    # Bomb right against the bottom boundary frame, radius large enough to
    # destroy boundary pixels but not reach the interior bar.
    with tempfile.TemporaryDirectory() as tmp:
        image = _render(
            _wall_grid(), [(0.0, -19.0)], radius=6.0, tmp=Path(tmp)
        )

    # World (0,-19) -> grid (39,20); (39,26) is a destroyed boundary pixel
    # within radius 6 but outside the bomb marker.
    assert _pixel(image, 39, 26) == _WALL_COLOR_DESTROYED_BOUNDARY
    # Far side of the frame remains intact.
    assert _pixel(image, 39, 35) == _WALL_COLOR_INTACT_BOUNDARY


def test_save_wall_destruction_image_normalizes_0_255_grids():
    truth = _wall_grid().copy()
    truth[truth > 0] = 255
    with tempfile.TemporaryDirectory() as tmp:
        image = _render(truth, [(0.0, 0.0)], tmp=Path(tmp))

    assert _pixel(image, 25, 20) == _WALL_COLOR_DESTROYED_INTERIOR

import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from swarm_rescue.simulation.reporting.bombs_io import (
    COMPARISON_HEADER_HEIGHT,
    COMPARISON_LEGEND_HEIGHT,
    COMPARISON_SEPARATOR_WIDTH,
    _COMPARISON_COLOR_CORRECT_EMPTY,
    _COMPARISON_COLOR_DEAD_ZONE,
    _COMPARISON_COLOR_HIT_WALL,
    _COMPARISON_COLOR_MISS,
    _COMPARISON_COLOR_NO_CREDIT,
    _COMPARISON_COLOR_OBSTACLE,
    _COMPARISON_COLOR_WHITE,
    _credit_colour,
    exploration_comparison_image_output_path,
    save_exploration_comparison_image,
)
from swarm_rescue.simulation.reporting.evaluation import EvalConfig
from swarm_rescue.tools.map_exploration_scoring import (
    MapExplorationScorer,
    ScoreConfig,
)


def _panel_origin(index: int, width: int) -> int:
    return index * (width + COMPARISON_SEPARATOR_WIDTH)


def _pixel(image: np.ndarray, panel: int, width: int, row: int,
           col: int) -> tuple:
    x0 = _panel_origin(panel, width)
    return tuple(int(v) for v in image[COMPARISON_HEADER_HEIGHT + row, x0 + col])


def test_exploration_comparison_image_output_path_naming():
    team_info = SimpleNamespace(team_number_str_padded="007", team_number_str="7")
    eval_config = EvalConfig(map_name="Map04", zones_config=())

    path = exploration_comparison_image_output_path(
        "/tmp/results", team_info, eval_config, 3
    )

    assert path is not None
    assert Path(path).name == "team007_Map04_none_rd3_exploration_diff.png"


def test_exploration_comparison_image_output_path_none_without_result_path():
    team_info = SimpleNamespace(team_number_str_padded="007", team_number_str="7")
    eval_config = EvalConfig(map_name="Map04", zones_config=())

    assert exploration_comparison_image_output_path(
        None, team_info, eval_config, 1
    ) is None


def test_save_exploration_comparison_image_layout():
    truth = np.zeros((12, 10), dtype=np.uint8)
    truth[4:8, 3:7] = 1
    pred = truth.copy()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        save_exploration_comparison_image(out, truth, pred)

        image = cv2.imread(str(out))

    h, w = truth.shape
    assert image.shape == (
        COMPARISON_HEADER_HEIGHT + h + COMPARISON_LEGEND_HEIGHT,
        3 * w + 2 * COMPARISON_SEPARATOR_WIDTH,
        3,
    )


def test_first_two_panels_render_the_raw_grids():
    truth = np.zeros((12, 10), dtype=np.uint8)
    truth[4:8, 3:7] = 1
    pred = np.zeros_like(truth)
    pred[4:8, 3:7] = 1
    pred[0, 0] = 1  # present only in the submission

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        save_exploration_comparison_image(out, truth, pred)
        image = cv2.imread(str(out))

    h, w = truth.shape
    # Ground-truth panel.
    assert _pixel(image, 0, w, 5, 4) == _COMPARISON_COLOR_OBSTACLE
    assert _pixel(image, 0, w, 0, 0) == _COMPARISON_COLOR_WHITE
    # Submission panel.
    assert _pixel(image, 1, w, 5, 4) == _COMPARISON_COLOR_OBSTACLE
    assert _pixel(image, 1, w, 0, 0) == _COMPARISON_COLOR_OBSTACLE


def test_merged_panel_marks_credit_and_misses():
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[10:14, 10:14] = 1
    truth[30:34, 30:34] = 1

    pred = np.zeros_like(truth)
    pred[10:14, 10:14] = 1  # first block found, second missed

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        save_exploration_comparison_image(out, truth, pred)
        image = cv2.imread(str(out))

    h, w = truth.shape
    assert _pixel(image, 2, w, 11, 11) == _COMPARISON_COLOR_HIT_WALL
    assert _pixel(image, 2, w, 31, 31) == _COMPARISON_COLOR_MISS


def _thick_block_truth() -> np.ndarray:
    """A block thick enough to hold both a wall surface and a deep interior.

    With ``wall_width = 4`` the 12x12 block at (10,10) splits into a wall ring
    plus a 4x4 interior whose centre pixel (16,16) is deep interior, while
    (10,10) stays on the wall surface.
    """
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[10:22, 10:22] = 1
    return truth


def _render(image_truth: np.ndarray, image_pred: np.ndarray,
            scorer: MapExplorationScorer) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        save_exploration_comparison_image(
            out, image_truth, image_pred, scorer=scorer
        )
        return cv2.imread(str(out))


def test_merged_panel_keeps_zero_weight_interior_out_of_credit_green():
    """Default config prices interior at 0.0 -- painting it green was the bug."""
    truth = _thick_block_truth()
    scorer = MapExplorationScorer(ScoreConfig())
    assert scorer.config.w_interior == 0.0

    image = _render(truth, truth.copy(), scorer)
    h, w = truth.shape

    # Deep interior is present but unpriced: neutral, never credit green.
    assert _pixel(image, 2, w, 16, 16) == _COMPARISON_COLOR_NO_CREDIT
    assert _pixel(image, 2, w, 16, 16) != _COMPARISON_COLOR_HIT_WALL
    # The priced wall surface is still credit green.
    assert _pixel(image, 2, w, 10, 10) == _COMPARISON_COLOR_HIT_WALL


def test_merged_panel_zero_weight_interior_miss_is_not_missed_truth():
    truth = _thick_block_truth()
    scorer = MapExplorationScorer(ScoreConfig())

    image = _render(truth, np.zeros_like(truth), scorer)
    h, w = truth.shape

    # An unpriced region forfeits nothing, so its miss is not "missed truth".
    assert _pixel(image, 2, w, 16, 16) == _COMPARISON_COLOR_NO_CREDIT
    assert _pixel(image, 2, w, 16, 16) != _COMPARISON_COLOR_MISS
    # The priced wall surface does read as missed truth.
    assert _pixel(image, 2, w, 10, 10) == _COMPARISON_COLOR_MISS


def test_merged_panel_credit_colour_tracks_region_weight():
    """Both regions priced: both go green, at the shade their weight implies."""
    truth = _thick_block_truth()
    scorer = MapExplorationScorer(ScoreConfig(w_wall=0.5, w_interior=0.5))

    image = _render(truth, truth.copy(), scorer)
    h, w = truth.shape

    midpoint = _credit_colour(0.5)
    assert _pixel(image, 2, w, 16, 16) == midpoint
    assert _pixel(image, 2, w, 10, 10) == midpoint
    assert midpoint != _COMPARISON_COLOR_NO_CREDIT
    assert midpoint != _COMPARISON_COLOR_HIT_WALL


def test_merged_panel_interior_only_config_recolours_both_regions():
    """Flipping the weights flips the picture -- same config drives both."""
    truth = _thick_block_truth()
    scorer = MapExplorationScorer(ScoreConfig(w_wall=0.0, w_interior=1.0))

    image = _render(truth, truth.copy(), scorer)
    h, w = truth.shape

    assert _pixel(image, 2, w, 16, 16) == _COMPARISON_COLOR_HIT_WALL
    assert _pixel(image, 2, w, 10, 10) == _COMPARISON_COLOR_NO_CREDIT


def test_merged_panel_distinguishes_correct_empty_from_background():
    """Scoring nothing must not look the same as having nothing to report."""
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[10:14, 10:14] = 1
    pred = truth.copy()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        save_exploration_comparison_image(out, truth, pred)
        image = cv2.imread(str(out))

    h, w = truth.shape
    empty = _pixel(image, 2, w, 38, 38)
    assert empty == _COMPARISON_COLOR_CORRECT_EMPTY
    assert empty != _COMPARISON_COLOR_WHITE


def test_merged_panel_marks_free_dead_zone_predictions():
    truth = np.zeros((60, 60), dtype=np.uint8)
    truth[10:14, 10:14] = 1

    pred = truth.copy()
    pred[11, 20] = 1  # ~7 px away: inside the default 10 px dead zone

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        save_exploration_comparison_image(out, truth, pred)
        image = cv2.imread(str(out))

    h, w = truth.shape
    assert _pixel(image, 2, w, 11, 20) == _COMPARISON_COLOR_DEAD_ZONE


def test_merged_panel_grades_charged_predictions_by_distance():
    truth = np.zeros((120, 120), dtype=np.uint8)
    truth[10:14, 10:14] = 1

    pred = truth.copy()
    pred[11, 40] = 1   # moderately far
    pred[11, 110] = 1  # far enough to hit the cap

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        save_exploration_comparison_image(out, truth, pred)
        image = cv2.imread(str(out))

    h, w = truth.shape
    near = _pixel(image, 2, w, 11, 40)
    far = _pixel(image, 2, w, 11, 110)

    # Both are red-ish; the farther one is darker (lower blue/green channels).
    assert near != far
    assert far[0] < near[0]
    assert far[1] < near[1]


def test_save_exploration_comparison_image_normalizes_0_255_grids():
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[10:14, 10:14] = 255
    pred = np.zeros((40, 40), dtype=np.uint8)
    pred[10:14, 10:14] = 255

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        save_exploration_comparison_image(out, truth, pred)
        image = cv2.imread(str(out))

    h, w = truth.shape
    assert _pixel(image, 2, w, 11, 11) == _COMPARISON_COLOR_HIT_WALL


def test_save_exploration_comparison_image_reports_score_when_time_given():
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[10:14, 10:14] = 1

    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "plain.png"
        annotated = Path(tmp) / "annotated.png"
        save_exploration_comparison_image(plain, truth, truth.copy())
        save_exploration_comparison_image(annotated, truth, truth.copy(),
                                          elapsed_walltime=12.0)

        # Same geometry either way; the annotated one just prints the arithmetic.
        assert cv2.imread(str(plain)).shape == cv2.imread(str(annotated)).shape
        assert plain.read_bytes() != annotated.read_bytes()


def test_save_exploration_comparison_image_shape_mismatch_raises():
    truth = np.zeros((4, 4), dtype=np.uint8)
    pred = np.zeros((5, 5), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        with pytest.raises(ValueError, match="shape"):
            save_exploration_comparison_image(out, truth, pred)
        assert not out.exists()


def test_save_exploration_comparison_image_rejects_non_2d():
    truth = np.zeros((4, 4), dtype=np.uint8)
    pred = np.zeros((4, 4, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diff.png"
        with pytest.raises(ValueError):
            save_exploration_comparison_image(out, truth, pred)

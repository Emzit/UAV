import pathlib
import sys

import cv2
import numpy as np
import pytest

# Insert the 'src' directory, located two levels up from the current script,
# into sys.path. This ensures Python can find project-specific modules
# (e.g., 'swarm_rescue') when the script is run from a subfolder like 'tests/'.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer, ScoreConfig


def _scorer(*, time_limit: float = 10.0, **overrides) -> MapExplorationScorer:
    cfg = ScoreConfig(
        wall_width=2.0,
        w_wall=0.7,
        w_interior=0.3,
        dead_zone=10.0,
        penalty_step=0.1,
        penalty_repeat=2,
        penalty_cap=1.0,
        time_best=0.0,
        time_limit=time_limit,
        w_map=0.8,
        w_time=0.2,
        **overrides,
    )
    return MapExplorationScorer(cfg)


def _block_truth() -> np.ndarray:
    """3x3 obstacle block in a 5x5 grid. Thin enough to be all wall surface."""
    truth = np.zeros((5, 5), dtype=np.uint8)
    truth[1:4, 1:4] = 1
    return truth


def test_identical_prediction_full_score() -> None:
    truth = _block_truth()
    scorer = _scorer()

    result = scorer.score(truth, truth.copy(), 0.0)

    assert result.wall.available is True
    assert result.wall.rate == pytest.approx(1.0)
    assert result.credit == pytest.approx(1.0)
    assert result.penalty == pytest.approx(0.0)
    assert result.s_map == pytest.approx(1.0)
    assert result.s_time == pytest.approx(1.0)
    assert result.score == pytest.approx(100.0)


def test_time_reduction_with_perfect_map() -> None:
    truth = _block_truth()
    scorer = _scorer(time_limit=10.0)

    result = scorer.score(truth, truth.copy(), 10.0)

    assert result.s_map == pytest.approx(1.0)
    assert result.s_time == pytest.approx(0.0)
    # Only the map term survives: 100 * (0.8 * 1.0 + 0.2 * 0.0)
    assert result.score == pytest.approx(80.0)


def test_interior_unavailable_for_thin_obstacles() -> None:
    """A 1px obstacle has no deep interior, so credit renormalizes onto the wall."""
    truth = np.zeros((5, 5), dtype=np.uint8)
    truth[2, 2] = 1
    scorer = _scorer()

    result = scorer.score(truth, truth.copy(), 0.0)

    assert result.wall.available is True
    assert result.interior.available is False
    assert result.wall.rate == pytest.approx(1.0)
    assert result.credit == pytest.approx(1.0)
    assert result.score == pytest.approx(100.0)


def test_interior_region_is_split_by_thickness() -> None:
    """A thick block has both a wall surface and a deep interior."""
    truth = np.zeros((21, 21), dtype=np.uint8)
    truth[2:19, 2:19] = 1
    scorer = _scorer()

    wall, interior = scorer.split_regions(truth)

    assert int(wall.sum()) > 0
    assert int(interior.sum()) > 0
    assert int(wall.sum()) + int(interior.sum()) == int(truth.sum())

    result = scorer.score(truth, truth.copy(), 0.0)
    assert result.wall.available is True
    assert result.interior.available is True
    assert result.s_map == pytest.approx(1.0)


def test_wall_flush_against_the_grid_edge_has_no_deep_interior() -> None:
    """The arena border is a normal wall whose outer face is the grid edge.

    Regression: the depth field used to ignore the space *outside* the grid, so
    a 6 px border wall read as 7, 6, 5, 4, 3, 2, 1 px deep instead of
    1, 2, 3, 4, 3, 2, 1. Everything past ``wall_width`` was demoted to the
    zero-weight interior -- 39% of the truth on the Map01 demo map -- which
    both shrank the penalty denominator and let a submission covering only the
    inner half of the border claim full credit.
    """
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[0:6, :] = 1  # a 6 px wall flush against the top edge
    scorer = MapExplorationScorer(ScoreConfig())

    wall, interior = scorer.split_regions(truth)

    assert int(interior.sum()) == 0
    assert int(wall.sum()) == int(truth.sum())


def test_border_depth_leaves_interior_walls_untouched() -> None:
    """Treating the outside as free must not disturb walls away from the edge."""
    truth = np.zeros((60, 60), dtype=np.uint8)
    truth[20:26, 20:46] = 1  # 6 px interior wall, 20 px from every edge
    scorer = MapExplorationScorer(ScoreConfig())

    obstacle = truth > 0
    depth = scorer.obstacle_depth(truth)
    naive = cv2.distanceTransform(truth, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    assert np.array_equal(depth[obstacle], naive[obstacle])
    wall, interior = scorer.split_regions(truth)
    assert int(interior.sum()) == 0
    assert int(wall.sum()) == int(truth.sum())


def test_empty_prediction_earns_no_credit() -> None:
    truth = _block_truth()
    scorer = _scorer()

    result = scorer.score(truth, np.zeros_like(truth), 0.0)

    assert result.wall.hits == 0
    assert result.credit == pytest.approx(0.0)
    # Nothing was predicted, so nothing can be charged.
    assert result.penalty == pytest.approx(0.0)
    assert result.s_map == pytest.approx(0.0)
    assert result.s_time == pytest.approx(1.0)
    assert result.score == pytest.approx(20.0)


def test_both_empty_gives_time_score_only() -> None:
    empty = np.zeros((5, 5), dtype=np.uint8)
    scorer = _scorer(time_limit=10.0)

    result = scorer.score(empty, empty.copy(), 5.0)

    assert result.wall.available is False
    assert result.interior.available is False
    assert result.credit == pytest.approx(0.0)
    assert result.s_map == pytest.approx(0.0)
    assert result.s_time == pytest.approx(0.5)
    # 100 * (0.8 * 0.0 + 0.2 * 0.5)
    assert result.score == pytest.approx(10.0)


def test_stray_prediction_far_from_truth_is_charged() -> None:
    truth = np.zeros((80, 80), dtype=np.uint8)
    truth[10:13, 10:13] = 1

    pred = truth.copy()
    pred[70, 70] = 1  # far beyond the dead zone

    scorer = _scorer()
    result = scorer.score(truth, pred, 0.0)

    assert result.credit == pytest.approx(1.0)
    assert result.charged_pixels == 1
    assert result.capped_pixels == 1
    assert result.total_charge == pytest.approx(1.0)
    # Normalized by the wall surface (not the whole obstacle layer) and scaled
    # by penalty_weight.
    assert result.surface_pixels == int(truth.sum())
    assert result.penalty == pytest.approx(
        scorer.config.penalty_weight / result.surface_pixels)
    assert result.s_map < 1.0


def test_prediction_inside_dead_zone_is_free() -> None:
    truth = np.zeros((80, 80), dtype=np.uint8)
    truth[10:13, 10:13] = 1

    pred = truth.copy()
    pred[10, 18] = 1  # 6 px away, inside the 10 px dead zone

    scorer = _scorer()
    result = scorer.score(truth, pred, 0.0)

    assert result.dead_zone_pixels == 1
    assert result.charged_pixels == 0
    assert result.total_charge == pytest.approx(0.0)
    assert result.s_map == pytest.approx(1.0)


def test_shape_mismatch_raises() -> None:
    scorer = _scorer()
    with pytest.raises(ValueError, match="shape mismatch"):
        scorer.score(np.zeros((4, 4), np.uint8), np.zeros((5, 5), np.uint8), 0.0)


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="w_wall"):
        MapExplorationScorer(ScoreConfig(w_wall=0.5, w_interior=0.3))
    with pytest.raises(ValueError, match="w_map"):
        MapExplorationScorer(ScoreConfig(w_map=0.5, w_time=0.1))


def test_penalty_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="penalty_step"):
        MapExplorationScorer(ScoreConfig(penalty_step=0.0))
    with pytest.raises(ValueError, match="penalty_repeat"):
        MapExplorationScorer(ScoreConfig(penalty_repeat=0))
    with pytest.raises(ValueError, match="penalty_cap"):
        MapExplorationScorer(ScoreConfig(penalty_cap=0.0))
    with pytest.raises(ValueError, match="dead_zone"):
        MapExplorationScorer(ScoreConfig(dead_zone=-1.0))

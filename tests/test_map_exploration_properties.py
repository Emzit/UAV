"""Properties the exploration-map scorer must hold.

These replace the old soft-boundary suite, which existed to justify the
exp(-d/tau) boundary term. The properties that still matter are kept: an exact
map scores full marks, accuracy loss is monotone, and -- the key incentive
property -- drawing honestly beats inflating or thinning the submission.
"""

import pathlib
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer, ScoreConfig


def _scorer(**overrides) -> MapExplorationScorer:
    return MapExplorationScorer(ScoreConfig(time_limit=100.0, **overrides))


def _walls(width: int = 800, height: int = 600) -> np.ndarray:
    """Grid with 6 px thick walls: a border plus one interior partition.

    The grid has to be large enough that the wall surface is a realistic share
    of the area -- the shipped maps sit near 6%, while a 6 px border on a small
    grid is far denser (a 240x200 grid reaches 13%). That matters because the
    penalty is normalized by the wall-surface count: ``split_regions`` now
    measures a border wall's thickness against the space outside the grid, so
    the denominator is honest and these anti-gaming properties hold at the
    densities that actually occur.
    """
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[0:6, :] = 1
    grid[-6:, :] = 1
    grid[:, 0:6] = 1
    grid[:, -6:] = 1
    grid[:, 100:106] = 1
    return grid


def _shift(grid: np.ndarray, offset: int) -> np.ndarray:
    return np.roll(grid, offset, axis=1)


def _dilate(grid: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return grid.copy()
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(grid, kernel, iterations=1)


def _erode(grid: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return grid.copy()
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.erode(grid, kernel, iterations=1)


def test_identical_prediction_scores_full_marks() -> None:
    truth = _walls()
    result = _scorer().score(truth, truth.copy(), 0.0)

    assert result.credit == pytest.approx(1.0)
    assert result.penalty == pytest.approx(0.0)
    assert result.s_map == pytest.approx(1.0)
    assert result.score == pytest.approx(100.0)


def test_empty_prediction_earns_no_credit() -> None:
    truth = _walls()
    result = _scorer().score(truth, np.zeros_like(truth), 0.0)

    assert result.credit == pytest.approx(0.0)
    assert result.s_map == pytest.approx(0.0)


def test_accuracy_loss_is_monotone_under_shift() -> None:
    """A more badly misplaced map must never score higher."""
    truth = _walls()
    scorer = _scorer()

    scores = [scorer.score(truth, _shift(truth, offset), 0.0).s_map
              for offset in range(0, 40, 2)]

    assert scores[0] == pytest.approx(1.0)
    for earlier, later in zip(scores, scores[1:]):
        assert later <= earlier + 1e-9


def test_bulk_filling_the_map_scores_zero() -> None:
    """Painting everything satisfies credit trivially, so the charge must sink it."""
    truth = _walls()
    result = _scorer().score(truth, np.ones_like(truth), 0.0)

    assert result.credit == pytest.approx(1.0)
    # The charge has to wipe out a perfect credit score almost entirely.
    assert result.penalty > 0.95
    assert result.s_map < 0.05


def test_dense_noise_scores_zero() -> None:
    truth = _walls()
    rng = np.random.default_rng(0)
    noise = (rng.random(truth.shape) < 0.5).astype(np.uint8)

    # With wall-only weighting, noise pixels landing on wall surfaces earn a
    # little credit; the charge still wipes out essentially all of it.
    assert _scorer().score(truth, noise, 0.0).s_map == pytest.approx(0.0, abs=0.01)


def test_runaway_inflation_collapses_the_score() -> None:
    """Inflating without bound must end in a near-zero score.

    On a sparse map some dilation can gain: this fixture is ~96% empty, so
    thickening walls picks up truth the shifted submission missed, and errors
    that small are exactly what the dead zone forgives. What matters is that the
    gain is bounded and reverses -- runaway inflation collapses.

    Dense real geometry has no such window; see
    ``test_inflation_never_pays_once_coverage_is_complete``.
    """
    truth = _walls()
    scorer = _scorer()
    submission = _shift(truth, 6)

    baseline = scorer.score(truth, submission, 0.0).s_map
    scores = [scorer.score(truth, _dilate(submission, radius), 0.0).s_map
              for radius in (10, 20, 30, 50)]

    # Monotonically worsening once past the peak, ending well below baseline.
    for earlier, later in zip(scores, scores[1:]):
        assert later < earlier
    assert scores[-1] < 0.05
    assert scores[-1] < baseline


def test_inflation_never_pays_once_coverage_is_complete() -> None:
    """With credit already maxed out, dilation can only add charge.

    This is the incentive that matters in practice: real submissions over-cover
    (team007 predicted 1.58x the truth pixel count), so every extra pixel is
    pure downside.
    """
    truth = _walls()
    scorer = _scorer()

    baseline = scorer.score(truth, truth.copy(), 0.0)
    assert baseline.credit == pytest.approx(1.0)

    previous = baseline.s_map
    for radius in (2, 4, 6, 8, 12, 16):
        inflated = scorer.score(truth, _dilate(truth, radius), 0.0)
        assert inflated.credit == pytest.approx(1.0)
        assert inflated.s_map <= previous + 1e-9
        previous = inflated.s_map

    assert previous < baseline.s_map


def test_charge_saturates_at_the_cap() -> None:
    truth = np.zeros((200, 200), dtype=np.uint8)
    truth[100, 100] = 1
    scorer = _scorer()

    near = np.array([scorer.config.dead_zone + 2.0], dtype=np.float32)
    far = np.array([scorer.config.dead_zone + 500.0], dtype=np.float32)

    assert scorer.penalty_curve(near)[0] < scorer.config.penalty_cap
    assert scorer.penalty_curve(far)[0] == pytest.approx(scorer.config.penalty_cap)


def test_dead_zone_boundary_is_exact() -> None:
    scorer = _scorer(dead_zone=10.0)
    at_edge = np.array([10.0], dtype=np.float32)
    just_past = np.array([10.5], dtype=np.float32)

    assert scorer.penalty_curve(at_edge)[0] == pytest.approx(0.0)
    assert scorer.penalty_curve(just_past)[0] > 0.0


def test_penalty_schedule_repeats_each_step() -> None:
    """penalty_repeat=2 gives the 0.1 0.1 0.2 0.2 ... staircase."""
    scorer = _scorer(dead_zone=10.0, penalty_step=0.1, penalty_repeat=2)
    distances = np.arange(11.0, 21.0, dtype=np.float32)

    charges = scorer.penalty_curve(distances)

    assert charges.tolist() == pytest.approx(
        [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5])


def test_partial_coverage_scores_between_zero_and_one() -> None:
    truth = _walls()
    half = truth.copy()
    half[:, truth.shape[1] // 2:] = 0

    result = _scorer().score(truth, half, 0.0)

    assert 0.0 < result.s_map < 1.0
    assert result.penalty == pytest.approx(0.0)


def test_thinning_loses_credit() -> None:
    truth = _walls()
    scorer = _scorer()

    thinned = scorer.score(truth, _erode(truth, 2), 0.0)

    assert thinned.credit < 1.0
    assert thinned.s_map < 1.0


def test_penalty_is_normalized_by_wall_surface_not_obstacle_count() -> None:
    """The same stray prediction must cost the same on both maps.

    Adding a large unreachable blob inflates the obstacle count without telling
    the team anything more about where wall faces are. Normalizing by the full
    obstacle layer would make the identical mistake cheaper on the blob map.
    """
    def build(with_blob: bool) -> np.ndarray:
        grid = np.zeros((400, 600), dtype=np.uint8)
        grid[0:6, :] = 1
        grid[-6:, :] = 1
        grid[:, 0:6] = 1
        grid[:, -6:] = 1
        grid[:, 200:206] = 1
        if with_blob:
            grid[300:390, 400:590] = 1
        return grid

    scorer = _scorer()
    penalties = []
    for with_blob in (False, True):
        truth = build(with_blob)
        stray = truth.copy()
        stray[100:110, 300:310] = 1  # identical mistake on both maps
        result = scorer.score(truth, stray, 0.0)
        penalties.append(result.penalty)
        assert result.surface_pixels < result.truth_pixels or not with_blob

    without_blob, with_blob_penalty = penalties
    # The blob changes the obstacle count a lot but the wall surface only a
    # little, so the cost stays within a modest factor.
    assert with_blob_penalty == pytest.approx(without_blob, rel=0.35)


def test_penalty_weight_scales_the_charge_linearly() -> None:
    truth = _walls()
    stray = truth.copy()
    stray[truth.shape[0] // 2, truth.shape[1] // 2] = 1

    full = _scorer(penalty_weight=0.4).score(truth, stray, 0.0)
    half = _scorer(penalty_weight=0.2).score(truth, stray, 0.0)

    assert full.total_charge == pytest.approx(half.total_charge)
    assert full.penalty == pytest.approx(2.0 * half.penalty)


def test_zero_penalty_weight_disables_the_charge() -> None:
    truth = _walls()
    result = _scorer(penalty_weight=0.0).score(truth, np.ones_like(truth), 0.0)

    assert result.total_charge > 0.0
    assert result.penalty == pytest.approx(0.0)
    assert result.s_map == pytest.approx(1.0)

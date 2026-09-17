import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.simulation.reporting.place_score_manager import (
    PlacePhase1Result,
    PlaceScoreConfig,
    PlaceScoreManager,
    ReferenceBlueRun,
    bomb_score_from_difficulty,
    collect_wall_polygons,
    difficulty_from_blue_score,
    load_place_score_config,
    placement_completion,
    rasterize_world_polygons,
    walls_and_boundary_grid,
    walls_only_grid,
)
from swarm_rescue.simulation.utils.definitions import CollisionTypes


def _scorer_manager() -> PlaceScoreManager:
    return PlaceScoreManager(PlaceScoreConfig())


def test_no_submission_gives_zero_map_score() -> None:
    truth = np.zeros((5, 5), dtype=np.uint8)
    truth[1:4, 1:4] = 1
    result = _scorer_manager().compute_map_score(
        truth,
        None,
        elapsed_walltime=0.0,
        max_walltime_limit=100.0,
    )
    assert result.score == 0.0
    assert result.error == "no exploration map submitted"


def test_shape_mismatch_gives_zero_map_score() -> None:
    truth = np.zeros((5, 5), dtype=np.uint8)
    pred = np.zeros((3, 3), dtype=np.uint8)
    result = _scorer_manager().compute_map_score(
        truth,
        pred,
        elapsed_walltime=0.0,
        max_walltime_limit=100.0,
    )
    assert result.score == 0.0
    assert "mismatch" in (result.error or "")


def test_identical_map_full_score() -> None:
    truth = np.zeros((5, 5), dtype=np.uint8)
    truth[1:4, 1:4] = 1
    pred = truth.copy()
    result = _scorer_manager().compute_map_score(
        truth,
        pred,
        elapsed_walltime=0.0,
        max_walltime_limit=10.0,
    )
    assert result.score == pytest.approx(100.0)


def test_map_score_penalizes_non_wall_fill_in_submission() -> None:
    # Truth is the walls-only grid (real wall entities). A submission that
    # also draws a non-wall solid fill -- e.g. the disposal center -- must
    # not earn credit for it; the stray blob sits beyond the dead zone from
    # any real wall and pays the false-prediction charge instead.
    truth = np.zeros((60, 60), dtype=np.uint8)
    truth[20:40, 29:31] = 1  # one real wall line
    pred_wall_only = truth.copy()
    pred_with_fill = pred_wall_only.copy()
    pred_with_fill[5:20, 5:22] = 1  # disposal-center-like solid rectangle

    manager = _scorer_manager()
    honest = manager.compute_map_score(
        truth, pred_wall_only,
        elapsed_walltime=0.0, max_walltime_limit=10.0,
    )
    filled = manager.compute_map_score(
        truth, pred_with_fill,
        elapsed_walltime=0.0, max_walltime_limit=10.0,
    )
    assert honest.score == pytest.approx(100.0)
    assert filled.score < honest.score
    assert filled.breakdown["truth_pixels"] == int(truth.sum())
    # The non-wall blob is not part of the truth: it is charged, not credited.
    assert filled.breakdown["charged_pixels"] > 0


def test_map_score_full_marks_with_walls_only_truth() -> None:
    # The production pipeline passes the walls-only grid as the exploration
    # truth (the full-obstacle render is never used for map scoring). A
    # perfect walls-only submission reaches full marks even when the arena
    # also contains a non-wall solid platform.
    truth = np.zeros((60, 60), dtype=np.uint8)
    truth[20:40, 29:31] = 1  # real wall
    pred_walls_only = truth.copy()

    manager = _scorer_manager()
    result = manager.compute_map_score(
        truth, pred_walls_only,
        elapsed_walltime=0.0, max_walltime_limit=10.0,
    )
    assert result.score == pytest.approx(100.0)
    assert result.breakdown["truth_pixels"] == int(truth.sum())


def test_phase1_score_is_partial_with_zero_bomb() -> None:
    truth = np.zeros((5, 5), dtype=np.uint8)
    truth[1:4, 1:4] = 1
    phase1 = _scorer_manager().compute_phase1_score(
        truth,
        truth.copy(),
        elapsed_walltime=0.0,
        max_walltime_limit=10.0,
        score_exploration_trajectory=42.0,
        placed_count=8,
        expected_count=10,
        has_crashed=False,
    )
    assert phase1.score_status == "partial"
    assert phase1.score_bomb == 0.0
    assert phase1.score_map == pytest.approx(100.0)
    assert phase1.round_score == pytest.approx(50.0)
    assert phase1.bomb_breakdown["pending"] is True


def test_bomb_score_uses_reference_blues_and_completion() -> None:
    manager = _scorer_manager()
    runs = [
        ReferenceBlueRun("a.zip", blue_score=40.0, difficulty=0.0, has_crashed=False),
        ReferenceBlueRun("b.zip", blue_score=60.0, difficulty=0.0, has_crashed=False),
    ]
    result = manager.compute_bomb_score(runs, placed_count=5, expected_count=10)
    # difficulty: min(100, (100-40)*2)=100, min(100, (100-60)*2)=80 -> mean 90.
    assert result.score == pytest.approx(0.5 * 90.0)
    assert result.sufficient_references is True
    assert result.breakdown["placement_completion"] == pytest.approx(0.5)
    assert len(result.breakdown["reference_blues"]) == 2


def test_difficulty_from_blue_score_stretch_and_clamp() -> None:
    """Blue score -> difficulty: (100 - s) * 2, clamped to [0, 100]."""
    f = difficulty_from_blue_score
    assert f(100.0) == pytest.approx(0.0)   # perfect blue -> no difficulty
    assert f(80.0) == pytest.approx(40.0)   # linear stretch in 50..100
    assert f(50.0) == pytest.approx(100.0)  # saturates at blue score 50
    assert f(30.0) == pytest.approx(100.0)  # clamped above saturation
    assert f(0.0) == pytest.approx(100.0)   # crashed-level score clamps too
    assert f(-10.0) == pytest.approx(100.0)  # defensive lower clamp


def test_crashed_reference_blue_excluded_from_difficulty() -> None:
    manager = PlaceScoreManager(
        PlaceScoreConfig(crash_difficulty=100.0)
    )
    runs = [
        ReferenceBlueRun("a.zip", blue_score=0.0, difficulty=0.0, has_crashed=True, success=False),
    ]
    result = manager.compute_bomb_score(runs, placed_count=2, expected_count=2)
    assert result.score == pytest.approx(0.0)
    assert result.sufficient_references is False
    assert len(result.breakdown["failed_reference_blues"]) == 1


def test_merge_final_score() -> None:
    manager = _scorer_manager()
    phase1 = PlacePhase1Result(
        score_map=80.0,
        score_bomb=0.0,
        round_score=40.0,
        score_status="partial",
        map_breakdown={"s_boundary": 1.0},
        bomb_breakdown={"pending": True},
        score_exploration_trajectory=30.0,
        score_wall=50.0,
        wall_breakdown={"score_fraction": 0.5},
        score_time=100.0,
        time_breakdown={"score_fraction": 1.0},
    )
    from swarm_rescue.simulation.reporting.place_score_manager import BombScoreResult

    bomb = BombScoreResult(
        score=60.0,
        breakdown={"placement_completion": 1.0, "reference_blues": []},
        sufficient_references=True,
    )
    final = manager.merge_final_score(phase1, bomb)
    assert final.score_status == "final"
    assert final.score_bomb == 60.0
    assert final.score_wall == 50.0
    assert final.score_time == 100.0
    # Default weights 0.5 / 0.2 / 0.2 / 0.1.
    assert final.round_score == pytest.approx(
        0.5 * 80 + 0.2 * 60 + 0.2 * 50 + 0.1 * 100
    )
    assert final.bomb_breakdown["pending"] is False


def test_bomb_score_from_difficulty_applies_completion() -> None:
    assert bomb_score_from_difficulty(
        [30.0, 40.0], placed_count=9, expected_count=10,
    ) == pytest.approx(0.9 * 35.0)
    assert bomb_score_from_difficulty(
        [30.0, 40.0], placed_count=10, expected_count=10,
    ) == pytest.approx(35.0)
    assert bomb_score_from_difficulty(
        [30.0, 40.0], placed_count=12, expected_count=10,
    ) == pytest.approx(35.0)


def test_bomb_score_from_difficulty_median_mode() -> None:
    assert bomb_score_from_difficulty(
        [30.0, 60.0], placed_count=10, expected_count=10, use_median=True,
    ) == pytest.approx(45.0)


def test_bomb_score_from_difficulty_empty_and_missing_expected() -> None:
    assert bomb_score_from_difficulty([], placed_count=9, expected_count=10) == 0.0
    # expected_count unknown -> completion = 1.0 when nothing was placed
    assert bomb_score_from_difficulty([30.0], placed_count=0, expected_count=None) == pytest.approx(30.0)
    assert bomb_score_from_difficulty([30.0], placed_count=3, expected_count=None) == 0.0


# ---------------------------------------------------------------------------
# Wall-destruction scoring
# ---------------------------------------------------------------------------


def _wall_test_grid() -> np.ndarray:
    """40x40 grid: boundary ring + an interior horizontal bar.

    World coordinates map via col = x + 20, row = 20 - y.
    """
    truth = np.zeros((40, 40), dtype=np.uint8)
    truth[0, :] = 1
    truth[-1, :] = 1
    truth[:, 0] = 1
    truth[:, -1] = 1
    # Interior bar, rows 10..29, cols 20..21 (world y in [-10, 10], x in [0, 2]).
    truth[10:30, 20:22] = 1
    return truth


def test_wall_score_requires_bombs() -> None:
    manager = _scorer_manager()
    result = manager.compute_wall_score([], _wall_test_grid())
    assert result.score == 0.0
    assert result.breakdown["n_bombs"] == 0


def test_wall_score_interior_destruction() -> None:
    manager = _scorer_manager()
    # Bomb at world (0, 0) -> grid (20, 20); radius 5 (ratio 5/40 = 0.125)
    # reaches only the interior bar, not the boundary frame.
    result = manager.compute_wall_score(
        [(0.0, 0.0)], _wall_test_grid(), blast_radius=0.125,
        boundary_wall_weight=0.5, expected_n_bombs=1,
    )
    assert result.score > 0.0
    assert result.breakdown["destroyed_interior_pixels"] > 0
    assert result.breakdown["destroyed_boundary_pixels"] == 0
    assert result.breakdown["total_boundary_pixels"] > 0
    assert result.breakdown["total_interior_pixels"] == 20 * 2
    assert result.breakdown["blast_radius"] == 5  # ratio -> px on 40x40 grid
    assert result.breakdown["blast_radius_ratio"] == pytest.approx(0.125)
    assert result.breakdown["theoretical_optimum"] > 0
    assert result.breakdown["reference_n_bombs"] == 1


def test_wall_score_boundary_weight_effect() -> None:
    manager = _scorer_manager()
    # Bomb at world (0, -18) -> grid (38, 20): only the bottom boundary frame
    # is within blast radius 4 (ratio 4/40 = 0.1).
    bombs = [(0.0, -18.0)]
    grid = _wall_test_grid()
    score_zero = manager.compute_wall_score(
        bombs, grid, blast_radius=0.1, boundary_wall_weight=0.0,
        expected_n_bombs=1,
    )
    score_half = manager.compute_wall_score(
        bombs, grid, blast_radius=0.1, boundary_wall_weight=0.5,
        expected_n_bombs=1,
    )
    score_full = manager.compute_wall_score(
        bombs, grid, blast_radius=0.1, boundary_wall_weight=1.0,
        expected_n_bombs=1,
    )
    assert score_zero.breakdown["destroyed_boundary_pixels"] > 0
    assert score_zero.score == 0.0
    assert 0.0 < score_half.score < score_full.score
    assert score_full.score > 0.0


def test_wall_score_radius_saturation() -> None:
    manager = _scorer_manager()
    # A radius large enough to cover every wall pixel -> full marks (the
    # theoretical optimum is also the whole wall surface in that case).
    result = manager.compute_wall_score(
        [(0.0, 0.0)], _wall_test_grid(), blast_radius=1.0, expected_n_bombs=1
    )
    assert result.score == pytest.approx(1.0)
    assert result.breakdown["coverage_fraction"] == pytest.approx(1.0)


def test_wall_score_overlapping_areas_not_double_counted() -> None:
    manager = _scorer_manager()
    # Two bombs right next to each other destroy no more wall than one bomb of
    # similar coverage; in particular score stays <= 1.0.
    result = manager.compute_wall_score(
        [(0.0, 0.0), (1.0, 0.0)], _wall_test_grid(), blast_radius=1.0,
        expected_n_bombs=2,
    )
    assert result.score == pytest.approx(1.0)


def test_wall_score_normalized_by_theoretical_optimum() -> None:
    """Score = coverage / deterministic greedy optimum; a sub-optimal layout
    scores below 1 and the reference is stable across calls."""
    manager = _scorer_manager()
    grid = _wall_test_grid()
    # A single bomb in a corner misses most interior walls -> below full marks.
    suboptimal = manager.compute_wall_score(
        [(0.0, -18.0)], grid, blast_radius=0.125, expected_n_bombs=2
    )
    assert 0.0 < suboptimal.score < 1.0
    # The optimum is recomputable and identical every call (stable reference).
    opt_a = manager.wall_theoretical_optimum(
        grid, n_bombs=2, blast_radius=5.0, boundary_weight=0.5)
    opt_b = manager.wall_theoretical_optimum(
        grid, n_bombs=2, blast_radius=5.0, boundary_weight=0.5)
    assert opt_a == pytest.approx(opt_b)
    assert opt_a > 0.0
    assert suboptimal.breakdown["theoretical_optimum"] == pytest.approx(opt_a)
    assert suboptimal.breakdown["score_fraction"] == pytest.approx(
        suboptimal.score)


def test_phase1_includes_wall_score() -> None:
    manager = _scorer_manager()
    truth = _wall_test_grid()
    phase1 = manager.compute_phase1_score(
        truth,
        None,  # no exploration map submitted -> score_map 0
        elapsed_walltime=0.0,
        max_walltime_limit=10.0,
        score_exploration_trajectory=42.0,
        placed_count=1,
        expected_count=2,
        has_crashed=False,
        bomb_positions=[(0.0, 0.0)],
    )
    assert phase1.score_wall > 0.0
    assert phase1.wall_breakdown["n_bombs"] == 1
    # map=0, bomb=0, wall>0 -> round = 0.2 * score_wall.
    assert phase1.round_score == pytest.approx(0.2 * phase1.score_wall)


def test_phase1_wall_score_zero_on_crash() -> None:
    manager = _scorer_manager()
    truth = _wall_test_grid()
    phase1 = manager.compute_phase1_score(
        truth,
        None,
        elapsed_walltime=0.0,
        max_walltime_limit=10.0,
        score_exploration_trajectory=42.0,
        placed_count=1,
        expected_count=2,
        has_crashed=True,
        bomb_positions=[(0.0, 0.0)],
    )
    assert phase1.score_wall == 0.0
    assert "crashed" in phase1.wall_breakdown.get("error", "")


def test_load_place_score_config_legacy_and_new() -> None:
    legacy = load_place_score_config({"w_exploration": 0.6, "w_bomb": 0.4})
    assert legacy.w_wall == 0.0
    assert legacy.w_time == 0.0
    assert legacy.w_exploration + legacy.w_bomb == pytest.approx(1.0)

    new_cfg = load_place_score_config({
        "w_exploration": 0.5,
        "w_bomb": 0.2,
        "w_wall": 0.2,
        "w_time": 0.1,
        "blast_radius": 0.15,
        "boundary_wall_weight": 0.3,
        "time_metric": "timestep",
        "time_best_ratio": 0.6,
    })
    assert new_cfg.w_wall == pytest.approx(0.2)
    assert new_cfg.w_time == pytest.approx(0.1)
    assert new_cfg.blast_radius == pytest.approx(0.15)
    assert new_cfg.boundary_wall_weight == pytest.approx(0.3)
    assert new_cfg.time_metric == "timestep"
    assert new_cfg.time_best_ratio == pytest.approx(0.6)
    PlaceScoreManager(new_cfg)  # must validate

    defaults = load_place_score_config(None)
    assert defaults.w_wall == pytest.approx(0.2)
    assert defaults.w_time == pytest.approx(0.1)
    assert defaults.blast_radius == pytest.approx(0.12)
    assert defaults.boundary_wall_weight == pytest.approx(0.5)
    assert defaults.time_metric == "timestep"
    assert defaults.time_best_ratio == pytest.approx(0.5)


def test_place_score_config_validation() -> None:
    with pytest.raises(ValueError):
        PlaceScoreConfig(w_wall=1.0).validate()  # sum 0.5+0.2+1.0+0.1 > 1
    with pytest.raises(ValueError):
        PlaceScoreConfig(w_time=0.3).validate()  # sum 1.2 > 1
    with pytest.raises(ValueError):
        PlaceScoreConfig(w_exploration=0.1).validate()  # sum 0.6 < 1
    with pytest.raises(ValueError):
        PlaceScoreConfig(boundary_wall_weight=1.5).validate()
    with pytest.raises(ValueError):
        PlaceScoreConfig(blast_radius=0.0).validate()
    with pytest.raises(ValueError):
        PlaceScoreConfig(blast_radius=1.5).validate()  # ratio must be <= 1
    with pytest.raises(ValueError):
        PlaceScoreConfig(time_metric="seconds").validate()
    with pytest.raises(ValueError):
        PlaceScoreConfig(time_best_ratio=0.0).validate()
    with pytest.raises(ValueError):
        PlaceScoreConfig(w_exploration=-0.1).validate()


# ---------------------------------------------------------------------------
# Walls-only grid (real wall entities, not the rendered obstacle grid)
# ---------------------------------------------------------------------------


class _FakeShape:
    def __init__(self, vertices):
        self._vertices = list(vertices)

    def get_vertices(self):
        return list(self._vertices)


class _FakeElement:
    def __init__(self, collision_type, vertices=None):
        self._collision_type = collision_type
        self._pm_shapes = (
            [_FakeShape(vertices)] if vertices is not None else []
        )


def _fake_playground(elements, size=(40, 40)):
    return type("PG", (), {"size": size, "elements": elements})()


def test_rasterize_world_polygons_from_world_coords() -> None:
    # World rect x in [-6,-2], y in [-1,1] on a 40x40 grid (world x,y in [-20,20]).
    poly = [(-6.0, -1.0), (-2.0, -1.0), (-2.0, 1.0), (-6.0, 1.0)]
    grid = rasterize_world_polygons([poly], 40, 40)

    assert grid.shape == (40, 40)
    # col = x+20 (14..18), row = 20-y (19..21) -> covered interior.
    assert grid[20, 16] == 1
    assert grid[19, 14] == 1
    assert grid[21, 18] == 1
    # Outside the rect stays free.
    assert grid[5, 5] == 0
    assert grid[0, 0] == 0


def test_collect_wall_polygons_only_wall_collision_type() -> None:
    wall_poly = [(-6.0, -1.0), (-2.0, -1.0), (-2.0, 1.0), (-6.0, 1.0)]
    disposal_poly = [(-20.0, -20.0), (-18.0, -20.0), (-18.0, 20.0), (-20.0, 20.0)]
    pg = _fake_playground([
        _FakeElement(CollisionTypes.WALL, wall_poly),
        _FakeElement(CollisionTypes.DISPOSAL_CENTER, disposal_poly),
        _FakeElement(CollisionTypes.RETURN_AREA, disposal_poly),
    ])
    polys = collect_wall_polygons(pg)
    assert len(polys) == 1
    grid = walls_only_grid(pg)
    assert grid is not None
    assert int(grid.sum()) > 0
    # The disposal/return-area rect (rows 0..40 near col 0) must be excluded.
    assert int(grid[:, 0:2].sum()) == 0


def test_walls_only_grid_excludes_non_wall_fills() -> None:
    wall_poly = [(-6.0, -1.0), (-2.0, -1.0), (-2.0, 1.0), (-6.0, 1.0)]
    big_fill_poly = [(-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)]
    pg = _fake_playground([
        _FakeElement(CollisionTypes.WALL, wall_poly),
        _FakeElement(CollisionTypes.DISPOSAL_CENTER, big_fill_poly),
    ])
    grid = walls_only_grid(pg)
    assert grid is not None
    # Only the small wall rect is present.
    assert int(grid.sum()) == int(grid[18:23, 13:19].sum())
    assert grid[20, 16] == 1


def test_walls_only_grid_falls_back_to_truth_grid() -> None:
    truth = np.ones((40, 40), dtype=np.uint8)
    pg = _fake_playground([
        _FakeElement(CollisionTypes.DISPOSAL_CENTER),
        _FakeElement(CollisionTypes.RETURN_AREA),
    ])
    assert walls_only_grid(pg, truth_grid=truth) is truth
    # Without fallback and without walls -> None.
    assert walls_only_grid(pg) is None


def test_walls_and_boundary_grid_marks_only_frame_walls() -> None:
    # A wall along the left map edge (x in [-20,-17]) and an interior wall.
    frame_poly = [(-20.0, -10.0), (-17.0, -10.0), (-17.0, 10.0), (-20.0, 10.0)]
    interior_poly = [(-6.0, -1.0), (-2.0, -1.0), (-2.0, 1.0), (-6.0, 1.0)]
    pg = _fake_playground([
        _FakeElement(CollisionTypes.WALL, frame_poly),
        _FakeElement(CollisionTypes.WALL, interior_poly),
    ])
    walls, boundary = walls_and_boundary_grid(pg)

    assert walls is not None and boundary is not None
    # Frame wall present in boundary; interior wall not in boundary.
    assert int((boundary & walls).sum()) > 0
    assert int((boundary & walls).sum()) < int(walls.sum())
    # Interior wall pixels (rows 19..21, cols 14..18) are excluded from boundary.
    assert int(boundary[18:23, 13:19].sum()) == 0
    # Boundary covers the frame pixels (rows 10..30, cols 0..3).
    assert int(boundary[10:31, 0:4].sum()) > 0


def test_boundary_excludes_near_edge_inner_walls() -> None:
    # Regression: Map04 has long inner-ring walls sitting just inside the
    # frame. Only the wall that touches the arena border is boundary -- a
    # wall whose extent runs close to the edge but does not touch it is
    # interior, even though it spans nearly the whole side.
    frame = [(-20.0, -10.0), (-17.0, -10.0), (-17.0, 10.0), (-20.0, 10.0)]
    inner = [(-11.0, -10.0), (-8.0, -10.0), (-8.0, 10.0), (-11.0, 10.0)]
    deep = [(-6.0, -1.0), (-2.0, -1.0), (-2.0, 1.0), (-6.0, 1.0)]
    pg = _fake_playground([
        _FakeElement(CollisionTypes.WALL, frame),
        _FakeElement(CollisionTypes.WALL, inner),
        _FakeElement(CollisionTypes.WALL, deep),
    ])
    walls, boundary = walls_and_boundary_grid(pg)

    assert walls is not None and boundary is not None
    # Boundary is exactly the frame wall (world x in [-20,-17] -> cols 0..3).
    expected = rasterize_world_polygons([frame], 40, 40)
    assert int(boundary.sum()) == int(expected.sum())
    # The near-edge inner wall (cols 9..12) is not boundary.
    assert int(boundary[8:32, 9:13].sum()) == 0
    # The deep interior wall (cols 14..18) is not boundary.
    assert int(boundary[18:23, 13:19].sum()) == 0


def test_boundary_excludes_partition_abutting_the_frame() -> None:
    # Regression: Map01's two interior partitions run from the frame inwards,
    # so their bounding box *touches* the arena border even though they are
    # not frame walls. Only walls running *along* the border are boundary.
    frame = [(-20.0, -20.0), (-17.0, -20.0), (-17.0, 20.0), (-20.0, 20.0)]
    # Partition hanging from the top border (y = 20) down to y = 0.
    partition = [(-1.0, 20.0), (1.0, 20.0), (1.0, 0.0), (-1.0, 0.0)]
    pg = _fake_playground([
        _FakeElement(CollisionTypes.WALL, frame),
        _FakeElement(CollisionTypes.WALL, partition),
    ])
    walls, boundary = walls_and_boundary_grid(pg)

    assert walls is not None and boundary is not None
    # Boundary is exactly the frame wall; the partition stays interior even
    # though its top edge lies on the arena border.
    expected = rasterize_world_polygons([frame], 40, 40)
    assert int(boundary.sum()) == int(expected.sum())
    # The partition body (world x in [-1,1] -> cols 19..21, y 0..20 -> rows 0..20)
    # carries no boundary pixels, so interior walls keep full weight.
    assert int(boundary[0:21, 19:22].sum()) == 0
    assert int(walls.sum()) > int(boundary.sum())


# ---------------------------------------------------------------------------
# Time score
# ---------------------------------------------------------------------------


def test_compute_time_score_half_limit_is_full_marks() -> None:
    manager = _scorer_manager()  # time_best_ratio = 0.5
    # elapsed <= half of limit -> 1.0
    assert manager.compute_time_score(0.0, 1000.0).score == pytest.approx(1.0)
    assert manager.compute_time_score(500.0, 1000.0).score == pytest.approx(1.0)


def test_compute_time_score_linear_between_half_and_limit() -> None:
    manager = _scorer_manager()
    result = manager.compute_time_score(750.0, 1000.0)
    assert result.score == pytest.approx(0.5)  # (1000-750)/(1000-500)
    assert result.breakdown["time_best"] == pytest.approx(500.0)
    assert result.breakdown["time_limit"] == pytest.approx(1000.0)


def test_compute_time_score_at_limit_is_zero() -> None:
    manager = _scorer_manager()
    assert manager.compute_time_score(1000.0, 1000.0).score == pytest.approx(0.0)
    assert manager.compute_time_score(1500.0, 1000.0).score == pytest.approx(0.0)


def test_phase1_includes_time_score_from_timesteps() -> None:
    manager = _scorer_manager()
    truth = _wall_test_grid()
    phase1 = manager.compute_phase1_score(
        truth,
        None,
        elapsed_walltime=0.0,
        max_walltime_limit=180.0,
        score_exploration_trajectory=42.0,
        placed_count=0,
        expected_count=2,
        has_crashed=False,
        bomb_positions=None,
        elapsed_timestep=2500,
        max_timestep_limit=5000,
    )
    # 2500 is exactly half of 5000 -> full time score.
    assert phase1.score_time == pytest.approx(100.0)
    assert phase1.time_breakdown["time_metric"] == "timestep"
    # map=0, bomb=0, wall=0 -> round = 0.1 * 100.
    assert phase1.round_score == pytest.approx(0.1 * 100.0)


def test_phase1_time_score_zero_on_crash_and_missing_inputs() -> None:
    manager = _scorer_manager()
    truth = _wall_test_grid()

    crashed = manager.compute_phase1_score(
        truth, None,
        elapsed_walltime=0.0, max_walltime_limit=10.0,
        score_exploration_trajectory=0.0,
        placed_count=0, expected_count=1,
        has_crashed=True,
        elapsed_timestep=500, max_timestep_limit=1000,
    )
    assert crashed.score_time == 0.0

    missing = manager.compute_phase1_score(
        truth, None,
        elapsed_walltime=0.0, max_walltime_limit=10.0,
        score_exploration_trajectory=0.0,
        placed_count=0, expected_count=1,
        has_crashed=False,
    )
    assert missing.score_time == 0.0
    assert "missing" in missing.time_breakdown.get("error", "")


def test_phase1_time_score_walltime_metric() -> None:
    manager = PlaceScoreManager(PlaceScoreConfig(time_metric="walltime"))
    truth = _wall_test_grid()
    phase1 = manager.compute_phase1_score(
        truth,
        None,
        elapsed_walltime=90.0,
        max_walltime_limit=180.0,
        score_exploration_trajectory=0.0,
        placed_count=0,
        expected_count=1,
        has_crashed=False,
        elapsed_timestep=None,
        max_timestep_limit=None,
    )
    # 90s is exactly half of 180s -> full time score.
    assert phase1.score_time == pytest.approx(100.0)
    assert phase1.time_breakdown["time_metric"] == "walltime"

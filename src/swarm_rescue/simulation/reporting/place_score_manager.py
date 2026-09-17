"""Red-team (place mode) scoring: map submission + wall destruction + time + reference-blue bomb placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from swarm_rescue.simulation.utils.definitions import CollisionTypes
from swarm_rescue.simulation.utils.utils import clamp

from swarm_rescue.tools.map_exploration_scoring import (
    MapExplorationScorer,
    breakdown_to_dict,
    scorer_for_walltime,
)

BinaryGrid = np.ndarray


def rasterize_world_polygons(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    width: int,
    height: int,
) -> BinaryGrid:
    """Rasterize world-space (x, y, y-up) wall polygons onto an (H, W) binary grid.

    Uses the same world -> grid mapping as the exploration map
    (col = x + width/2, row = height/2 - y). Pixels covered by any polygon
    become 1.
    """
    grid = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(
            [
                [float(x) + width / 2.0, height / 2.0 - float(y)]
                for x, y in polygon
            ],
            dtype=np.float32,
        )
        cv2.fillPoly(grid, [np.round(pts).astype(np.int32)], 1)
    return grid


def _polygon_is_boundary(
    polygon: Sequence[Tuple[float, float]],
    width: int,
    height: int,
    border_tolerance: float = 1.0,
    min_span_ratio: float = 0.5,
) -> bool:
    """True when a wall polygon runs *along* one of the arena border sides.

    The boundary frame is at a fixed position: ``ClosedPlayground``
    (``_walls_creation``) builds the four border walls so their *outer* edge
    coincides exactly with the arena border (``x = +/- width/2``,
    ``y = +/- height/2``); being flush with the border, a frame wall also runs
    along it for the whole side.

    Merely *touching* the border is not enough to be a frame wall: interior
    walls may abut the frame end-on (a T-junction) and then meet it across
    nothing but their own ~6 px thickness. Map01 has exactly two such
    partitions -- ``walls_01.py`` wall 4 (``x = -115``, reaching y = +400 =
    height/2) and wall 5 (``x = 285``, reaching y = -399) -- and the old
    "bounding box touches the border" test classified *both* as boundary,
    leaving the map with zero interior wall pixels and discounting two real
    interior partitions to half weight.

    A polygon is a boundary wall therefore when, for one of the borders it
    touches, the span of that contact *along* the border covers at least
    ``min_span_ratio`` of that side: the four frame walls span all of it,
    while a partition spans only its thickness (6 px / 1250 px = 0.5% on
    Map01). Walls sitting just inside the frame (e.g. Map04's long inner ring,
    whose outer edge lies one thickness inside the border) do not touch the
    border at all and stay interior -- no per-map heuristics needed.
    """
    half_w = width / 2.0
    half_h = height / 2.0
    # Contact with each border line: (span coordinates of the vertices lying
    # on/near that line, length of the side being touched).
    sides = (
        ([p[1] for p in polygon if p[0] <= -half_w + border_tolerance], height),
        ([p[1] for p in polygon if p[0] >= half_w - border_tolerance], height),
        ([p[0] for p in polygon if p[1] <= -half_h + border_tolerance], width),
        ([p[0] for p in polygon if p[1] >= half_h - border_tolerance], width),
    )
    for contact_span, side_length in sides:
        if len(contact_span) < 2:
            continue
        if max(contact_span) - min(contact_span) >= min_span_ratio * side_length:
            return True
    return False


def collect_wall_geometry(
    playground: Any,
) -> Tuple[List[List[Tuple[float, float]]], List[bool]]:
    """World-space wall polygons and their boundary-frame flags.

    Walls are identified by their collision type (``CollisionTypes.WALL``:
    ``NormalWall`` / ``NormalBox`` / ``DisappearingWall`` / ``DisappearingBox``
    and the ``ClosedPlayground`` border walls). Other playground elements --
    disposal center, return area, disabler zones -- carry different collision
    types and are therefore excluded.

    ``pymunk`` shapes store vertices in local (entity-centred) coordinates, so
    they are transformed into world coordinates via the entity body before
    being returned.

    Returns ``(polygons, is_boundary)`` where ``is_boundary[i]`` marks the map
    boundary-frame walls (see ``_polygon_is_boundary``).
    """
    width, height = 0, 0
    size = getattr(playground, "size", None)
    if size:
        width, height = int(size[0]), int(size[1])

    polygons: List[List[Tuple[float, float]]] = []
    boundary_flags: List[bool] = []
    for element in getattr(playground, "elements", None) or []:
        ct = getattr(element, "_collision_type", None)
        if ct != CollisionTypes.WALL:
            continue
        body = getattr(element, "_pm_body", None)
        for shape in getattr(element, "_pm_shapes", None) or []:
            get_vertices = getattr(shape, "get_vertices", None)
            if get_vertices is None:  # non-poly shapes (e.g. circles)
                continue
            vertices = list(get_vertices())
            if len(vertices) < 3:
                continue
            if body is not None:
                vertices = [body.local_to_world(v) for v in vertices]
            poly = [(float(v[0]), float(v[1])) for v in vertices]
            if width and height:
                boundary_flags.append(
                    _polygon_is_boundary(poly, width, height)
                )
            else:
                boundary_flags.append(False)
            polygons.append(poly)
    return polygons, boundary_flags


def collect_wall_polygons(playground: Any) -> List[List[Tuple[float, float]]]:
    """World-space vertex polygons of every true wall entity in a playground.

    See :func:`collect_wall_geometry` for details; this returns only the
    polygon list for callers that do not need the boundary-frame flags.
    """
    polygons, _ = collect_wall_geometry(playground)
    return polygons


def walls_only_grid(
    playground: Any,
    truth_grid: Optional[BinaryGrid] = None,
) -> Optional[BinaryGrid]:
    """
    Binary grid containing only the map's real walls.

    Built from the playground's actual wall entities and their collision
    geometry -- exactly "the real positions and ranges of the map's wall
    elements" -- rather than from a thresholded render of every playground
    element. This excludes non-wall fills (disposal center, return area,
    disabler zones, unreachable blobs) from the wall-destruction score.

    Falls back to ``truth_grid`` (the full obstacle render) when no wall
    entities are found, so callers always get a usable grid.
    """
    size = getattr(playground, "size", None)
    if not size:
        return truth_grid
    width, height = int(size[0]), int(size[1])
    polygons = collect_wall_polygons(playground)
    if not polygons:
        return truth_grid
    return rasterize_world_polygons(polygons, width, height)


def walls_and_boundary_grid(
    playground: Any,
    truth_grid: Optional[BinaryGrid] = None,
) -> Tuple[Optional[BinaryGrid], Optional[BinaryGrid]]:
    """
    Walls-only grid plus the boundary-frame grid derived from wall geometry.

    The boundary grid marks exactly the map's boundary-frame walls (the four
    ``ClosedPlayground`` border walls); interior walls are excluded even if
    they are connected to the frame. See :func:`walls_only_grid` and
    :func:`_polygon_is_boundary`.

    Returns ``(walls_grid, boundary_grid)``; when no wall entities are found
    both fall back to ``truth_grid`` / its border-connected mask.
    """
    size = getattr(playground, "size", None)
    grid_w, grid_h = None, None
    if size:
        grid_w, grid_h = int(size[0]), int(size[1])

    polygons, boundary_flags = (
        collect_wall_geometry(playground)
        if grid_w and grid_h
        else ([], [])
    )
    if not polygons or grid_w is None:
        return truth_grid, truth_grid

    walls = rasterize_world_polygons(polygons, grid_w, grid_h)
    boundary_polys = [
        p for p, is_b in zip(polygons, boundary_flags) if is_b
    ]
    boundary = (
        rasterize_world_polygons(boundary_polys, grid_w, grid_h)
        if boundary_polys
        else np.zeros((grid_h, grid_w), dtype=np.uint8)
    )
    return walls, boundary

_TIME_METRICS = ("timestep", "walltime")


@dataclass(frozen=True)
class PlaceScoreConfig:
    """Weights and options for red-team scoring.

    Defaults follow the competition rubric (weights sum to 1.0):
      - exploration 50% (the exploration term is pure mapping accuracy --
        ``MapExplorationScorer.ScoreConfig`` no longer blends an internal time
        component; ``w_time=0``),
      - bomb placement difficulty 20%,
      - wall destruction 20%,
      - time 10% (dedicated top-level time score, timestep-based).
    """

    w_exploration: float = 0.5
    w_bomb: float = 0.2
    w_wall: float = 0.2
    w_time: float = 0.1
    crash_difficulty: float = 100.0
    use_median_for_bomb: bool = False
    min_successful_references: Optional[int] = None

    # Wall-destruction scoring.
    #: Blast radius of every bomb, as a fraction of the map's *shorter* grid
    #: dimension (px = ``blast_radius * min(grid_h, grid_w)``). A map-relative
    #: radius keeps the blast on the same relative scale on maps of different
    #: sizes (e.g. 0.12 -> ~135 px on Map04, ~96 px on Map01).
    blast_radius: float = 0.12
    #: Weight applied to map-boundary wall pixels (0..1); interior walls weigh 1.
    boundary_wall_weight: float = 0.5

    # Time scoring.
    #: Clock used: ``"timestep"`` (deterministic, machine-independent, matches
    #: the blue-team time score) or ``"walltime"`` (wall-clock seconds).
    time_metric: str = "timestep"
    #: Fraction of the time limit at which the time score saturates to 1.0
    #: (e.g. 0.5 -> finishing within half the limit earns full marks).
    time_best_ratio: float = 0.5

    def validate(self) -> None:
        for name, value in (
            ("w_exploration", self.w_exploration),
            ("w_bomb", self.w_bomb),
            ("w_wall", self.w_wall),
            ("w_time", self.w_time),
        ):
            if value < 0:
                raise ValueError(f"place score weight {name} must be non-negative")
        total = self.w_exploration + self.w_bomb + self.w_wall + self.w_time
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"place score weights must sum to 1.0, got {total}")
        if not (0.0 <= self.boundary_wall_weight <= 1.0):
            raise ValueError(
                "boundary_wall_weight must be within [0, 1], got "
                f"{self.boundary_wall_weight}"
            )
        if not (0.0 < self.blast_radius <= 1.0):
            raise ValueError(
                "blast_radius (fraction of the map's shorter side) must be "
                f"within (0, 1], got {self.blast_radius}"
            )
        if self.time_metric not in _TIME_METRICS:
            raise ValueError(
                f"time_metric must be one of {_TIME_METRICS}, got "
                f"{self.time_metric!r}"
            )
        if not (0.0 < self.time_best_ratio <= 1.0):
            raise ValueError(
                f"time_best_ratio must be within (0, 1], got "
                f"{self.time_best_ratio}"
            )


@dataclass
class MapScoreResult:
    score: float
    breakdown: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class BombScoreResult:
    score: float
    breakdown: Dict[str, Any]
    sufficient_references: bool = True


@dataclass
class WallScoreResult:
    """Wall-destruction score from the union of bomb blast circles.

    ``score`` is the team's weighted destroyed-wall pixel value divided by the
    map's **theoretical optimum** (a deterministic greedy cover with the same
    number of bombs and radius), in [0, 1]. Full marks therefore means "as
    good as the best achievable layout for this map", not "every wall gone".
    """

    score: float
    breakdown: Dict[str, Any]


@dataclass
class TimeScoreResult:
    """Time score for a place round.

    ``score`` is in [0, 1]: 1.0 when the round finishes within
    ``time_best_ratio`` of the limit, 0.0 at (or past) the limit, linear in
    between.
    """

    score: float
    breakdown: Dict[str, Any]


@dataclass
class PlacePhase1Result:
    """Scores available immediately after a place round ends."""

    score_map: float
    score_bomb: float
    round_score: float
    score_status: str
    map_breakdown: Dict[str, Any]
    bomb_breakdown: Dict[str, Any]
    score_exploration_trajectory: float
    score_wall: float = 0.0
    wall_breakdown: Dict[str, Any] = field(default_factory=dict)
    score_time: float = 0.0
    time_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceBlueRun:
    zip_name: str
    blue_score: float
    difficulty: float
    has_crashed: bool
    percent_disposed: float = 0.0
    full_disposal_timestep: int = 0
    success: bool = True
    error: Optional[str] = None


def _normalize_pred_grid(grid: Union[np.ndarray, None]) -> Optional[BinaryGrid]:
    if grid is None:
        return None
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return None
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    unique_vals = set(np.unique(arr).tolist())
    if unique_vals.issubset({0, 1}):
        return arr.astype(np.uint8)
    if unique_vals.issubset({0, 255}):
        return (arr == 255).astype(np.uint8)
    return (arr != 0).astype(np.uint8)


def difficulty_from_blue_score(blue_score: float) -> float:
    """
    Single source of truth for the blue-score -> bomb-difficulty conversion.

    ``difficulty = min(100, max(0, (100 - blue_score) * 2))``:
    a blue score of 50 or lower saturates the difficulty at 100, and the
    remaining 50–100 range is stretched to keep reference-blue gaps
    discriminative. Shared by every scoring path (compute_bomb_score,
    score-bombs CLI, competition calibration, re-place) so all flows convert
    identically.
    """
    return min(100.0, max(0.0, 100.0 - float(blue_score)) * 2.0)


def placement_completion(placed_count: int, expected_count: Optional[int]) -> float:
    """Placement completion in [0, 1]; mirrors compute_bomb_score semantics."""
    if expected_count is None or expected_count <= 0:
        return 1.0 if placed_count <= 0 else 0.0
    return min(1.0, max(0.0, placed_count / expected_count))


def bomb_score_from_difficulty(
    difficulties: Sequence[float],
    *,
    placed_count: int,
    expected_count: Optional[int],
    use_median: bool = False,
) -> float:
    """
    Bomb score = aggregate difficulty (mean or median) x placement completion.

    Single source of truth shared by the score-bombs CLI (via
    PlaceScoreManager.compute_bomb_score) and the competition-pipeline
    calibration path, so all evaluation flows score identically.
    """
    difficulties = [max(0.0, float(d)) for d in difficulties]
    if not difficulties:
        return 0.0
    aggregate = float(median(difficulties)) if use_median else (
        sum(difficulties) / len(difficulties)
    )
    completion = placement_completion(placed_count, expected_count)
    return aggregate * completion


class PlaceScoreManager:
    """Compute red-team scores from map submission and reference-blue runs."""

    def __init__(self, config: Optional[PlaceScoreConfig] = None):
        self.config = config or PlaceScoreConfig()
        self.config.validate()

    def compute_map_score(
        self,
        truth_grid: BinaryGrid,
        pred_grid: Union[BinaryGrid, np.ndarray, None],
        *,
        elapsed_walltime: float,
        max_walltime_limit: float,
        has_crashed: bool = False,
        map_scorer: Optional[MapExplorationScorer] = None,
    ) -> MapScoreResult:
        """Score the submitted exploration map against the walls-only truth.

        ``truth_grid`` is the walls-only grid built from the map's real wall
        entities (the same source as the wall-destruction score): the disposal
        center, return area and disabler zones carry no wall collision type and
        are not part of the truth, so drawing them earns no credit and instead
        pays the false-prediction charge.
        """
        if has_crashed:
            return MapScoreResult(
                score=0.0,
                breakdown={},
                error="simulation crashed",
            )

        pred = _normalize_pred_grid(pred_grid)
        if pred is None:
            return MapScoreResult(
                score=0.0,
                breakdown={},
                error="no exploration map submitted",
            )

        if truth_grid.shape != pred.shape:
            return MapScoreResult(
                score=0.0,
                breakdown={},
                error=(
                    f"grid shape mismatch: truth={truth_grid.shape}, "
                    f"pred={pred.shape}"
                ),
            )

        scorer = map_scorer or scorer_for_walltime(max_walltime_limit)
        breakdown = scorer.score(truth_grid, pred, float(elapsed_walltime))
        return MapScoreResult(
            score=float(breakdown.score),
            breakdown=breakdown_to_dict(breakdown),
        )

    def compute_bomb_score(
        self,
        reference_runs: Sequence[ReferenceBlueRun],
        *,
        placed_count: int,
        expected_count: int,
    ) -> BombScoreResult:
        if expected_count <= 0:
            completion = 1.0 if placed_count <= 0 else 0.0
        else:
            completion = min(1.0, max(0.0, placed_count / expected_count))

        difficulties: List[float] = []
        ref_details: List[Dict[str, Any]] = []
        failed_details: List[Dict[str, Any]] = []
        for run in reference_runs:
            is_valid = run.success and not run.has_crashed
            if is_valid:
                difficulty = difficulty_from_blue_score(run.blue_score)
                difficulties.append(difficulty)
                ref_details.append({
                    "zip": run.zip_name,
                    "blue_score": run.blue_score,
                    "difficulty": difficulty,
                    "has_crashed": run.has_crashed,
                    "percent_disposed": run.percent_disposed,
                    "full_disposal_timestep": run.full_disposal_timestep,
                })
            else:
                failed_details.append({
                    "zip": run.zip_name,
                    "blue_score": run.blue_score,
                    "has_crashed": run.has_crashed,
                    "success": run.success,
                    "error": run.error,
                })

        min_required = self.config.min_successful_references
        if min_required is None:
            min_required = len(reference_runs)
        sufficient = len(difficulties) >= min_required and min_required > 0

        if not difficulties or not sufficient:
            aggregate = 0.0
            score = 0.0
        elif self.config.use_median_for_bomb:
            aggregate = float(median(difficulties))
            score = completion * aggregate
        else:
            aggregate = float(sum(difficulties) / len(difficulties))
            score = completion * aggregate

        return BombScoreResult(
            score=score,
            sufficient_references=sufficient,
            breakdown={
                "placement_completion": completion,
                "placed_count": placed_count,
                "expected_count": expected_count,
                "aggregate_difficulty": aggregate,
                "reference_blues": ref_details,
                "failed_reference_blues": failed_details,
                "successful_reference_count": len(difficulties),
                "required_reference_count": min_required,
                "sufficient_references": sufficient,
            },
        )

    def blast_radius_px(self, grid_shape: Optional[Tuple[int, int]] = None) -> int:
        """Pixel blast radius from the config ratio (fraction of the shorter side)."""
        if grid_shape is None:
            return 0
        h, w = int(grid_shape[0]), int(grid_shape[1])
        return int(round(float(self.config.blast_radius) * min(h, w)))

    @classmethod
    def wall_theoretical_optimum(
        cls,
        truth_grid: BinaryGrid,
        *,
        n_bombs: int,
        blast_radius: float,
        boundary_weight: float = 0.5,
        boundary_mask: Optional[BinaryGrid] = None,
    ) -> float:
        """
        Deterministic greedy optimum for the wall-destruction score.

        The wall score is normalized by "theoretical optimum": the best
        weighted-wall destruction achievable on this map with ``n_bombs``
        bombs of the given (pixel) radius. This method computes a stable,
        reproducible surrogate by greedily placing each bomb at the position
        that adds the most *new* weighted wall coverage (interior walls weigh
        1.0, boundary walls ``boundary_weight``), exactly mirroring the union
        semantics of :meth:`compute_wall_score`.

        The greedy cover is deterministic (np.argmax tie-breaks to the first
        cell) and only places bombs on free (non-wall) cells -- a drone drops a
        bomb at its own position, so centers inside a wall are unreachable --
        so the reference is identical for every team on the same map and round
        config ("full-marks reference" does not vary per sample). It is a
        lower bound on the true optimum, so a perfect real placement reaches
        (or clamps at) full marks without the score being trivial.

        Args:
            truth_grid: Binary wall grid (1 = wall).
            n_bombs: Number of bombs available this round (the reference count).
            blast_radius: Blast radius in px (1:1 with the grid).
            boundary_weight: Weight applied to boundary-frame wall pixels.
            boundary_mask: Explicit boundary-frame wall grid; falls back to
                border-connected wall components when omitted.

        Returns:
            The greedy optimum as a weighted-coverage value (same units as
            ``weighted_destroyed``); 0.0 when no bombs or no walls.
        """
        truth = _normalize_pred_grid(truth_grid)
        if truth is None or truth.size == 0 or int(n_bombs) <= 0:
            return 0.0
        truth = (truth > 0).astype(np.uint8)
        height, width = truth.shape

        if boundary_mask is None:
            boundary = cls._boundary_wall_mask(truth)
        else:
            norm = _normalize_pred_grid(boundary_mask)
            boundary = (norm > 0).astype(np.uint8) if norm is not None else np.zeros_like(truth)

        values = np.where(boundary > 0, float(boundary_weight), 1.0).astype(np.float64)
        values *= truth

        r = max(1, int(round(float(blast_radius))))
        radius_sq = float(r * r)
        rng = r
        yy, xx = np.mgrid[-rng:rng + 1, -rng:rng + 1]
        disk = (
            (yy.astype(np.float64) ** 2 + xx.astype(np.float64) ** 2) <= radius_sq + 0.5
        ).astype(np.float64)

        covered = np.zeros((height, width), dtype=bool)
        # Bomb centers must lie on free space: a drone places a bomb at its own
        # position, which is never inside a wall. Restricting the greedy
        # candidates to non-wall cells keeps the full-marks reference
        # achievable in the simulator.
        allowed = (truth == 0)
        optimum = 0.0
        for _ in range(max(0, int(n_bombs))):
            remaining = values.copy()
            remaining[covered] = 0.0
            coverage = cv2.filter2D(
                remaining, -1, disk, borderType=cv2.BORDER_CONSTANT
            )
            pick = coverage.copy()
            pick[~allowed] = -1.0
            best = int(np.argmax(pick))
            row, col = divmod(best, width)
            added = float(coverage[row, col])
            if added <= 0.0:
                break
            optimum += added
            r0 = max(0, row - rng)
            r1 = min(height, row + rng + 1)
            c0 = max(0, col - rng)
            c1 = min(width, col + rng + 1)
            sub = covered[r0:r1, c0:c1]
            kernel_slice = disk[
                (r0 - (row - rng)):(r0 - (row - rng)) + (r1 - r0),
                (c0 - (col - rng)):(c0 - (col - rng)) + (c1 - c0),
            ]
            sub[:] |= (kernel_slice > 0.5)
        return float(optimum)

    def compute_wall_score(
        self,
        bomb_positions: Sequence[Sequence[float]],
        truth_grid: BinaryGrid,
        *,
        blast_radius: Optional[float] = None,
        boundary_wall_weight: Optional[float] = None,
        boundary_mask: Optional[BinaryGrid] = None,
        expected_n_bombs: Optional[int] = None,
    ) -> WallScoreResult:
        """
        Wall-destruction score from the union of circular bomb blast areas.

        Every bomb (world coordinates) gets a circular blast radius on the
        obstacle grid. A wall pixel counts as destroyed if it lies within reach
        of at least one bomb (areas are unioned, so overlapping circles are not
        double-counted).

        Boundary walls (the map's boundary-frame walls, discounted by
        ``boundary_wall_weight``) are distinguished from interior walls via an
        explicit ``boundary_mask``; when none is passed, border-connected wall
        components are used as a fallback. The returned score is the team's
        weighted destroyed wall divided by the map's theoretical optimum
        (:meth:`wall_theoretical_optimum` with the same radius and
        ``expected_n_bombs`` bombs), clamped to [0, 1]. Passing
        ``expected_n_bombs`` (the round's fixed bomb count) keeps the
        full-marks reference identical for every team on the same map; without
        it the number of placed bombs is used.

        Args:
            bomb_positions: Sequence of (x, y) world coordinates.
            truth_grid: Binary grid with 1 = wall (in production, the
                walls-only grid; see ``walls_and_boundary_grid``).
            blast_radius: Override for ``config.blast_radius`` (fraction of the
                map's shorter side, in (0, 1]).
            boundary_wall_weight: Override for ``config.boundary_wall_weight``.
            boundary_mask: Binary grid marking the boundary-frame walls;
                defaults to border-connected wall components.
            expected_n_bombs: Fixed bomb count of the round used to build the
                "theoretical optimum" full-marks reference; the number of
                placed bombs is used when omitted.

        Returns:
            WallScoreResult with ``score`` in [0, 1] and a detailed breakdown.
        """
        radius_ratio = (
            self.config.blast_radius if blast_radius is None else float(blast_radius)
        )
        boundary_weight = (
            self.config.boundary_wall_weight
            if boundary_wall_weight is None
            else float(boundary_wall_weight)
        )

        truth = _normalize_pred_grid(truth_grid)
        if truth is None or truth.size == 0:
            return WallScoreResult(0.0, {"error": "empty or invalid truth grid"})
        truth = (truth > 0).astype(np.uint8)

        height, width = truth.shape
        radius_px = int(round(radius_ratio * min(height, width)))
        centers = [
            (row, col)
            for row, col in (
                self._world_to_grid(x, y, width, height)
                for x, y in bomb_positions or []
            )
            if row is not None
        ]
        total_bombs = len(list(bomb_positions or []))
        if not centers:
            return WallScoreResult(
                0.0,
                {
                    "n_bombs": total_bombs,
                    "reference_n_bombs": total_bombs,
                    "blast_radius": radius_px,
                    "blast_radius_ratio": radius_ratio,
                    "boundary_wall_weight": boundary_weight,
                },
            )

        _, blast, destroyed, boundary_mask = self.wall_destruction_masks(
            truth_grid, bomb_positions, radius_px, boundary_mask=boundary_mask
        )

        destroyed_boundary = int((destroyed & (boundary_mask > 0)).sum())
        destroyed_interior = int((destroyed & (boundary_mask == 0)).sum())
        total_boundary = int((truth & (boundary_mask > 0)).sum())
        total_interior = int((truth & (boundary_mask == 0)).sum())

        weighted_hit = destroyed_interior + boundary_weight * destroyed_boundary
        weighted_total = total_interior + boundary_weight * total_boundary

        n_ref = (
            int(expected_n_bombs)
            if expected_n_bombs is not None
            else len(centers)
        )
        # The full-marks reference: best weighted coverage achievable on this
        # exact map with n_ref bombs and this radius (deterministic greedy).
        optimum = self.wall_theoretical_optimum(
            truth,
            n_bombs=max(0, n_ref),
            blast_radius=float(radius_px),
            boundary_weight=boundary_weight,
            boundary_mask=boundary_mask,
        )
        coverage_fraction = (
            weighted_hit / weighted_total if weighted_total > 0 else 0.0
        )
        score = (weighted_hit / optimum) if optimum > 0 else 0.0
        score = float(np.clip(score, 0.0, 1.0))

        breakdown: Dict[str, Any] = {
            "score_fraction": score,
            "coverage_fraction": float(coverage_fraction),
            "blast_radius": radius_px,
            "blast_radius_ratio": radius_ratio,
            "boundary_wall_weight": boundary_weight,
            "n_bombs": len(centers),
            "reference_n_bombs": n_ref,
            "blast_pixels": int(blast.sum()),
            "destroyed_wall_pixels": int(destroyed.sum()),
            "destroyed_boundary_pixels": destroyed_boundary,
            "destroyed_interior_pixels": destroyed_interior,
            "total_boundary_pixels": total_boundary,
            "total_interior_pixels": total_interior,
            "total_wall_pixels": int(truth.sum()),
            "covered_wall_pixels": int(destroyed.sum()),
            "weighted_destroyed": float(weighted_hit),
            "weighted_total": float(weighted_total),
            "theoretical_optimum": float(optimum),
            "theoretical_optimum_fraction": (
                float(optimum / weighted_total) if weighted_total > 0 else 0.0
            ),
        }
        return WallScoreResult(score=score, breakdown=breakdown)

    @classmethod
    def wall_destruction_masks(
        cls,
        truth_grid: BinaryGrid,
        bomb_positions: Sequence[Sequence[float]],
        blast_radius: float,
        boundary_mask: Optional[BinaryGrid] = None,
    ) -> Tuple[BinaryGrid, BinaryGrid, BinaryGrid, BinaryGrid]:
        """
        Compute the blast-coverage masks used by both scoring and rendering.

        Single source of truth for the wall-destruction geometry so the
        visualization (``bombs_io.save_wall_destruction_image``) always shows
        exactly what ``compute_wall_score`` scored. Bomb positions are world
        coordinates; grids use the same world -> (row, col) mapping as the
        exploration map.

        Returns ``(truth, blast, destroyed, boundary)`` -- all uint8 {0,1}
        grids of the same shape:
          - truth:      normalized wall grid (1 = wall);
          - blast:      pixels within ``blast_radius`` of any bomb center;
          - destroyed:  wall pixels covered by the blast union;
          - boundary:   boundary-frame wall pixels (explicit mask, or
                        border-connected wall components as a fallback).
        """
        truth = _normalize_pred_grid(truth_grid)
        if truth is None or truth.size == 0:
            raise ValueError("empty or invalid truth grid")
        truth = (truth > 0).astype(np.uint8)

        height, width = truth.shape
        centers = [
            (row, col)
            for row, col in (
                cls._world_to_grid(x, y, width, height)
                for x, y in bomb_positions or []
            )
            if row is not None
        ]

        blast = np.zeros((height, width), dtype=np.uint8)
        if centers:
            center_mask = np.zeros((height, width), dtype=np.uint8)
            rows = np.array([r for r, _ in centers], dtype=np.int32)
            cols = np.array([c for _, c in centers], dtype=np.int32)
            center_mask[rows, cols] = 1
            dist = cv2.distanceTransform(
                1 - center_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            blast = (dist <= float(blast_radius)).astype(np.uint8)

        destroyed = (truth & blast).astype(np.uint8)
        if boundary_mask is None:
            boundary = cls._boundary_wall_mask(truth)
        else:
            norm = _normalize_pred_grid(boundary_mask)
            boundary = (
                (norm > 0).astype(np.uint8) if norm is not None
                else np.zeros_like(truth)
            )
        return truth, blast, destroyed, boundary

    @staticmethod
    def _world_to_grid(
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int]]:
        """World (centered at 0,0, y up) -> grid (row, col), or None if out of bounds."""
        col = int(round(float(x) + width / 2.0))
        row = int(round(height / 2.0 - float(y)))
        if 0 <= col < width and 0 <= row < height:
            return row, col
        return None

    @staticmethod
    def _boundary_wall_mask(truth: BinaryGrid) -> np.ndarray:
        """Obstacle pixels belonging to wall components touching the image border.

        Coarse last-resort fallback for callers that pass no ``boundary_mask``:
        since it works on connected components it cannot tell a partition
        abutting the frame (Map01's two walls) from the frame itself, and marks
        both as boundary. Production paths (``launcher`` / batch eval) always
        pass the exact polygon-derived mask from ``walls_and_boundary_grid``,
        so this is only a degraded safety net.
        """
        truth = (truth > 0).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(truth)
        if n_labels <= 1:
            return np.zeros_like(truth, dtype=np.uint8)

        border = np.zeros_like(truth, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        boundary = np.zeros_like(truth, dtype=np.uint8)
        border_labels = np.unique(labels[border & (labels > 0)])
        for lab in border_labels:
            boundary[labels == lab] = 1
        return boundary

    def compute_time_score(
        self,
        elapsed: float,
        time_limit: float,
    ) -> TimeScoreResult:
        """
        Time score: shorter is better, saturating to 1.0 within a "reasonable"
        time range and dropping to 0 at the limit.

        ``s_time = clamp((time_limit - elapsed) / (time_limit - time_best),
        0, 1)`` where ``time_best = time_best_ratio * time_limit``; i.e.
        finishing within half the limit (default) earns full marks, reaching
        the limit earns zero.

        Args:
            elapsed: Elapsed time (in the configured ``time_metric`` units).
            time_limit: The time limit (same units as ``elapsed``).

        Returns:
            TimeScoreResult with ``score`` in [0, 1] and a breakdown.
        """
        time_limit = float(time_limit)
        elapsed = float(elapsed)
        best = float(self.config.time_best_ratio) * time_limit

        if time_limit <= 0:
            score = 0.0
        elif time_limit == best:
            score = 1.0 if elapsed <= best else 0.0
        else:
            score = (time_limit - elapsed) / (time_limit - best)
            score = float(clamp(score, 0.0, 1.0))

        return TimeScoreResult(
            score=score,
            breakdown={
                "score_fraction": score,
                "time_metric": self.config.time_metric,
                "time_limit": time_limit,
                "time_best": best,
                "time_best_ratio": self.config.time_best_ratio,
                "elapsed": elapsed,
            },
        )

    def _compute_round_time_score(
        self,
        *,
        has_crashed: bool,
        elapsed_timestep: Optional[int],
        max_timestep_limit: Optional[int],
        elapsed_walltime: Optional[float],
        max_walltime_limit: Optional[float],
    ) -> TimeScoreResult:
        """Pick the clock (timestep/walltime) and compute the round time score."""
        if has_crashed:
            return TimeScoreResult(0.0, {"error": "simulation crashed"})

        if self.config.time_metric == "walltime":
            if elapsed_walltime is None or max_walltime_limit is None:
                return TimeScoreResult(0.0, {"error": "walltime inputs missing"})
            return self.compute_time_score(elapsed_walltime, max_walltime_limit)

        if elapsed_timestep is None or max_timestep_limit is None:
            return TimeScoreResult(0.0, {"error": "timestep inputs missing"})
        return self.compute_time_score(float(elapsed_timestep), float(max_timestep_limit))

    def compute_round_score(
        self,
        score_map: float,
        score_bomb: float,
        score_wall: float = 0.0,
        score_time: float = 0.0,
    ) -> float:
        cfg = self.config
        return (cfg.w_exploration * score_map
                + cfg.w_bomb * score_bomb
                + cfg.w_wall * score_wall
                + cfg.w_time * score_time)

    def compute_phase1_score(
        self,
        truth_grid: BinaryGrid,
        pred_grid: Union[BinaryGrid, np.ndarray, None],
        *,
        elapsed_walltime: float,
        max_walltime_limit: float,
        score_exploration_trajectory: float,
        placed_count: int,
        expected_count: int,
        has_crashed: bool = False,
        bomb_positions: Optional[Sequence[Sequence[float]]] = None,
        elapsed_timestep: Optional[int] = None,
        max_timestep_limit: Optional[int] = None,
        wall_grid: Optional[BinaryGrid] = None,
        boundary_grid: Optional[BinaryGrid] = None,
    ) -> PlacePhase1Result:
        """Compute the partial place score (map + wall + time) for one round.

        ``truth_grid`` is the exploration-map truth (in production the
        walls-only grid from the map's real wall entities; see
        :meth:`compute_map_score`). ``wall_grid`` / ``boundary_grid`` are the
        walls-only and boundary-frame grids for the wall-destruction score and
        fall back to ``truth_grid`` when not provided.
        """
        map_result = self.compute_map_score(
            truth_grid,
            pred_grid,
            elapsed_walltime=elapsed_walltime,
            max_walltime_limit=max_walltime_limit,
            has_crashed=has_crashed,
        )
        bomb_breakdown: Dict[str, Any] = {
            "placement_completion": (
                min(1.0, placed_count / expected_count)
                if expected_count > 0
                else 1.0
            ),
            "placed_count": placed_count,
            "expected_count": expected_count,
            "reference_blues": [],
            "pending": True,
        }

        if has_crashed:
            wall_result = WallScoreResult(
                0.0,
                {"error": "simulation crashed"},
            )
        elif bomb_positions is None:
            wall_result = WallScoreResult(
                0.0,
                {"error": "no bomb positions provided"},
            )
        else:
            # Wall scoring uses the walls-only grid (real wall entities) when
            # provided; otherwise it falls back to the full obstacle render.
            wall_source = wall_grid if wall_grid is not None else truth_grid
            wall_result = self.compute_wall_score(
                bomb_positions,
                wall_source,
                boundary_mask=boundary_grid,
                expected_n_bombs=expected_count,
            )

        time_result = self._compute_round_time_score(
            has_crashed=has_crashed,
            elapsed_timestep=elapsed_timestep,
            max_timestep_limit=max_timestep_limit,
            elapsed_walltime=elapsed_walltime,
            max_walltime_limit=max_walltime_limit,
        )

        score_wall = 100.0 * wall_result.score
        score_time = 100.0 * time_result.score
        return PlacePhase1Result(
            score_map=map_result.score,
            score_bomb=0.0,
            round_score=self.compute_round_score(
                map_result.score, 0.0, score_wall, score_time
            ),
            score_status="partial",
            map_breakdown=map_result.breakdown,
            bomb_breakdown=bomb_breakdown,
            score_exploration_trajectory=score_exploration_trajectory,
            score_wall=score_wall,
            wall_breakdown=wall_result.breakdown,
            score_time=score_time,
            time_breakdown=time_result.breakdown,
        )

    def merge_final_score(
        self,
        phase1: PlacePhase1Result,
        bomb_result: BombScoreResult,
    ) -> PlacePhase1Result:
        round_score = self.compute_round_score(
            phase1.score_map,
            bomb_result.score,
            phase1.score_wall,
            phase1.score_time,
        )
        bomb_breakdown = dict(bomb_result.breakdown)
        bomb_breakdown["pending"] = False
        return PlacePhase1Result(
            score_map=phase1.score_map,
            score_bomb=bomb_result.score,
            round_score=round_score,
            score_status="final",
            map_breakdown=phase1.map_breakdown,
            bomb_breakdown=bomb_breakdown,
            score_exploration_trajectory=phase1.score_exploration_trajectory,
            score_wall=phase1.score_wall,
            wall_breakdown=phase1.wall_breakdown,
            score_time=phase1.score_time,
            time_breakdown=phase1.time_breakdown,
        )


def place_details_from_phase1(
    phase1: PlacePhase1Result,
    *,
    score_exploration_trajectory: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build score.json details dict for a place round."""
    trajectory = (
        score_exploration_trajectory
        if score_exploration_trajectory is not None
        else phase1.score_exploration_trajectory
    )
    details: Dict[str, Any] = {
        "score_map": phase1.score_map,
        "score_bomb": phase1.score_bomb,
        "score_wall": phase1.score_wall,
        "score_time": phase1.score_time,
        "map_breakdown": phase1.map_breakdown,
        "bomb_breakdown": phase1.bomb_breakdown,
        "wall_breakdown": phase1.wall_breakdown,
        "time_breakdown": phase1.time_breakdown,
        "score_status": phase1.score_status,
        "diagnostics": {
            "score_exploration_trajectory": trajectory,
            "informational_only": True,
        },
        # Kept for backward compatibility with existing reports/tools.
        "score_exploration_trajectory": trajectory,
    }
    if extra:
        details.update(extra)
    return details


def load_place_score_config(data: Optional[Dict[str, Any]]) -> PlaceScoreConfig:
    """Load PlaceScoreConfig from a YAML ``place_scoring`` section.

    Backward compatibility: configs that only set the legacy two weights
    (``w_exploration`` + ``w_bomb``) and no ``w_wall`` / ``w_time`` keep their
    old 100% split (wall and time scoring are off). New configs opt into the
    wall/time scores by setting ``w_wall`` / ``w_time``.
    """
    if not data:
        return PlaceScoreConfig()
    min_refs = data.get("min_successful_references")
    if "w_wall" not in data and "w_time" not in data:
        # Legacy two-weight config: keep historical behavior (no wall/time score).
        return PlaceScoreConfig(
            w_exploration=float(data.get("w_exploration", 0.5)),
            w_bomb=float(data.get("w_bomb", 0.5)),
            w_wall=0.0,
            w_time=0.0,
            crash_difficulty=float(data.get("crash_difficulty", 100.0)),
            use_median_for_bomb=bool(data.get("use_median", False)),
            min_successful_references=(
                int(min_refs) if min_refs is not None else None
            ),
        )
    return PlaceScoreConfig(
        w_exploration=float(data.get("w_exploration", 0.5)),
        w_bomb=float(data.get("w_bomb", 0.2)),
        w_wall=float(data.get("w_wall", 0.2)),
        w_time=float(data.get("w_time", 0.1)),
        crash_difficulty=float(data.get("crash_difficulty", 100.0)),
        use_median_for_bomb=bool(data.get("use_median", False)),
        min_successful_references=(
            int(min_refs) if min_refs is not None else None
        ),
        blast_radius=float(data.get("blast_radius", 100.0)),
        boundary_wall_weight=float(data.get("boundary_wall_weight", 0.5)),
        time_metric=str(data.get("time_metric", "timestep")),
        time_best_ratio=float(data.get("time_best_ratio", 0.5)),
    )

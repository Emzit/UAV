import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from swarm_rescue.simulation.utils.utils import clamp
from swarm_rescue.tools.map_wall_editor import load_elements, render_elements_to_image


BinaryGrid = np.ndarray


def _to_binary_grid(grid: Union[List[List[Any]], np.ndarray], *, name: str) -> BinaryGrid:
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got ndim={arr.ndim}.")

    # Normalize common representations to {0,1}
    # - allow bool
    # - allow 0/1 ints
    # - allow 0/255 ints (common for images)
    if arr.dtype == np.bool_:
        out = arr.astype(np.uint8)
    else:
        unique_vals = set(np.unique(arr).tolist())
        if unique_vals.issubset({0, 1}):
            out = arr.astype(np.uint8)
        elif unique_vals.issubset({0, 255}):
            out = (arr == 255).astype(np.uint8)
        else:
            # Best-effort: treat non-zero as 1
            # but keep strictness for floats that look like 0/1.
            out = (arr != 0).astype(np.uint8)

    return out


@dataclass(frozen=True)
class ScoreConfig:
    """Parameters for the exploration-map scorer.

    The model is "hit rate earns credit, stray prediction pays a charge". Credit
    and penalty are measured by two independent passes over disjoint pixel sets,
    so no single error is ever billed twice.
    """

    # Truth obstacles are split by thickness, measured as the distance from an
    # obstacle pixel to the nearest free pixel. Pixels within wall_width of free
    # space are the observable wall surface; anything deeper is bulk interior.
    #
    # The truth grid is the walls-only grid built from the map's real wall
    # entities (line/rect walls, boxes and the boundary frame): non-wall fills
    # such as the disposal center, return area and disabler zones are not part
    # of it. Solid rectangle obstacles (NormalBox) are rasterized as filled
    # rectangles whose effective thickness is their short side (48-188 px on the
    # shipped maps), so pixels deeper than wall_width inside them land in the
    # interior region; standard 6 px line walls have no deep interior at all.
    # The region weights below therefore favour the surface, which is all a
    # drone can actually observe.
    #
    # The default wall_width follows the engine: walls are 6 px thick (ColorWall
    # and NormalWall both default to wall_thickness=6, and no map overrides it),
    # so the deepest pixel of a standard wall sits exactly 3 px from free space.
    # 4.0 covers a whole standard wall with 1 px of slack for the extra
    # thickness where walls meet at corners and junctions.
    wall_width: float = 4.0

    # Weights for the two hit rates. Must sum to 1.
    #
    # These act on hit *rates*, which are already fractions, so they mean exactly
    # what they say. The default scores the wall surface only; the deep interior
    # carries zero weight.
    w_wall: float = 1.0
    w_interior: float = 0.0

    # Predicted pixels within this distance of a real obstacle are free: they
    # earn no credit but pay no charge either. Sized to absorb localization
    # noise -- GPS sigma is 5 px/axis and compass sigma 4 deg, which puts the 2D
    # RMS error near 10 px at 100 px lidar range.
    dead_zone: float = 10.0

    # Charge schedule beyond the dead zone. The charge starts at penalty_step and
    # grows by penalty_step every penalty_repeat pixels, saturating at
    # penalty_cap. Defaults give 0.1 0.1 0.2 0.2 0.3 0.3 ... reaching the cap
    # 20 px past the dead zone.
    #
    # Saturation matters: an unbounded charge would let a handful of far-flung
    # pixels dominate, and would make the achievable range depend on how much the
    # team drew, which breaks comparability across maps.
    penalty_step: float = 0.1
    penalty_repeat: int = 2
    penalty_cap: float = 1.0

    # How hard the accumulated charge bites, relative to credit.
    #
    # The charge is normalized by the wall-surface pixel count, which is roughly
    # 3.5x smaller than the full obstacle count on Map01, so a raw
    # coefficient of 1 would sink every real submission. 0.2 keeps scores in a
    # usable range while leaving honest drawing optimal: measured on real
    # submissions, neither dilating nor eroding gains more than ~0.5%.
    penalty_weight: float = 0.2

    # Time score: s_time = clamp((time_limit - t) / (time_limit - time_best), 0, 1)
    time_best: float = 0.0
    time_limit: float = 1440.0

    # Final blend. Must sum to 1.
    #
    # The time component is disabled (w_time = 0): in the competition the red
    # team's time bonus is scored by a dedicated top-level time score
    # (PlaceScoreConfig.w_time), so the exploration term is pure mapping
    # accuracy. ``s_time`` is still computed and reported for diagnostics but
    # carries zero weight.
    w_map: float = 1.0
    w_time: float = 0.0

    # Map walls JSON -> obstacle grid.
    # Rendered walls are anti-aliased, so we threshold by grayscale intensity.
    wall_threshold: int = 128

    def validate(self) -> None:
        if self.w_wall < 0 or self.w_interior < 0:
            raise ValueError("region weights must be non-negative")
        region_sum = self.w_wall + self.w_interior
        if not np.isclose(region_sum, 1.0, atol=1e-6):
            raise ValueError(f"w_wall + w_interior must sum to 1.0, got {region_sum}")
        if self.w_map < 0 or self.w_time < 0:
            raise ValueError("final weights must be non-negative")
        final_sum = self.w_map + self.w_time
        if not np.isclose(final_sum, 1.0, atol=1e-6):
            raise ValueError(f"w_map + w_time must sum to 1.0, got {final_sum}")
        if self.wall_width < 0:
            raise ValueError("wall_width must be >= 0")
        if self.dead_zone < 0:
            raise ValueError("dead_zone must be >= 0")
        if self.penalty_step <= 0:
            raise ValueError("penalty_step must be > 0")
        if self.penalty_repeat < 1:
            raise ValueError("penalty_repeat must be >= 1")
        if self.penalty_cap <= 0:
            raise ValueError("penalty_cap must be > 0")
        if self.penalty_weight < 0:
            raise ValueError("penalty_weight must be >= 0")
        if self.time_limit < self.time_best:
            raise ValueError("time_limit must be >= time_best")
        if self.wall_threshold < 0 or self.wall_threshold > 255:
            raise ValueError("wall_threshold must be in [0,255]")


@dataclass(frozen=True)
class RegionHitRate:
    """How much of one truth region the prediction covered."""

    pixels: int
    hits: int
    available: bool

    @property
    def misses(self) -> int:
        return self.pixels - self.hits

    @property
    def rate(self) -> float:
        if self.pixels <= 0:
            return 0.0
        return self.hits / self.pixels


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float

    s_time: float
    s_map: float

    # Credit side: hit rates over the two truth regions.
    credit: float
    wall: RegionHitRate
    interior: RegionHitRate

    # Penalty side: charge over predicted pixels, relative to truth size.
    penalty: float
    predicted_pixels: int
    exact_pixels: int
    dead_zone_pixels: int
    charged_pixels: int
    capped_pixels: int
    total_charge: float
    truth_pixels: int

    #: Denominator of the penalty: the wall-surface pixel count.
    surface_pixels: int


class MapExplorationScorer:
    """Score a submitted exploration map against ground-truth obstacles.

    Credit comes from hit rates over ground-truth obstacles ("did the team find
    this wall?"). Penalty comes from predicted obstacles that sit too far from
    any real wall ("does this wall actually exist?"). The two passes walk
    disjoint pixel sets, so an error is charged once and only once.

    The penalty is normalized by the *wall surface* pixel count, not the whole
    obstacle layer and not the prediction size:

    * Not the prediction size, because that denominator grows with whatever the
      team drew, letting a bulk-filled map dilute its own average charge.
    * Not the whole obstacle layer, because a stray prediction is a mistake about
      where the wall face is, and deep interior pixels say nothing about that.
      Using the full count would make the identical mistake cheaper on maps that
      happen to contain large unreachable blobs -- 2.2x cheaper on a controlled
      pair of maps differing only by one such blob.

    ``penalty_weight`` then scales the result. The wall surface is several times
    smaller than the obstacle layer, so an unscaled charge would sink every real
    submission. 0.2 was chosen by sweeping it against real submissions: it keeps
    scores in a usable range, zeroes dense noise and bulk fills, and leaves
    honest drawing optimal -- neither dilating nor eroding gains more than ~0.5%.
    """

    def __init__(self, config: Optional[ScoreConfig] = None) -> None:
        self.config = config or ScoreConfig()
        self.config.validate()

    def score(self, truth_grid: BinaryGrid, pred_grid: BinaryGrid,
              time_seconds: float) -> ScoreBreakdown:
        truth = _to_binary_grid(truth_grid, name="truth_grid")
        pred = _to_binary_grid(pred_grid, name="pred_grid")

        if truth.shape != pred.shape:
            raise ValueError(f"shape mismatch: truth={truth.shape} pred={pred.shape}")

        truth = (truth > 0).astype(np.uint8)
        pred = (pred > 0).astype(np.uint8)

        wall_region, interior_region = self.split_regions(truth)
        to_pred = self.distance_field(pred)
        to_truth = self.distance_field(truth)

        wall = self._hit_rate(wall_region, to_pred)
        interior = self._hit_rate(interior_region, to_pred)
        credit = self._combine_credit(wall, interior)

        charges = self._charges(pred, to_truth)
        truth_pixels = int(truth.sum())
        # Normalize by the wall surface, not the whole obstacle layer. A stray
        # prediction is a mistake about where the wall *face* is, and deep
        # interior pixels carry no information about that. Dividing by the full
        # obstacle count would dilute the same mistake on maps that happen to
        # contain large unreachable blobs -- measured at 2.2x on a controlled
        # pair of maps differing only by one such blob.
        surface_pixels = int(wall_region.sum())
        penalty = 0.0
        if surface_pixels > 0:
            penalty = (self.config.penalty_weight
                       * charges["total"] / surface_pixels)

        s_map = float(np.clip(credit - penalty, 0.0, 1.0))
        s_time = self._score_time(time_seconds)

        cfg = self.config
        score = 100.0 * (cfg.w_map * s_map + cfg.w_time * s_time)

        return ScoreBreakdown(
            score=float(np.clip(score, 0.0, 100.0)),
            s_time=float(s_time),
            s_map=s_map,
            credit=float(credit),
            wall=wall,
            interior=interior,
            penalty=float(penalty),
            predicted_pixels=int(charges["pixels"]),
            exact_pixels=int(charges["exact"]),
            dead_zone_pixels=int(charges["dead"]),
            charged_pixels=int(charges["charged"]),
            capped_pixels=int(charges["capped"]),
            total_charge=float(charges["total"]),
            truth_pixels=truth_pixels,
            surface_pixels=surface_pixels,
        )

    def penalty_curve(self, distance: np.ndarray) -> np.ndarray:
        """Charge for each predicted pixel at ``distance`` from the nearest wall.

        Zero inside the dead zone, then a staircase rising by ``penalty_step``
        every ``penalty_repeat`` pixels, saturating at ``penalty_cap``.
        """
        cfg = self.config
        over = distance - cfg.dead_zone
        steps = np.ceil(np.maximum(over, 0.0) / cfg.penalty_repeat)
        charge = np.where(over > 0.0,
                          np.minimum(cfg.penalty_step * steps, cfg.penalty_cap),
                          0.0)
        return np.maximum(charge, 0.0)

    @staticmethod
    def distance_field(mask: BinaryGrid) -> np.ndarray:
        """Euclidean distance from every pixel to the nearest nonzero of ``mask``.

        DIST_MASK_PRECISE gives the true euclidean distance rather than the 3x3
        chamfer approximation, so the score does not depend on wall orientation.
        """
        if not mask.any():
            return np.full(mask.shape, np.float32(1e9), np.float32)
        inverted = 1 - (mask > 0).astype(np.uint8)
        return cv2.distanceTransform(inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    @staticmethod
    def obstacle_depth(truth: BinaryGrid) -> np.ndarray:
        """Thickness of every obstacle pixel: distance to the nearest free space.

        Free space includes the region *outside* the grid. The arena border is
        an ordinary wall whose outer face is the grid edge, and its thickness
        has to be measured against the space beyond that edge exactly as an
        interior wall is measured against the space around it -- otherwise the
        border's outer pixels are scored as if they sat deep inside a solid
        blob. On a 6 px border wall that misreads depth 7, 6, 5, 4, 3, 2, 1
        where the truth is 1, 2, 3, 4, 3, 2, 1, and half the wall is demoted to
        the zero-weight interior.

        Padding the mask with a single free pixel is enough: the outside is
        unbounded, so the nearest outside pixel to an edge pixel is always the
        one directly beyond it, and ``DIST_MASK_PRECISE`` then reports the true
        euclidean distance to whichever side is closer.
        """
        truth = (truth > 0).astype(np.uint8)
        padded = np.zeros((truth.shape[0] + 2, truth.shape[1] + 2), np.uint8)
        padded[1:-1, 1:-1] = truth
        depth = cv2.distanceTransform(
            padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        return depth[1:-1, 1:-1]

    def split_regions(self, truth: BinaryGrid) -> Tuple[BinaryGrid, BinaryGrid]:
        """Split truth obstacles into wall surface and deep interior.

        Thickness comes from :meth:`obstacle_depth`, which counts the space
        outside the grid as free, so a wall that touches the arena border is
        split on the same terms as one in the middle of the map.
        """
        truth = (truth > 0).astype(np.uint8)
        depth = self.obstacle_depth(truth)
        obstacle = truth > 0
        wall = (obstacle & (depth <= self.config.wall_width)).astype(np.uint8)
        interior = (obstacle & (depth > self.config.wall_width)).astype(np.uint8)
        return wall, interior

    @staticmethod
    def _hit_rate(region: BinaryGrid, to_pred: np.ndarray) -> RegionHitRate:
        pixels = int(region.sum())
        if pixels == 0:
            return RegionHitRate(pixels=0, hits=0, available=False)
        hits = int((to_pred[region > 0] == 0.0).sum())
        return RegionHitRate(pixels=pixels, hits=hits, available=True)

    def _combine_credit(self, wall: RegionHitRate,
                        interior: RegionHitRate) -> float:
        """Weighted mean of the available hit rates.

        Renormalizing over whichever regions exist keeps a thin-walled map from
        being penalized for having no deep interior at all.
        """
        cfg = self.config
        weighted = ((wall, cfg.w_wall), (interior, cfg.w_interior))
        active = sum(w for region, w in weighted if region.available)
        if active <= 0:
            return 0.0
        return sum(region.rate * w for region, w in weighted
                   if region.available) / active

    def _charges(self, pred: BinaryGrid,
                 to_truth: np.ndarray) -> Dict[str, float]:
        pixels = int(pred.sum())
        if pixels == 0:
            return {"pixels": 0, "exact": 0, "dead": 0, "charged": 0,
                    "capped": 0, "total": 0.0}

        d = to_truth[pred > 0]
        charge = self.penalty_curve(d)
        cap = self.config.penalty_cap
        return {
            "pixels": pixels,
            "exact": int((d == 0.0).sum()),
            "dead": int(((d > 0.0) & (d <= self.config.dead_zone)).sum()),
            "charged": int((d > self.config.dead_zone).sum()),
            "capped": int((charge >= cap - 1e-12).sum()),
            "total": float(charge.sum()),
        }

    def _score_time(self, t: float) -> float:
        cfg = self.config
        if cfg.time_limit == cfg.time_best:
            return 1.0 if t <= cfg.time_best else 0.0
        s_time = (cfg.time_limit - t) / (cfg.time_limit - cfg.time_best)
        return float(clamp(s_time, 0.0, 1.0))


def scorer_for_walltime(max_walltime_limit: float) -> MapExplorationScorer:
    """Build the exploration scorer used by the evaluation pipeline.

    This is the single construction point shared by every consumer of the
    exploration score -- the recorded score in ``PlaceScoreManager`` and the
    ``*_exploration_diff.png`` picture written by the launcher. Building them
    from one place is what keeps the picture honest: change a term here and both
    the number and the rendering move together.
    """
    return MapExplorationScorer(
        ScoreConfig(time_limit=float(max_walltime_limit))
    )


def load_walls_json_as_obstacle_grid(
    walls_json_path: Union[str, Path],
    *,
    wall_threshold: int = 128,
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
) -> BinaryGrid:
    """
    Convert a walls JSON (like `map_data/map_01.json` from this repo) into a binary obstacle grid.
    """
    walls_json_path = Path(walls_json_path)
    with walls_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    canvas = data.get("canvas", {}) or {}
    json_w = int(canvas.get("width", 1113))
    json_h = int(canvas.get("height", 750))
    width = int(canvas_width) if canvas_width is not None else json_w
    height = int(canvas_height) if canvas_height is not None else json_h

    elements = load_elements(data, width=width, height=height)
    image = render_elements_to_image(width=width, height=height, elements=elements)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    obstacle = (gray < wall_threshold).astype(np.uint8)
    return obstacle


def load_prediction_grid(pred_json_path: Union[str, Path], *, grid_key_candidates: Optional[List[str]] = None) -> BinaryGrid:
    """
    Load the player's predicted exploration grid (binary matrix) from JSON.

    Supported formats:
    - A raw 2D array: [[0,1,...], ...]
    - An object containing a matrix under one of:
      - default keys: "grid", "matrix", "prediction"
    """
    pred_json_path = Path(pred_json_path)
    with pred_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if grid_key_candidates is None:
        grid_key_candidates = ["grid", "matrix", "prediction"]

    if isinstance(data, list):
        return _to_binary_grid(data, name="pred_grid")

    if isinstance(data, dict):
        for k in grid_key_candidates:
            if k in data:
                return _to_binary_grid(data[k], name=f"pred_grid[{k!r}]")

    raise ValueError(f"Unrecognized prediction JSON format: {pred_json_path}")

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    defaults = ScoreConfig()
    parser = argparse.ArgumentParser(
        description="Map exploration scoring tool (walls-json -> binary grid).")
    parser.add_argument("--truth-walls-json", required=True,
                        help="Ground-truth walls JSON (e.g. map_data/map_01.json).")
    parser.add_argument("--pred-grid-json", required=True,
                        help="Player predicted binary grid JSON.")
    parser.add_argument("--time-seconds", required=True, type=float,
                        help="Exploration time in seconds.")

    parser.add_argument("--wall-width", type=float, default=defaults.wall_width,
                        help="Obstacle pixels within this distance of free space "
                             "count as wall surface.")
    parser.add_argument("--w-wall", type=float, default=defaults.w_wall,
                        help="Weight of the wall-surface hit rate.")
    parser.add_argument("--w-interior", type=float, default=defaults.w_interior,
                        help="Weight of the deep-interior hit rate.")
    parser.add_argument("--dead-zone", type=float, default=defaults.dead_zone,
                        help="Predicted pixels this close to a real wall are free.")
    parser.add_argument("--penalty-step", type=float,
                        default=defaults.penalty_step,
                        help="Charge increment per step beyond the dead zone.")
    parser.add_argument("--penalty-repeat", type=int,
                        default=defaults.penalty_repeat,
                        help="Pixels per charge step (2 gives 0.1 0.1 0.2 0.2 ...).")
    parser.add_argument("--penalty-cap", type=float, default=defaults.penalty_cap,
                        help="Maximum charge for a single pixel.")
    parser.add_argument("--penalty-weight", type=float,
                        default=defaults.penalty_weight,
                        help="Scale of the charge relative to credit.")

    parser.add_argument("--time-best", type=float, default=defaults.time_best)
    parser.add_argument("--time-limit", type=float, default=defaults.time_limit)
    parser.add_argument("--w-map", type=float, default=defaults.w_map)
    parser.add_argument("--w-time", type=float, default=defaults.w_time)

    parser.add_argument("--wall-threshold", type=int,
                        default=defaults.wall_threshold)
    parser.add_argument("--canvas-width", type=int, default=None)
    parser.add_argument("--canvas-height", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional output path.")
    return parser.parse_args(argv)


def breakdown_to_dict(result: ScoreBreakdown) -> Dict[str, Any]:
    """Flatten a breakdown for JSON reporting (used by score.json and the CLI)."""
    return {
        "s_time": result.s_time,
        "s_map": result.s_map,
        "credit": result.credit,
        "penalty": result.penalty,
        "wall_pixels": result.wall.pixels,
        "wall_hits": result.wall.hits,
        "wall_hit_rate": result.wall.rate,
        "wall_available": result.wall.available,
        "interior_pixels": result.interior.pixels,
        "interior_hits": result.interior.hits,
        "interior_hit_rate": result.interior.rate,
        "interior_available": result.interior.available,
        "predicted_pixels": result.predicted_pixels,
        "exact_pixels": result.exact_pixels,
        "dead_zone_pixels": result.dead_zone_pixels,
        "charged_pixels": result.charged_pixels,
        "capped_pixels": result.capped_pixels,
        "total_charge": result.total_charge,
        "truth_pixels": result.truth_pixels,
        "surface_pixels": result.surface_pixels,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    config = ScoreConfig(
        wall_width=args.wall_width,
        w_wall=args.w_wall,
        w_interior=args.w_interior,
        dead_zone=args.dead_zone,
        penalty_step=args.penalty_step,
        penalty_repeat=args.penalty_repeat,
        penalty_cap=args.penalty_cap,
        penalty_weight=args.penalty_weight,
        time_best=args.time_best,
        time_limit=args.time_limit,
        w_map=args.w_map,
        w_time=args.w_time,
        wall_threshold=args.wall_threshold,
    )
    scorer = MapExplorationScorer(config=config)

    truth_grid = load_walls_json_as_obstacle_grid(
        args.truth_walls_json,
        wall_threshold=args.wall_threshold,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
    )
    pred_grid = load_prediction_grid(args.pred_grid_json)

    result = scorer.score(truth_grid, pred_grid, args.time_seconds)
    payload: Dict[str, Any] = {"score": result.score, **breakdown_to_dict(result)}

    output_path = Path(args.output_json) if args.output_json else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

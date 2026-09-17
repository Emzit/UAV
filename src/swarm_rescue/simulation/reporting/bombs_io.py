import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from swarm_rescue.simulation.reporting.evaluation import EvalConfig
from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreManager
from swarm_rescue.simulation.reporting.team_info import TeamInfo
from swarm_rescue.tools.map_exploration_scoring import MapExplorationScorer


def _zones_config_to_list(zones_config) -> List[str]:
    if not zones_config:
        return []
    return [zone.name for zone in zones_config]


def load_bombs_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load bomb placement data from a JSON file.

    Expected format:
        {
          "map_name": "Map01",
          "zones_config": [],
          "bombs": [{"x": 0, "y": 0, "theta": 0, "path": [[1, 2]]}, ...]
        }
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid bombs file (expected object): {path}")
    bombs = data.get("bombs", data.get("positions"))
    if bombs is None:
        raise ValueError(f"Invalid bombs file (missing 'bombs' list): {path}")
    if not isinstance(bombs, list):
        raise ValueError(f"Invalid bombs file ('bombs' must be a list): {path}")
    return data


def bombs_from_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return normalized bomb entry dicts from loaded file data."""
    bombs = data.get("bombs", data.get("positions", []))
    result = []
    for entry in bombs:
        if isinstance(entry, dict):
            result.append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            result.append({
                "x": float(entry[0]),
                "y": float(entry[1]),
                "theta": float(entry[2]) if len(entry) > 2 else 0.0,
            })
        else:
            raise ValueError(f"Invalid bomb entry: {entry!r}")
    return result


def validate_bombs_file_for_eval(
    data: Dict[str, Any],
    eval_config: EvalConfig,
    *,
    strict_map_name: bool = False,
) -> None:
    """Warn or raise if file metadata does not match the current evaluation."""
    file_map = data.get("map_name")
    if file_map and file_map != eval_config.map_name:
        msg = (
            f"Bombs file map_name {file_map!r} does not match "
            f"eval map {eval_config.map_name!r}"
        )
        if strict_map_name:
            raise ValueError(msg)
        print(f"Warning: {msg}")

    file_zones = data.get("zones_config")
    if file_zones is not None:
        expected = _zones_config_to_list(eval_config.zones_config)
        if list(file_zones) != expected:
            print(
                f"Warning: bombs file zones_config {file_zones!r} "
                f"does not match eval zones {expected!r}"
            )


def bombs_to_serializable(the_map) -> List[Dict[str, Any]]:
    """Collect current bomb positions from a map instance."""
    entries = []
    bombs = getattr(the_map, "_bombs", None) or []
    for bomb in bombs:
        pos = bomb.true_position()
        entries.append({
            "x": float(pos[0]),
            "y": float(pos[1]),
            "theta": float(bomb.true_angle()),
        })
        if bomb.path.length() > 1:
            path_pts = []
            for i in range(bomb.path.length()):
                p = bomb.path.get(index=i).position
                path_pts.append([float(p[0]), float(p[1])])
            if len(path_pts) > 1:
                entries[-1]["path"] = path_pts
    return entries


def build_bombs_document(
    the_map,
    eval_config: EvalConfig,
) -> Dict[str, Any]:
    return {
        "map_name": eval_config.map_name,
        "zones_config": _zones_config_to_list(eval_config.zones_config),
        "bombs": bombs_to_serializable(the_map),
    }


def placed_bombs_output_path(
    result_path: Optional[str],
    team_info: TeamInfo,
    eval_config: EvalConfig,
    round_number: int,
) -> Optional[str]:
    if not result_path:
        return None
    num_round_str = str(round_number)
    filename = (
        f"team{team_info.team_number_str_padded}_"
        f"{eval_config.map_name}_"
        f"{eval_config.zones_name_for_filename}_"
        f"rd{num_round_str}_bombs.json"
    )
    return str(Path(result_path) / filename)


def save_placed_bombs(
    path: Union[str, Path],
    the_map,
    eval_config: EvalConfig,
) -> None:
    """Write bomb positions placed during a red-team round."""
    document = build_bombs_document(the_map, eval_config)
    save_bombs_document(path, document)


def save_bombs_document(
    path: Union[str, Path],
    document: Dict[str, Any],
) -> None:
    """Write a bombs JSON document to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
        f.write("\n")


def exploration_map_output_path(
    result_path: Optional[str],
    team_info: TeamInfo,
    eval_config: EvalConfig,
    round_number: int,
) -> Optional[str]:
    """
    Output path for a red-team exploration map JSON file.

    Naming mirrors the bombs output, but uses `_exploration.json` suffix.
    """
    if not result_path:
        return None
    num_round_str = str(round_number)
    filename = (
        f"team{team_info.team_number_str_padded}_"
        f"{eval_config.map_name}_"
        f"{eval_config.zones_name_for_filename}_"
        f"rd{num_round_str}_exploration.json"
    )
    return str(Path(result_path) / filename)


def _normalize_binary_grid(
    grid: Union[List[List[Any]], np.ndarray],
    *,
    name: str = "grid",
) -> np.ndarray:
    """
    Normalize a 2D matrix to a {0,1} uint8 grid.

    Accepts bool, {0,1} ints or {0,255} ints; any other values are treated
    as non-zero -> 1.
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D matrix, got ndim={arr.ndim} "
            f"(shape={arr.shape})"
        )

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)

    unique_vals = set(np.unique(arr).tolist())
    if unique_vals.issubset({0, 1}):
        return arr.astype(np.uint8)
    if unique_vals.issubset({0, 255}):
        return (arr == 255).astype(np.uint8)
    # Best-effort: treat non-zero as 1
    return (arr != 0).astype(np.uint8)


def save_exploration_map(
    path: Union[str, Path],
    grid: Union[List[List[Any]], np.ndarray],
    elapsed_walltime: float,
) -> None:
    """
    Save a submitted red-team exploration grid as JSON.

    The saved JSON format is compatible with `map_exploration_scoring.py`
    (`--pred-grid-json` accepts `{"grid": [[...], ...]}`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = _normalize_binary_grid(grid, name="grid")

    payload: Dict[str, Any] = {
        # Some evaluators may read this field directly.
        "elapsed_walltime": float(elapsed_walltime),
        # And some may expect the scorer's `--time-seconds` argument name.
        "time_seconds": float(elapsed_walltime),
        # Scoring tool expects this key.
        "grid": out.tolist(),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


# Layout constants for the exploration comparison image.
COMPARISON_HEADER_HEIGHT = 24
COMPARISON_SEPARATOR_WIDTH = 2
COMPARISON_LEGEND_HEIGHT = 104

# Plain obstacle rendering, used by the ground-truth and submission panels.
_COMPARISON_COLOR_WHITE = (255, 255, 255)
_COMPARISON_COLOR_OBSTACLE = (40, 40, 40)
_COMPARISON_COLOR_SEPARATOR = (128, 128, 128)
_COMPARISON_COLOR_HEADER_BG = (238, 238, 238)
_COMPARISON_COLOR_INK = (30, 30, 30)

# Merged panel: one colour per scoring outcome, mirroring MapExplorationScorer.
#
# Credit is green, and the exact shade is derived at render time from the
# region's coefficient in the *active* ScoreConfig (see ``_credit_colour``), not
# from a palette baked in here: brighter means the region carries more weight.
# The two names below are that ramp's endpoints, so a region priced at 1.0 comes
# out as ``_COMPARISON_COLOR_HIT_WALL`` -- which is what the default config
# (w_wall=1.0) does for the wall surface.
_COMPARISON_COLOR_HIT_CREDIT_LIGHT = (150, 215, 150)
_COMPARISON_COLOR_HIT_CREDIT_HEAVY = (60, 170, 60)
_COMPARISON_COLOR_HIT_WALL = _COMPARISON_COLOR_HIT_CREDIT_HEAVY

# Truth the active scorer gives zero weight: present in the map, worth nothing
# either way. One flat neutral grey for found and missed alike, because the
# score does not tell those two cases apart. Never green -- green means credit.
_COMPARISON_COLOR_NO_CREDIT = (175, 172, 168)

# Truth the team never covered: credit forfeited, but never charged. Blue-grey
# reads as "absent" rather than "wrong".
_COMPARISON_COLOR_MISS = (190, 150, 110)

# Predicted inside the dead zone: present, free of charge.
_COMPARISON_COLOR_DEAD_ZONE = (110, 200, 240)

# Charged predictions ramp pale -> deep red as the charge nears the cap.
_COMPARISON_COLOR_PENALTY_LIGHT = (205, 205, 255)
_COMPARISON_COLOR_PENALTY_HEAVY = (0, 0, 105)

# Correctly-empty space. Deliberately a faint tint rather than pure white, so
# "scores nothing" is not confused with "nothing to report here".
_COMPARISON_COLOR_CORRECT_EMPTY = (242, 238, 232)


def exploration_comparison_image_output_path(
    result_path: Optional[str],
    team_info: TeamInfo,
    eval_config: EvalConfig,
    round_number: int,
) -> Optional[str]:
    """
    Output path for the red-team exploration comparison image (PNG).

    Naming mirrors `exploration_map_output_path`, but uses the
    `_exploration_diff.png` suffix.
    """
    if not result_path:
        return None
    num_round_str = str(round_number)
    filename = (
        f"team{team_info.team_number_str_padded}_"
        f"{eval_config.map_name}_"
        f"{eval_config.zones_name_for_filename}_"
        f"rd{num_round_str}_exploration_diff.png"
    )
    return str(Path(result_path) / filename)


def _grid_to_panel(grid: np.ndarray) -> np.ndarray:
    """Binary grid -> BGR panel: white background, dark obstacles."""
    out = np.full((*grid.shape, 3), _COMPARISON_COLOR_WHITE, dtype=np.uint8)
    out[grid > 0] = _COMPARISON_COLOR_OBSTACLE
    return out


def _penalty_ramp(charge: np.ndarray, penalty_cap: float) -> np.ndarray:
    """Map charge magnitude in [0, cap] onto a pale-to-deep red ramp."""
    fraction = np.clip(charge / penalty_cap, 0.0, 1.0)[..., None]
    light = np.array(_COMPARISON_COLOR_PENALTY_LIGHT, np.float32)
    heavy = np.array(_COMPARISON_COLOR_PENALTY_HEAVY, np.float32)
    return (light + (heavy - light) * fraction).astype(np.uint8)


def _credit_colour(weight: float) -> Tuple[int, int, int]:
    """Credit green for a region of the given scoring weight.

    ``weight`` is read straight off the active ``ScoreConfig`` (the two region
    weights sum to 1), so the shade is a read-out of what a pixel is actually
    worth: the heavier the region, the brighter the green. A zero-weight region
    is not routed here at all -- it gets ``_COMPARISON_COLOR_NO_CREDIT``, so the
    picture cannot advertise credit the scorer is not paying.
    """
    fraction = float(np.clip(weight, 0.0, 1.0))
    light = np.array(_COMPARISON_COLOR_HIT_CREDIT_LIGHT, np.float32)
    heavy = np.array(_COMPARISON_COLOR_HIT_CREDIT_HEAVY, np.float32)
    return tuple(int(v) for v in (light + (heavy - light) * fraction))


def _diff_panel(truth: np.ndarray, pred: np.ndarray,
                scorer: "MapExplorationScorer") -> np.ndarray:
    """Both grids merged, each pixel coloured by what it contributes.

    Painted in order of increasing importance, so the layers that carry score
    stay visible:

    1. correctly-empty space (faint tint -- scores nothing, but that differs
       from "nothing here to report")
    2. charged predictions (red, graded by charge)
    3. predictions inside the dead zone (amber, free)
    4. truth the active scorer does not price (neutral grey -- score-neutral)
    5. missed truth in a priced region (blue-grey -- credit forfeited, charged
       nothing)
    6. earned credit (green, brighter for the heavier region)

    Which truth region is which comes from ``scorer.config`` itself, so the
    picture follows the same ``w_wall`` / ``w_interior`` the score was computed
    with. A region weighted at exactly 0 earns nothing and forfeits nothing, so
    its hit and miss cases are deliberately indistinguishable; when both region
    weights are 0 (not a valid config, but defensive) no green is painted.

    A pixel can fall in more than one category -- a prediction sitting on a real
    wall is both a hit and an exact match -- so later layers intentionally
    overwrite earlier ones.
    """
    truth_b = truth > 0
    pred_b = pred > 0
    out = np.full((*truth.shape, 3), _COMPARISON_COLOR_WHITE, dtype=np.uint8)

    out[~truth_b & ~pred_b] = _COMPARISON_COLOR_CORRECT_EMPTY

    config = scorer.config
    if pred_b.any():
        to_truth = scorer.distance_field(truth)
        distance = np.where(pred_b, to_truth, 0.0)
        charge = scorer.penalty_curve(distance)
        charged = pred_b & (distance > config.dead_zone)
        out[charged] = _penalty_ramp(charge, config.penalty_cap)[charged]
        out[pred_b & (distance > 0.0)
            & (distance <= config.dead_zone)] = _COMPARISON_COLOR_DEAD_ZONE

    to_pred = scorer.distance_field(pred)
    found = to_pred == 0.0
    wall, interior = scorer.split_regions(truth)
    region_weights = (
        (interior, config.w_interior),
        (wall, config.w_wall),
    )
    for region, weight in region_weights:
        active = region > 0
        if weight <= 0.0:
            out[active] = _COMPARISON_COLOR_NO_CREDIT
            continue
        out[active & ~found] = _COMPARISON_COLOR_MISS
        out[active & found] = _credit_colour(weight)

    return out


def _draw_comparison_legend(canvas: np.ndarray, y0: int,
                            scorer: "MapExplorationScorer",
                            result: Optional[Any]) -> None:
    """Colour key, charge ramp, and the arithmetic behind the score."""
    config = scorer.config
    canvas[y0:y0 + COMPARISON_LEGEND_HEIGHT, :] = _COMPARISON_COLOR_HEADER_BG
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas_width = canvas.shape[1]
    pad = 10
    swatch_w, swatch_h = 24, 12

    # Region swatches come straight from the active config, so the key can never
    # advertise credit for a region the scorer prices at zero.
    entries = []
    for name, weight in (("wall surface", config.w_wall),
                         ("interior", config.w_interior)):
        if weight > 0.0:
            entries.append((_credit_colour(weight),
                            f"credit: {name} found (w={weight:g})"))
        else:
            entries.append((_COMPARISON_COLOR_NO_CREDIT,
                            f"{name} w=0 (scores nothing)"))
    entries += [
        (_COMPARISON_COLOR_MISS, "missed truth (no credit, no charge)"),
        (_COMPARISON_COLOR_DEAD_ZONE,
         f"predicted within dead zone {config.dead_zone:g}px (free)"),
        (_COMPARISON_COLOR_CORRECT_EMPTY, "correctly empty (scores nothing)"),
    ]
    x = pad
    y = y0 + pad
    for colour, label in entries:
        # Narrow canvases (small maps, unit tests) cannot fit the whole key.
        if x + swatch_w >= canvas_width:
            break
        cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h), colour, -1)
        cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h),
                      _COMPARISON_COLOR_INK, 1)
        cv2.putText(canvas, label, (x + swatch_w + 5, y + swatch_h - 1),
                    font, 0.37, _COMPARISON_COLOR_INK, 1, cv2.LINE_AA)
        x += swatch_w + 10 + int(6.6 * len(label))

    y += swatch_h + 11
    ramp_w = min(180, max(0, canvas_width - 2 * pad))
    if ramp_w > 0:
        steps = np.linspace(0.0, config.penalty_cap, ramp_w, dtype=np.float32)
        strip = _penalty_ramp(steps, config.penalty_cap)[None, :, :]
        canvas[y:y + swatch_h, pad:pad + ramp_w] = strip.repeat(swatch_h, axis=0)
        cv2.rectangle(canvas, (pad, y), (pad + ramp_w, y + swatch_h),
                      _COMPARISON_COLOR_INK, 1)
    cap_at = config.dead_zone + config.penalty_repeat * (
        config.penalty_cap / config.penalty_step)
    cv2.putText(canvas,
                f"charge {config.penalty_step:g} -> {config.penalty_cap:g} per px"
                f" (darker = larger; step x{config.penalty_repeat} px,"
                f" capped at {cap_at:g}px)",
                (pad + ramp_w + 8, y + swatch_h - 1), font, 0.37,
                _COMPARISON_COLOR_INK, 1, cv2.LINE_AA)

    if result is None:
        return

    y += swatch_h + 15
    lines = [
        f"credit  = {config.w_wall:g} x {result.wall.hits}/{result.wall.pixels}"
        f" + {config.w_interior:g} x"
        f" {result.interior.hits}/{result.interior.pixels}"
        f"  =  {result.credit:.4f}",
        f"penalty = {config.penalty_weight:g} x {result.total_charge:.0f} charge"
        f" over {result.charged_pixels} px ({result.capped_pixels} at cap)"
        f" / {result.surface_pixels} wall-surface px  =  {result.penalty:.4f}",
        f"map     = clamp({result.credit:.4f} - {result.penalty:.4f})"
        f"  =  {result.s_map:.4f}",
    ]
    for line in lines:
        cv2.putText(canvas, line, (pad, y), font, 0.37,
                    _COMPARISON_COLOR_INK, 1, cv2.LINE_AA)
        y += 15


def save_exploration_comparison_image(
    path: Union[str, Path],
    truth_grid: Union[List[List[Any]], np.ndarray],
    pred_grid: Union[List[List[Any]], np.ndarray],
    *,
    scorer: Optional["MapExplorationScorer"] = None,
    elapsed_walltime: Optional[float] = None,
) -> None:
    """
    Save a three-panel comparison of ground truth and the submitted grid.

    Layout (left to right): ground truth | submission | merged score view.

    The merged panel colours every pixel by what it contributes to the map score:
    green for earned credit (brighter = the region carries more weight), blue-grey
    for missed truth in a priced region, neutral grey for truth the active scorer
    prices at zero, amber for predictions inside the free dead zone, and a
    pale-to-deep red ramp for charged predictions. Correctly-empty space gets a
    faint tint so it stays distinguishable from the panel background.

    Pass ``scorer`` to match the parameters actually used for scoring: the panel
    and its key read ``w_wall`` / ``w_interior`` off that scorer's config, so a
    region that stops earning credit stops being drawn green with it. Pass
    ``elapsed_walltime`` as well to print the score arithmetic in the legend.
    """
    truth = _normalize_binary_grid(truth_grid, name="truth_grid")
    pred = _normalize_binary_grid(pred_grid, name="pred_grid")

    if truth.shape != pred.shape:
        raise ValueError(
            f"shape mismatch: truth={truth.shape}, pred={pred.shape}"
        )

    if scorer is None:
        scorer = MapExplorationScorer()

    result = None
    if elapsed_walltime is not None:
        result = scorer.score(truth, pred, float(elapsed_walltime))

    h, w = truth.shape
    wall, interior = scorer.split_regions(truth)
    merged_label = ("score view: green=credit red=charged amber=free"
                    " grey=unscored")
    if result is not None:
        merged_label = (
            f"credit {result.credit:.4f} - penalty {result.penalty:.4f}"
            f" = map {result.s_map:.4f}"
        )
    panels = [
        (f"Ground Truth: {int(truth.sum())} obstacle px"
         f" (wall {int(wall.sum())}, interior {int(interior.sum())})",
         _grid_to_panel(truth)),
        (f"Submission: {int(pred.sum())} obstacle px", _grid_to_panel(pred)),
        (merged_label, _diff_panel(truth, pred, scorer)),
    ]

    total_width = 3 * w + 2 * COMPARISON_SEPARATOR_WIDTH
    total_height = COMPARISON_HEADER_HEIGHT + h + COMPARISON_LEGEND_HEIGHT
    canvas = np.full(
        (total_height, total_width, 3),
        _COMPARISON_COLOR_SEPARATOR,
        dtype=np.uint8,
    )
    canvas[:COMPARISON_HEADER_HEIGHT, :] = _COMPARISON_COLOR_HEADER_BG

    for idx, (label, panel) in enumerate(panels):
        x0 = idx * (w + COMPARISON_SEPARATOR_WIDTH)
        canvas[COMPARISON_HEADER_HEIGHT:COMPARISON_HEADER_HEIGHT + h,
               x0:x0 + w] = panel
        cv2.putText(
            canvas,
            label,
            (x0 + 4, COMPARISON_HEADER_HEIGHT - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            _COMPARISON_COLOR_INK,
            1,
            cv2.LINE_AA,
        )

    _draw_comparison_legend(canvas, COMPARISON_HEADER_HEIGHT + h, scorer, result)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise IOError(f"failed to write comparison image: {path}")


# ---------------------------------------------------------------------------
# Wall-destruction visualization (red-team bomb blast coverage)
#
# Rendered with the same canvas conventions as the exploration comparison
# image: a header band, map panels, then a legend that prints the exact
# arithmetic behind the wall-destruction score. The geometry comes from
# ``PlaceScoreManager.wall_destruction_masks``, which is the single source of
# truth shared with ``compute_wall_score`` -- the picture always shows what
# was actually scored.
# ---------------------------------------------------------------------------

WALL_DESTRUCTION_LEGEND_HEIGHT = 142

# Colors (BGR). Symmetric grays match the exploration panels.
_WALL_COLOR_INTACT_INTERIOR = (105, 105, 105)
_WALL_COLOR_INTACT_BOUNDARY = (60, 60, 60)
_WALL_COLOR_DESTROYED_INTERIOR = (70, 180, 70)
_WALL_COLOR_DESTROYED_BOUNDARY = (200, 90, 60)
_WALL_COLOR_BLAST_OVERLAY = (60, 60, 220)
_WALL_COLOR_BOMB = (30, 30, 235)
_WALL_COLOR_RING = (255, 255, 255)


def wall_destruction_image_output_path(
    result_path: Optional[str],
    team_info: TeamInfo,
    eval_config: EvalConfig,
    round_number: int,
) -> Optional[str]:
    """
    Output path for the red-team wall-destruction PNG.

    Naming mirrors ``exploration_comparison_image_output_path``, using the
    ``_wall_destruction.png`` suffix.
    """
    if not result_path:
        return None
    num_round_str = str(round_number)
    filename = (
        f"team{team_info.team_number_str_padded}_"
        f"{eval_config.map_name}_"
        f"{eval_config.zones_name_for_filename}_"
        f"rd{num_round_str}_wall_destruction.png"
    )
    return str(Path(result_path) / filename)


def _wall_destruction_panel(
    truth: np.ndarray,
    destroyed: np.ndarray,
    boundary: np.ndarray,
    blast: np.ndarray,
    centers: List[Tuple[int, int]],
) -> np.ndarray:
    """The single wall-destruction panel: walls by state, blast, bombs.

    White background; every wall pixel is coloured by its destruction state
    (green/blue = destroyed, grey = intact, with the map boundary frame
    distinguished by a darker grey). Bomb blast coverage tints the free space
    blue, and every bomb is marked with a red dot. Wall pixels keep their
    exact state colour so destroyed walls stay unambiguous.
    """
    out = np.full((*truth.shape, 3), _COMPARISON_COLOR_WHITE, dtype=np.uint8)
    wall = truth > 0
    destroyed_b = destroyed > 0
    boundary_b = boundary > 0
    out[wall & ~boundary_b & ~destroyed_b] = _WALL_COLOR_INTACT_INTERIOR
    out[wall & boundary_b & ~destroyed_b] = _WALL_COLOR_INTACT_BOUNDARY
    out[wall & ~boundary_b & destroyed_b] = _WALL_COLOR_DESTROYED_INTERIOR
    out[wall & boundary_b & destroyed_b] = _WALL_COLOR_DESTROYED_BOUNDARY

    # Bomb blast coverage fills the free space blue; walls keep their colours.
    if (blast > 0).any():
        out[(blast > 0) & ~wall] = _WALL_COLOR_BLAST_OVERLAY

    for row, col in centers:
        cv2.circle(out, (col, row), 4, _WALL_COLOR_BOMB, -1)
        cv2.circle(out, (col, row), 4, _WALL_COLOR_RING, 1)
    return out


def _draw_wall_destruction_legend(
    canvas: np.ndarray,
    y0: int,
    *,
    boundary_weight: float,
    breakdown: Dict[str, Any],
    score_percent: Optional[float],
) -> None:
    """Colour key and the arithmetic behind the wall-destruction score."""
    canvas[y0:y0 + WALL_DESTRUCTION_LEGEND_HEIGHT, :] = _COMPARISON_COLOR_HEADER_BG
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas_width = canvas.shape[1]
    pad = 10
    swatch_w, swatch_h = 24, 12

    entries = [
        (_WALL_COLOR_DESTROYED_INTERIOR, "destroyed interior wall"),
        (_WALL_COLOR_DESTROYED_BOUNDARY, "destroyed boundary wall"),
        (_WALL_COLOR_INTACT_INTERIOR, "intact interior wall"),
        (_WALL_COLOR_INTACT_BOUNDARY, "intact boundary wall"),
        (_WALL_COLOR_BLAST_OVERLAY, "bomb blast coverage"),
    ]
    x = pad
    y = y0 + pad
    for colour, label in entries:
        if x + swatch_w >= canvas_width:
            break
        cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h), colour, -1)
        cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h),
                      _COMPARISON_COLOR_INK, 1)
        cv2.putText(canvas, label, (x + swatch_w + 5, y + swatch_h - 1),
                    font, 0.37, _COMPARISON_COLOR_INK, 1, cv2.LINE_AA)
        x += swatch_w + 10 + int(6.6 * len(label))
        if x > canvas_width - pad:
            x = pad
            y += swatch_h + 4

    y += swatch_h + 12
    di = breakdown.get("destroyed_interior_pixels", 0)
    ti = breakdown.get("total_interior_pixels", 0)
    db = breakdown.get("destroyed_boundary_pixels", 0)
    tb = breakdown.get("total_boundary_pixels", 0)
    wd = breakdown.get("weighted_destroyed", 0.0)
    wt = breakdown.get("weighted_total", 0.0)
    opt = breakdown.get("theoretical_optimum", 0.0)
    opt_frac = breakdown.get("theoretical_optimum_fraction", 0.0)
    score = breakdown.get("score_fraction", 0.0)
    lines = [
        f"interior: {di}/{ti} px destroyed",
        f"boundary: {db}/{tb} px destroyed (weight {boundary_weight:g})",
        f"weighted reward = {wd:.1f}/{wt:.1f}  raw coverage = "
        f"{breakdown.get('coverage_fraction', 0.0)*100:.1f}%",
        f"theoretical optimum = {opt:.1f} ({opt_frac*100:.1f}% of walls)",
        f"score = {score*100:.1f}% of optimum",
    ]
    if score_percent is not None:
        lines[-1] += f"  ({score_percent:.1f}/100)"
    for line in lines:
        cv2.putText(canvas, line, (pad, y), font, 0.45,
                    _COMPARISON_COLOR_INK, 1, cv2.LINE_AA)
        y += 18


def save_wall_destruction_image(
    path: Union[str, Path],
    truth_grid: Union[List[List[Any]], np.ndarray],
    bomb_positions: Sequence[Sequence[float]],
    *,
    blast_radius: float,
    boundary_wall_weight: float,
    wall_breakdown: Optional[Dict[str, Any]] = None,
    score_wall_percent: Optional[float] = None,
    title: Optional[str] = None,
    boundary_mask: Optional[Union[List[List[Any]], np.ndarray]] = None,
) -> None:
    """
    Render one PNG showing how much of the map's walls the bombs destroy.

    A single panel draws the map, every bomb's circular blast coverage and
    the bomb markers, and colours every affected wall pixel by its
    destruction state (green/blue = destroyed, grey = intact, with the map
    boundary frame distinguished). The legend prints the weighted destroyed
    fraction and score, using the same geometry as ``compute_wall_score``.

    Args:
        path: Output PNG path.
        truth_grid: Binary ground-truth wall grid (1 = wall).
        bomb_positions: Sequence of (x, y) world coordinates.
        blast_radius: Bomb blast radius (px, 1:1 with the grid).
        boundary_wall_weight: Weight applied to boundary walls (0..1).
        wall_breakdown: Optional breakdown from compute_wall_score (used for
            the legend numbers).
        score_wall_percent: Optional wall score in [0, 100] for the legend.
        title: Optional header text (defaults to a generic description).
        boundary_mask: Optional binary grid marking the map boundary-frame
            walls; defaults to border-connected wall components.
    """
    truth = _normalize_binary_grid(truth_grid, name="truth_grid")
    truth, blast, destroyed, boundary = PlaceScoreManager.wall_destruction_masks(
        truth,
        bomb_positions,
        float(blast_radius),
        boundary_mask=(
            _normalize_binary_grid(boundary_mask, name="boundary_mask")
            if boundary_mask is not None
            else None
        ),
    )
    height, width = truth.shape
    centers = [
        (row, col)
        for row, col in (
            PlaceScoreManager._world_to_grid(x, y, width, height)
            for x, y in bomb_positions or []
        )
        if row is not None
    ]

    panel = _wall_destruction_panel(
        truth, destroyed, boundary, blast, centers
    )

    total_width = width
    total_height = COMPARISON_HEADER_HEIGHT + height + WALL_DESTRUCTION_LEGEND_HEIGHT
    canvas = np.full(
        (total_height, total_width, 3),
        _COMPARISON_COLOR_SEPARATOR,
        dtype=np.uint8,
    )
    canvas[:COMPARISON_HEADER_HEIGHT, :] = _COMPARISON_COLOR_HEADER_BG

    header = title or (
        "Wall Destruction"
        f"  ({len(centers)} bombs, radius {blast_radius:g} px,"
        f" boundary weight {boundary_wall_weight:g})"
    )
    cv2.putText(
        canvas,
        header,
        (8, COMPARISON_HEADER_HEIGHT - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _COMPARISON_COLOR_INK,
        1,
        cv2.LINE_AA,
    )

    canvas[COMPARISON_HEADER_HEIGHT:COMPARISON_HEADER_HEIGHT + height, :width] = (
        panel
    )

    if wall_breakdown is None:
        wall_breakdown = {"blast_radius": float(blast_radius)}
    _draw_wall_destruction_legend(
        canvas,
        COMPARISON_HEADER_HEIGHT + height,
        boundary_weight=float(boundary_wall_weight),
        breakdown=dict(wall_breakdown),
        score_percent=score_wall_percent,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise IOError(f"failed to write wall-destruction image: {path}")

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from swarm_rescue.tools.map_exploration_scoring import load_walls_json_as_obstacle_grid


def _predict_from_mode(truth_grid: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    truth = (truth_grid > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)

    if mode == "perfect":
        pred = truth.copy()
    elif mode == "under_detect":
        pred = cv2.erode(truth, kernel, iterations=1)
    elif mode == "over_detect":
        pred = cv2.dilate(truth, kernel, iterations=1)
    elif mode == "shift_right":
        pred = np.roll(truth, shift=4, axis=1)
        pred[:, :4] = 0
    elif mode == "noisy":
        pred = truth.copy()
        noise = (rng.random(truth.shape) < 0.015).astype(np.uint8)
        pred = np.bitwise_xor(pred, noise)
    elif mode == "boundary_only":
        pred = truth - cv2.erode(truth, kernel, iterations=1)
    elif mode == "blank":
        pred = np.zeros_like(truth)
    elif mode == "full_black":
        pred = np.ones_like(truth)
    elif mode == "random_binary":
        pred = (rng.random(truth.shape) < 0.5).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return (pred > 0).astype(np.uint8)


def _to_vis(grid: np.ndarray) -> np.ndarray:
    gray = ((grid == 0).astype(np.uint8) * 255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _diff_vis(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # BGR coloring:
    # - TP (both obstacle): white
    # - FN (missed obstacle): red
    # - FP (false obstacle): blue
    out = np.zeros((truth.shape[0], truth.shape[1], 3), dtype=np.uint8)
    truth_b = truth > 0
    pred_b = pred > 0
    tp = truth_b & pred_b
    fn = truth_b & (~pred_b)
    fp = (~truth_b) & pred_b
    out[tp] = (255, 255, 255)
    out[fn] = (0, 0, 255)
    out[fp] = (255, 0, 0)
    return out


def generate_mock_outputs(
    *,
    truth_walls_json: Path,
    output_dir: Path,
    mode: str = "under_detect",
    seed: int = 0,
    wall_threshold: int = 128,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = load_walls_json_as_obstacle_grid(truth_walls_json, wall_threshold=wall_threshold)
    rng = np.random.default_rng(seed)
    pred = _predict_from_mode(truth, mode=mode, rng=rng)

    pred_payload = {
        "grid": pred.astype(int).tolist(),
        "meta": {
            "mode": mode,
            "seed": seed,
            "truth_walls_json": str(truth_walls_json),
        },
    }

    pred_path = output_dir / f"pred_{mode}.json"
    truth_png = output_dir / f"truth_{mode}.png"
    pred_png = output_dir / f"pred_{mode}.png"
    diff_png = output_dir / f"diff_{mode}.png"

    pred_path.write_text(json.dumps(pred_payload, ensure_ascii=False), encoding="utf-8")
    cv2.imwrite(str(truth_png), _to_vis(truth))
    cv2.imwrite(str(pred_png), _to_vis(pred))
    cv2.imwrite(str(diff_png), _diff_vis(truth, pred))

    return {
        "prediction_json": str(pred_path),
        "truth_png": str(truth_png),
        "prediction_png": str(pred_png),
        "diff_png": str(diff_png),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock player prediction and visualization for a map.")
    parser.add_argument("--truth-walls-json", required=True, help="Ground-truth walls json, e.g. map_data/map_01.json")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated outputs")
    parser.add_argument(
        "--mode",
        default="under_detect",
        choices=[
            "perfect",
            "under_detect",
            "over_detect",
            "shift_right",
            "noisy",
            "boundary_only",
            "blank",
            "full_black",
            "random_binary",
        ],
        help="Prediction simulation mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noisy mode")
    parser.add_argument("--wall-threshold", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = generate_mock_outputs(
        truth_walls_json=Path(args.truth_walls_json),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        seed=args.seed,
        wall_threshold=args.wall_threshold,
    )
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


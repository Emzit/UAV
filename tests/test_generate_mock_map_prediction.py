import json
import pathlib
import sys

import cv2
import numpy as np

# Insert the 'src' directory, located two levels up from the current script,
# into sys.path. This ensures Python can find project-specific modules
# (e.g., 'swarm_rescue') when the script is run from a subfolder like 'tests/'.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.tools.generate_mock_map_prediction import generate_mock_outputs
from swarm_rescue.tools.map_exploration_scoring import load_walls_json_as_obstacle_grid


def test_generate_mock_outputs_writes_json_and_visualizations(tmp_path) -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    truth_json = repo_root / "map_data" / "map_01.json"

    output_dir = tmp_path / "mock_outputs"
    paths = generate_mock_outputs(
        truth_walls_json=truth_json,
        output_dir=output_dir,
        mode="under_detect",
        seed=7,
    )

    pred_path = pathlib.Path(paths["prediction_json"])
    vis_truth_path = pathlib.Path(paths["truth_png"])
    vis_pred_path = pathlib.Path(paths["prediction_png"])
    vis_diff_path = pathlib.Path(paths["diff_png"])

    assert pred_path.exists()
    assert vis_truth_path.exists()
    assert vis_pred_path.exists()
    assert vis_diff_path.exists()

    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    assert "grid" in payload
    pred_grid = np.array(payload["grid"], dtype=np.uint8)
    assert pred_grid.ndim == 2

    img_truth = cv2.imread(str(vis_truth_path), cv2.IMREAD_COLOR)
    img_pred = cv2.imread(str(vis_pred_path), cv2.IMREAD_COLOR)
    img_diff = cv2.imread(str(vis_diff_path), cv2.IMREAD_COLOR)
    assert img_truth is not None
    assert img_pred is not None
    assert img_diff is not None
    assert img_truth.shape == img_pred.shape == img_diff.shape


def test_generate_mock_outputs_supports_extra_requested_modes(tmp_path) -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    truth_json = repo_root / "map_data" / "map_01.json"
    truth = load_walls_json_as_obstacle_grid(truth_json)

    expected_modes = {
        "boundary_only",
        "blank",
        "full_black",
        "random_binary",
    }

    for mode in expected_modes:
        paths = generate_mock_outputs(
            truth_walls_json=truth_json,
            output_dir=tmp_path / mode,
            mode=mode,
            seed=7,
        )
        payload = json.loads(pathlib.Path(paths["prediction_json"]).read_text(encoding="utf-8"))
        pred = np.array(payload["grid"], dtype=np.uint8)

        assert pred.shape == truth.shape
        assert set(np.unique(pred).tolist()).issubset({0, 1})

        if mode == "boundary_only":
            assert pred.sum() > 0
            assert np.all(pred <= truth)
            assert pred.sum() < truth.sum()
        elif mode == "blank":
            assert pred.sum() == 0
        elif mode == "full_black":
            assert pred.sum() == pred.size
        elif mode == "random_binary":
            assert 0 < pred.sum() < pred.size


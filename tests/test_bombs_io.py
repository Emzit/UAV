import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from swarm_rescue.simulation.reporting.bombs_io import (
    bombs_from_data,
    bombs_to_serializable,
    build_bombs_document,
    exploration_map_output_path,
    load_bombs_file,
    save_placed_bombs,
    save_exploration_map,
)
from swarm_rescue.simulation.reporting.evaluation import EvalConfig
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.pose import Pose


class _DummyDisposal:
    pass


def test_load_and_roundtrip_bombs_file():
    data = {
        "map_name": "MapTest",
        "zones_config": [],
        "bombs": [
            {"x": 10.0, "y": -20.0, "theta": 0.5},
            {"x": 0.0, "y": 0.0, "theta": 0.0, "path": [[0, 0], [5, 5]]},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bombs.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_bombs_file(path)
        entries = bombs_from_data(loaded)
        assert len(entries) == 2
        assert entries[0]["x"] == 10.0
        assert entries[1]["path"] == [[0, 0], [5, 5]]


def test_bombs_to_serializable_from_mock_bomb():
    bomb = MagicMock()
    bomb.true_position.return_value = np.array([1.5, -2.5])
    bomb.true_angle.return_value = 0.25
    bomb.path.length.return_value = 1

    the_map = MagicMock()
    the_map._bombs = [bomb]
    entries = bombs_to_serializable(the_map)
    assert entries == [{"x": 1.5, "y": -2.5, "theta": 0.25}]


def test_save_placed_bombs_writes_json():
    bomb = MagicMock()
    bomb.true_position.return_value = np.array([100.0, 200.0])
    bomb.true_angle.return_value = 1.0
    bomb.path.length.return_value = 0

    the_map = MagicMock()
    the_map._bombs = [bomb]
    eval_config = EvalConfig(map_name="Map04", zones_config=())

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out_bombs.json"
        save_placed_bombs(out, the_map, eval_config)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["map_name"] == "Map04"
        assert doc["bombs"][0]["x"] == 100.0


def test_map_clear_and_spawn_bombs():
    class MiniMap(MapAbstract):
        def __init__(self):
            super().__init__(drone_type=MagicMock)
            self._number_bombs = 0
            self._playground = MagicMock()
            self._disposal_center = _DummyDisposal()

    the_map = MiniMap()
    bomb = MagicMock()
    the_map._bombs = [bomb]
    the_map._number_bombs = 1
    the_map._playground.elements = [bomb]

    the_map.clear_bombs()
    assert the_map._bombs == []
    assert the_map._number_bombs == 0
    the_map._playground.remove.assert_called_once_with(bomb)


def test_exploration_map_output_path_naming():
    from types import SimpleNamespace

    team_info = SimpleNamespace(team_number_str_padded="007", team_number_str="7")
    eval_config = EvalConfig(map_name="Map04", zones_config=())

    with tempfile.TemporaryDirectory() as tmp:
        out_path = exploration_map_output_path(
            result_path=tmp,
            team_info=team_info,
            eval_config=eval_config,
            round_number=3,
        )
        assert out_path is not None
        assert (
            Path(out_path).name
            == "team007_Map04_none_rd3_exploration.json"
        )


def test_save_exploration_map_writes_json_and_binary_grid():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out_exploration.json"
        grid = np.array([[0, 255], [1, 2]], dtype=np.uint8)

        save_exploration_map(
            path=out,
            grid=grid,
            elapsed_walltime=12.5,
        )

        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["elapsed_walltime"] == 12.5
        assert doc["time_seconds"] == 12.5
        assert doc["grid"] == [[0, 1], [1, 1]]

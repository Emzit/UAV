import json
import pathlib
import sys

# Insert the 'src' directory, located two levels up from the current script,
# into sys.path. This ensures Python can find project-specific modules.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.tools.json_to_map import generate_map_from_json


def _sample_editor_json(path: pathlib.Path) -> None:
    payload = {
        "version": 1,
        "canvas": {"width": 100, "height": 80, "background": "white"},
        "elements": [
            {
                "id": 1,
                "type": "line_wall",
                "points": [[10, 20], [90, 20]],
                "thickness": 6,
                "color": "#000000",
            },
            {
                "id": 2,
                "type": "rect_wall",
                "top_left": [60, 30],
                "bottom_right": [80, 50],
                "color": "#000000",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_map_from_json_writes_map_and_walls(tmp_path) -> None:
    input_json = tmp_path / "editor_map.json"
    _sample_editor_json(input_json)

    result = generate_map_from_json(
        input_json_path=input_json,
        name="custom_example",
        maps_dir=tmp_path / "maps",
    )

    # 新版只输出 map_py 与 walls_py（不再生成 map_data JSON/PNG）
    assert set(result) == {"map_py", "walls_py"}

    for path in result.values():
        assert pathlib.Path(path).exists()

    walls_text = pathlib.Path(result["walls_py"]).read_text(encoding="utf-8")
    assert "def add_walls(playground):" in walls_text
    assert "wall = NormalWall(pos_start=(-40, 20)," in walls_text
    assert "                      pos_end=(40, 20))" in walls_text
    assert "def add_boxes(playground):" in walls_text
    assert "box = NormalBox(up_left_point=(10, 10)," in walls_text
    assert "                    width=20, height=20)" in walls_text

    map_text = pathlib.Path(result["map_py"]).read_text(encoding="utf-8")
    assert "class MapCustomExample(MapAbstract):" in map_text
    assert "from swarm_rescue.maps.walls_custom_example import add_walls, add_boxes" in map_text
    assert "self._size_area = (100, 80)" in map_text
    assert "self._start_area_drones = " in map_text
    assert "self._return_area" not in map_text
    assert "ReturnArea" not in map_text
    assert "start_area_drones = self._start_area_drones" in map_text
    assert "number_drones: int | None = None" in map_text


def test_generate_map_from_json_uses_editor_start_and_rescue_placements(tmp_path) -> None:
    input_json = tmp_path / "with_zones.json"
    payload = {
        "version": 1,
        "canvas": {"width": 100, "height": 80, "background": "white"},
        "elements": [
            {
                "id": 1,
                "type": "start_area",
                "top_left": [0, 0],
                "bottom_right": [40, 40],
                "color": "#1E90FF",
            },
            {
                "id": 2,
                "type": "disposal_center",
                "top_left": [60, 20],
                "bottom_right": [99, 79],
                "color": "#228B22",
            },
        ],
    }
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    result = generate_map_from_json(
        input_json_path=input_json,
        name="zones_map",
        maps_dir=tmp_path / "maps",
    )

    map_text = pathlib.Path(result["map_py"]).read_text(encoding="utf-8")
    # start_area (0,0)-(40,40) 中心 (20,20) -> 世界 (-30, 20)
    assert "self._start_area_drones = (-30, 20)" in map_text
    # disposal_center (60,20)-(99,79) -> size (39,59) 中心 (30,-10)
    assert "self._disposal_center = DisposalCenter(size=(39, 59))" in map_text
    assert "self._disposal_center_pos = ((30, -10), 0)" in map_text


def test_generate_map_from_json_defaults_when_no_zones(tmp_path) -> None:
    input_json = tmp_path / "walls_only.json"
    _sample_editor_json(input_json)

    result = generate_map_from_json(
        input_json_path=input_json,
        name="walls_only_map",
        maps_dir=tmp_path / "maps",
    )

    map_text = pathlib.Path(result["map_py"]).read_text(encoding="utf-8")
    # 没有起点区域/处置中心时使用默认位置（左下角附近起点 + 右下角处置中心）
    assert "self._start_area_drones = " in map_text
    assert "self._disposal_center = DisposalCenter(size=" in map_text
    assert "add_walls(self._playground)" in map_text

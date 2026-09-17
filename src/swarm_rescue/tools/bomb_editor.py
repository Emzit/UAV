#!/usr/bin/env python3
"""
Bomb Editor — 交互式炸弹放置工具

为已有地图添加炸弹并保存为 JSON 文件，供救援模式使用。

用法:
    .venv/bin/python -m swarm_rescue.tools.bomb_editor Map01
    .venv/bin/python -m swarm_rescue.tools.bomb_editor Map04 -o my_bombs.json
    .venv/bin/python -m swarm_rescue.tools.bomb_editor Map05
"""
import argparse
import importlib
import os
import sys
from typing import Type

from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case (e.g. Map01 -> map_01, Map04 -> map_04)."""
    result = [name[0].lower()]
    for prev, ch in zip(name, name[1:]):
        if ch.isupper() or (ch.isdigit() and prev.isalpha()):
            result.append("_")
        result.append(ch.lower())
    return "".join(result)


def _resolve_map_class(map_name: str) -> Type[MapAbstract]:
    # Ensure we have a proper class name
    if not map_name.startswith("Map"):
        class_name = f"Map{map_name}"
    else:
        class_name = map_name

    module_name = f"swarm_rescue.maps.{_camel_to_snake(class_name)}"

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: cannot find module {module_name!r} for map {map_name!r}", file=sys.stderr)
        sys.exit(1)

    cls = getattr(module, class_name, None)
    if cls is None or not issubclass(cls, MapAbstract):
        print(f"Error: class {class_name!r} not found in {module_name}", file=sys.stderr)
        sys.exit(1)

    return cls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="交互式炸弹编辑器 — 给地图添加炸弹并保存为 JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "快捷键:\n"
            "  左键点击  — 添加炸弹\n"
            "  右键点击  — 移除最近的炸弹\n"
            "  J        — 导出炸弹到 JSON\n"
            "  K        — 清除所有炸弹\n"
            "  I        — 从 JSON 导入炸弹\n"
            "  Enter    — 开始模拟\n"
            "  Q        — 退出\n"
        ),
    )
    parser.add_argument("map_name", help="地图类名，如 Map01, Map02, ..., Map05")
    parser.add_argument("-o", "--output", default=None, help="炸弹 JSON 输出路径")
    args = parser.parse_args()

    # Resolve map class
    map_class = _resolve_map_class(args.map_name)
    class_name = map_class.__name__

    # Output path
    output = args.output
    if output is None:
        output = f"bombs_{class_name}.json"

    # Instantiate map (no drones needed for editing)
    print(f"Loading map {class_name} ...")
    the_map = map_class(drone_type=DroneMotionless)
    the_map.clear_bombs()

    # Launch GUI in bomb-edit mode
    gui = GuiSR(
        the_map=the_map,
        use_mouse_measure=True,
        manual_bomb_edit_enabled=True,
        manual_bombs_out_path=output,
    )
    gui.set_caption(f"Bomb Editor — {class_name}")

    print()
    print("=" * 66)
    print(f"  Bomb Editor — Map: {class_name}")
    print(f"  Output:       {output}")
    print("=" * 66)
    print("  Left-click  → add bomb       Right-click → remove nearest")
    print("  J           → export JSON    K           → clear all")
    print("  I           → import JSON    Enter       → start simulation")
    print("  Q           → quit")
    print("=" * 66)
    print()

    gui.run()

    # After GUI closes, check if bombs were exported and print instructions
    if os.path.exists(output):
        print()
        print("=" * 66)
        print(f"  ✅ Bombs saved to: {output}")
        print("=" * 66)
        print()
        print("  Preview with Launcher (full simulation):")
        print()
        print(f"    .venv/bin/python -m swarm_rescue.launcher \\")
        print(f"        --manual-bomb-edit \\")
        print(f"        --manual-bombs-in {output} \\")
        print(f"        -c config/competition_rescue_eval_plan.yml")
        print()
        print("  Preview with Python (static view with motionless drones):")
        print()
        print(f"    .venv/bin/python -c \"")
        module_snake = _camel_to_snake(class_name)
        print(f"    from swarm_rescue.maps.{module_snake} import {class_name}")
        print(f"    from swarm_rescue.simulation.reporting.bombs_io import bombs_from_data, load_bombs_file")
        print(f"    from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless")
        print(f"    from swarm_rescue.simulation.gui_map.gui_sr import GuiSR")
        print(f"    ")
        print(f"    the_map = {class_name}(drone_type=DroneMotionless)")
        print(f"    the_map.clear_bombs()")
        print(f"    the_map.spawn_bombs(bombs_from_data(load_bombs_file('{output}')))")
        print(f"    ")
        print(f"    gui = GuiSR(the_map=the_map, use_mouse_measure=True)")
        print(f"    gui.set_caption('Preview — {class_name} with bombs')")
        print(f"    gui.run()")
        print(f"    \"")
        print()
        print("  Use the saved JSON with a YAML config bombs_file for evaluation:")
        print()
        print(f"    .venv/bin/python -m swarm_rescue.launcher \\")
        print(f"        --bombs-file {output} \\")
        print(f"        -c config/competition_rescue_eval_plan.yml")
        print()
    else:
        print()
        print("  No bombs exported. Press J during editing to save the JSON file.")
        print()


if __name__ == "__main__":
    main()

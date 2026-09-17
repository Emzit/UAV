"""
将 walls_xxx.py 转换为地图编辑器可识别的 JSON 文件。
通过静态解析 Python 源码提取墙和方块坐标，无需导入模块。

用法：
    python src/swarm_rescue/tools/walls_to_json.py src/swarm_rescue/maps/walls_04.py output.json [--width 1660] [--height 1122]
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path


def parse_walls_file(filepath: Path) -> tuple[list, list, int]:
    """
    静态解析 walls_xxx.py，提取所有 NormalWall 和 NormalBox 的坐标。
    返回 (walls, boxes, skipped)
    walls: [(pos_start, pos_end), ...]
    boxes: [(up_left_point, width, height), ...]
    skipped: 无法解析而跳过的元素数量
    """
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)

    walls = []
    boxes = []
    skipped = 0

    # 遍历所有函数调用
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        # 获取被调用的类名
        if isinstance(func, ast.Name):
            class_name = func.id
        elif isinstance(func, ast.Attribute):
            class_name = func.attr
        else:
            continue

        if class_name not in ("NormalWall", "NormalBox"):
            continue

        # 提取关键字参数
        kwargs = {}
        for kw in node.keywords:
            key = kw.arg
            val = _eval_ast(kw.value)
            if val is not None:
                kwargs[key] = val

        # 也检查位置参数
        args = [_eval_ast(a) for a in node.args]

        if class_name == "NormalWall":
            pos_start = kwargs.get("pos_start") or (args[0] if len(args) > 0 else None)
            pos_end = kwargs.get("pos_end") or (args[1] if len(args) > 1 else None)
            if (pos_start and pos_end
                    and all(v is not None for v in pos_start)
                    and all(v is not None for v in pos_end)):
                walls.append((tuple(pos_start), tuple(pos_end)))
            else:
                skipped += 1

        elif class_name == "NormalBox":
            up_left = kwargs.get("up_left_point") or (args[0] if len(args) > 0 else None)
            width = kwargs.get("width") or (args[1] if len(args) > 1 else None)
            height = kwargs.get("height") or (args[2] if len(args) > 2 else None)
            if (up_left and width is not None and height is not None
                    and all(v is not None for v in up_left)):
                boxes.append((tuple(up_left), float(width), float(height)))
            else:
                skipped += 1

    return walls, boxes, skipped


def _eval_ast(node):
    """简单地求值一些常量 AST 节点（数字、tuple、list、UnaryOp等）。"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_eval_ast(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval_ast(node.operand)
        return -v if v is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if left is not None and right is not None:
            return left - right
    return None


def walls_to_json(walls_file: Path, width: int, height: int) -> tuple[dict, int]:
    walls, boxes, skipped = parse_walls_file(walls_file)

    cx = width / 2
    cy = height / 2

    def world_to_canvas(x_world, y_world):
        x_canvas = int(round(x_world + cx))
        y_canvas = int(round(cy - y_world))  # y 轴翻转
        return x_canvas, y_canvas

    elements = []
    element_id = 1

    # 线段墙
    for pos_start, pos_end in walls:
        x0, y0 = world_to_canvas(pos_start[0], pos_start[1])
        x1, y1 = world_to_canvas(pos_end[0], pos_end[1])
        elements.append({
            "id": element_id,
            "type": "line_wall",
            "points": [[x0, y0], [x1, y1]],
            "thickness": 6,
            "color": "#000000",
        })
        element_id += 1

    # 方块墙
    for up_left, w, h in boxes:
        # up_left 是世界坐标的左上角（x 最小，y 最大）
        ul_x, ul_y = world_to_canvas(up_left[0], up_left[1])
        br_x, br_y = world_to_canvas(up_left[0] + w, up_left[1] - h)
        # 保证 top_left 在左上角
        tl_x = min(ul_x, br_x)
        tl_y = min(ul_y, br_y)
        br_x2 = max(ul_x, br_x)
        br_y2 = max(ul_y, br_y)
        elements.append({
            "id": element_id,
            "type": "rect_wall",
            "top_left": [tl_x, tl_y],
            "bottom_right": [br_x2, br_y2],
            "color": "#000000",
        })
        element_id += 1

    data = {
        "version": 1,
        "canvas": {
            "width": width,
            "height": height,
            "background": "white",
        },
        "elements": elements,
    }
    return data, skipped


def main():
    parser = argparse.ArgumentParser(description="将 walls_xxx.py 转换为地图编辑器 JSON")
    parser.add_argument("walls_file", type=Path, help="walls_xxx.py 文件路径")
    parser.add_argument("output", type=Path, help="输出 JSON 文件路径")
    parser.add_argument("--width", type=int, default=1113, help="地图宽度（像素）")
    parser.add_argument("--height", type=int, default=750, help="地图高度（像素）")
    args = parser.parse_args()

    if not args.walls_file.exists():
        print(f"错误：文件不存在 {args.walls_file}", file=sys.stderr)
        sys.exit(1)

    data, skipped = walls_to_json(args.walls_file, args.width, args.height)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    wall_count = sum(1 for e in data["elements"] if e["type"] == "line_wall")
    box_count = sum(1 for e in data["elements"] if e["type"] == "rect_wall")
    print(f"转换完成！")
    print(f"  输入：{args.walls_file}")
    print(f"  输出：{args.output}")
    print(f"  地图尺寸：{args.width} x {args.height}")
    print(f"  线段墙：{wall_count} 条")
    print(f"  方块墙：{box_count} 个")
    print(f"  元素总数：{len(data['elements'])}")
    if skipped > 0:
        print(f"  跳过（无法解析）：{skipped} 个")
        print(f"  提示：使用了变量/表达式作为坐标的墙体无法被静态解析，需手动在编辑器中添加。")


if __name__ == "__main__":
    main()

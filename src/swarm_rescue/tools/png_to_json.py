# -*- coding: utf-8 -*-
"""
PNG 图片转地图编辑器 JSON 文件。

从 PNG 图片中矢量化提取墙体（黑色=墙体，白色=背景），
输出为地图编辑器可识别的 JSON 格式，方便后续在编辑器中精修。

算法（两阶段）：
  阶段 1 - 实心矩形块：RETR_CCOMP 找所有连通域，矩形度高且两边都比较厚 → 矩形墙
  阶段 2 - 线条墙体：骨架化 + HoughLinesP 概率霍夫变换 → 线段墙

用法：
    python src/swarm_rescue/tools/png_to_json.py <输入PNG> <输出JSON>
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from swarm_rescue.tools.map_wall_editor import (
    DEFAULT_COLOR,
    DEFAULT_LINE_THICKNESS,
    LINE_WALL,
    RECT_WALL,
    WallElement,
    _clamp,
    render_elements_to_image,
    serialize_elements,
)


def image_to_elements(image_path: str) -> Tuple[int, int, List[WallElement]]:
    """
    从 PNG 图片中矢量化提取墙体元素。
    黑色像素 = 墙体，白色/浅色 = 背景。
    返回 (width, height, elements)。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    height, width = img.shape[:2]

    # 二值化：Otsu 自动阈值，暗色（墙体）为 255，亮色为 0
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elements: List[WallElement] = []
    next_id = 1

    # ============================================================
    # 阶段 1：识别实心矩形块
    # ============================================================
    MIN_RECT_AREA = 200        # 最小面积
    MIN_SOLID_SIDE = 15        # 实心块最薄边也要超过这个值（区分粗线和方块）
    RECT_RATIO_THRESH = 0.90   # 矩形度（轮廓面积 / 包围盒面积）

    solid_mask = np.zeros_like(binary)

    # RETR_CCOMP: 所有顶层轮廓就是所有黑色连通域
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
    )

    for i, cnt in enumerate(contours):
        # 只看顶层轮廓（parent == -1）
        if hierarchy[0][i][3] != -1:
            continue

        area = cv2.contourArea(cnt)
        if area < MIN_RECT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 跳过几乎等于整张图大小的外边框
        fills_most = (w * h) / (width * height) > 0.9
        touches_edge = (x < 20 or y < 20
                        or x + w > width - 20
                        or y + h > height - 20)
        if fills_most and touches_edge:
            continue

        rect_area = w * h
        if rect_area <= 0:
            continue

        # 实心方块判断：矩形度高 + 两边都有一定厚度
        if (area / rect_area >= RECT_RATIO_THRESH
                and min(w, h) >= MIN_SOLID_SIDE):
            elements.append(
                WallElement(
                    element_id=next_id,
                    element_type=RECT_WALL,
                    color=DEFAULT_COLOR,
                    top_left=(int(x), int(y)),
                    bottom_right=(int(x + w - 1), int(y + h - 1)),
                )
            )
            next_id += 1
            cv2.drawContours(solid_mask, [cnt], -1, 255, -1)

    # 从二值图中移除实心块，避免干扰直线检测
    binary_lines = cv2.bitwise_and(binary, cv2.bitwise_not(solid_mask))

    # 骨架化（细化）：把粗线变成 1 像素宽，避免霍夫检测出双线
    def _skeletonize(img_bin: np.ndarray) -> np.ndarray:
        """形态学细化算法，输入二值图（255=前景），输出细化后的二值图。"""
        skel = np.zeros_like(img_bin)
        img = img_bin.copy()
        kernel = np.ones((3, 3), np.uint8)
        while True:
            _, img2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            eroded = cv2.erode(img2, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img2, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    skeleton = _skeletonize(binary_lines)

    # ============================================================
    # 阶段 2：Hough 直线检测提取线段墙
    # ============================================================
    MIN_LINE_LEN = 8       # 最小线段长度
    MAX_LINE_GAP = 30      # 线段间最大间隙（同一直线上可被连接）

    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=8,
        minLineLength=MIN_LINE_LEN,
        maxLineGap=MAX_LINE_GAP,
    )

    if lines is None:
        return width, height, elements

    # 兼容不同 OpenCV 版本的返回格式：(N,1,4) 或 (N,4)
    lines_arr = np.asarray(lines)
    if lines_arr.ndim == 3 and lines_arr.shape[1] == 1:
        lines_arr = lines_arr.reshape(-1, 4)
    elif lines_arr.ndim != 2 or lines_arr.shape[1] != 4:
        return width, height, elements

    all_lines = [tuple(row) for row in lines_arr]

    # ---- 合并共线且接近的线段 ----
    def _line_len(x1, y1, x2, y2):
        return float(np.hypot(x2 - x1, y2 - y1))

    MERGE_ANGLE_TOL = 5.0   # 角度容差（度）
    MERGE_DIST_TOL = 5.0    # 点到直线距离容差（像素）
    MERGE_GAP_TOL = 25.0    # 两条线段沿直线方向的最大允许间距（像素）

    merged = list(all_lines)

    for _ in range(3):
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            x1, y1, x2, y2 = merged[i]
            used[i] = True

            dx = x2 - x1
            dy = y2 - y1
            llen = float(np.hypot(dx, dy))
            if llen < 0.1:
                continue

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                ox1, oy1, ox2, oy2 = merged[j]

                # 角度差
                ang_a = float(np.degrees(np.arctan2(dy, dx)))
                ang_b = float(np.degrees(np.arctan2(oy2 - oy1, ox2 - ox1)))
                diff = abs(ang_a - ang_b) % 180
                if diff > 90:
                    diff = 180 - diff
                if diff > MERGE_ANGLE_TOL:
                    continue

                # 两条线是否共线：对方两个端点到我方无限直线的距离都很近
                def _point_to_line_dist(px, py, ax, ay, bx, by):
                    """点到无限直线的距离。"""
                    dx = bx - ax
                    dy = by - ay
                    line_len = (dx * dx + dy * dy) ** 0.5
                    if line_len < 0.001:
                        return float('inf')
                    return abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / line_len

                d1 = _point_to_line_dist(
                    float(ox1), float(oy1), float(x1), float(y1), float(x2), float(y2),
                )
                d2 = _point_to_line_dist(
                    float(ox2), float(oy2), float(x1), float(y1), float(x2), float(y2),
                )
                if max(d1, d2) > MERGE_DIST_TOL:
                    continue

                # 检查沿直线方向的间距：两条线段之间的空隙不能太大
                ux, uy = dx / llen, dy / llen
                p_a1 = x1 * ux + y1 * uy
                p_a2 = x2 * ux + y2 * uy
                a_min = min(p_a1, p_a2)
                a_max = max(p_a1, p_a2)
                p_b1 = ox1 * ux + oy1 * uy
                p_b2 = ox2 * ux + oy2 * uy
                b_min = min(p_b1, p_b2)
                b_max = max(p_b1, p_b2)
                gap = max(a_min, b_min) - min(a_max, b_max)
                if gap > MERGE_GAP_TOL:
                    continue

                # 合并：所有四点在直线方向上投影取极值
                pts = [(x1, y1), (x2, y2), (ox1, oy1), (ox2, oy2)]
                projs = [px * ux + py * uy for px, py in pts]
                imin = int(np.argmin(projs))
                imax = int(np.argmax(projs))
                x1, y1 = pts[imin]
                x2, y2 = pts[imax]
                dx = x2 - x1
                dy = y2 - y1
                llen = float(np.hypot(dx, dy))
                if llen < 0.1:
                    break
                used[j] = True

            new_merged.append((x1, y1, x2, y2))
        merged = new_merged

    # 转成线段墙元素
    for (x1, y1, x2, y2) in merged:
        if _line_len(x1, y1, x2, y2) < MIN_LINE_LEN:
            continue

        x1i = _clamp(int(round(x1)), 0, width - 1)
        y1i = _clamp(int(round(y1)), 0, height - 1)
        x2i = _clamp(int(round(x2)), 0, width - 1)
        y2i = _clamp(int(round(y2)), 0, height - 1)

        if x1i == x2i and y1i == y2i:
            continue

        elements.append(
            WallElement(
                element_id=next_id,
                element_type=LINE_WALL,
                color=DEFAULT_COLOR,
                thickness=DEFAULT_LINE_THICKNESS,
                points=[(x1i, y1i), (x2i, y2i)],
            )
        )
        next_id += 1

    return width, height, elements


def png_to_json(png_path: Path, json_path: Path) -> dict:
    """将 PNG 图片转为地图编辑器 JSON 文件，返回统计信息。"""
    width, height, elements = image_to_elements(str(png_path))
    data = serialize_elements(width, height, elements)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    line_count = sum(1 for e in data["elements"] if e["type"] == "line_wall")
    rect_count = sum(1 for e in data["elements"] if e["type"] == "rect_wall")
    return {
        "input": str(png_path),
        "output": str(json_path),
        "width": width,
        "height": height,
        "line_walls": line_count,
        "rect_walls": rect_count,
        "total_elements": len(data["elements"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PNG 地图图片矢量化为地图编辑器 JSON（黑色=墙体，白色=背景）"
    )
    parser.add_argument("input_png", type=Path, help="输入的 PNG 图片路径")
    parser.add_argument("output_json", type=Path, help="输出的 JSON 文件路径")
    parser.add_argument(
        "--export-png",
        action="store_true",
        help="同时导出识别结果的渲染 PNG（与输出 JSON 同目录同名）",
    )
    args = parser.parse_args()

    if not args.input_png.exists():
        print(f"错误：文件不存在 {args.input_png}", file=sys.stderr)
        sys.exit(1)

    try:
        info = png_to_json(args.input_png, args.output_json)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    print("转换完成！")
    print(f"  输入：{info['input']}")
    print(f"  输出：{info['output']}")
    print(f"  地图尺寸：{info['width']} x {info['height']}")
    print(f"  线段墙：{info['line_walls']} 条")
    print(f"  方块墙：{info['rect_walls']} 个")
    print(f"  元素总数：{info['total_elements']}")

    if args.export_png:
        # 再读取一次来渲染（保证与 JSON 内容一致）
        from swarm_rescue.tools.map_wall_editor import load_elements
        with open(args.output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        canvas = data.get("canvas", {}) or {}
        w = int(canvas.get("width", 1113))
        h = int(canvas.get("height", 750))
        elems = load_elements(data, width=w, height=h)
        rendered = render_elements_to_image(w, h, elems)
        out_png = args.output_json.with_suffix(".png")
        cv2.imwrite(str(out_png), rendered)
        print(f"  渲染预览：{out_png}")

    print()
    print("提示：矢量化结果可能存在边角断裂或误识别，建议在地图编辑器中手动精修。")


if __name__ == "__main__":
    main()

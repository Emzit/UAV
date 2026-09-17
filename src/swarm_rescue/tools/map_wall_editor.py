# -*- coding: utf-8 -*-
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SELECT = "select"
LINE_WALL = "line_wall"
RECT_WALL = "rect_wall"
START_AREA = "start_area"
DISPOSAL_CENTER = "disposal_center"
HIT_TOLERANCE = 8  # 线段墙点击命中容差（像素）
DEFAULT_COLOR = "#000000"
DEFAULT_START_AREA_COLOR = "#1E90FF"
DEFAULT_DISPOSAL_CENTER_COLOR = "#228B22"
DEFAULT_LINE_THICKNESS = 6
DEFAULT_CANVAS_WIDTH = 1113
DEFAULT_CANVAS_HEIGHT = 750


def _is_rect_like_tool(tool: str) -> bool:
    return tool in (RECT_WALL, START_AREA, DISPOSAL_CENTER)


@dataclass
class WallElement:
    element_id: int
    element_type: str
    color: str = DEFAULT_COLOR
    thickness: int = DEFAULT_LINE_THICKNESS
    points: Optional[List[Tuple[int, int]]] = None
    top_left: Optional[Tuple[int, int]] = None
    bottom_right: Optional[Tuple[int, int]] = None


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def _normalize_rect(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))


def _point_to_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """计算点到线段的最短距离。"""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _hit_test(element: WallElement, x: int, y: int) -> bool:
    """判断点 (x, y) 是否命中一个元素。"""
    if element.element_type == LINE_WALL and element.points:
        dist = _point_to_segment_distance(
            x, y,
            element.points[0][0], element.points[0][1],
            element.points[1][0], element.points[1][1],
        )
        return dist <= HIT_TOLERANCE
    if element.top_left and element.bottom_right:
        return (
            element.top_left[0] <= x <= element.bottom_right[0]
            and element.top_left[1] <= y <= element.bottom_right[1]
        )
    return False


def _parse_hex_color(color_hex: str) -> Tuple[int, int, int]:
    color_hex = color_hex.lstrip("#")
    if len(color_hex) != 6:
        return 0, 0, 0
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return b, g, r


def serialize_elements(width: int, height: int, elements: List[WallElement]) -> Dict:
    serialized_elements = []
    for element in elements:
        if element.element_type == LINE_WALL and element.points:
            serialized_elements.append(
                {
                    "id": element.element_id,
                    "type": LINE_WALL,
                    "points": [
                        [element.points[0][0], element.points[0][1]],
                        [element.points[1][0], element.points[1][1]],
                    ],
                    "thickness": element.thickness,
                    "color": element.color,
                }
            )
        elif element.element_type == RECT_WALL and element.top_left and element.bottom_right:
            serialized_elements.append(
                {
                    "id": element.element_id,
                    "type": RECT_WALL,
                    "top_left": [element.top_left[0], element.top_left[1]],
                    "bottom_right": [element.bottom_right[0], element.bottom_right[1]],
                    "color": element.color,
                }
            )
        elif element.element_type == START_AREA and element.top_left and element.bottom_right:
            serialized_elements.append(
                {
                    "id": element.element_id,
                    "type": START_AREA,
                    "top_left": [element.top_left[0], element.top_left[1]],
                    "bottom_right": [element.bottom_right[0], element.bottom_right[1]],
                    "color": element.color,
                }
            )
        elif element.element_type == DISPOSAL_CENTER and element.top_left and element.bottom_right:
            serialized_elements.append(
                {
                    "id": element.element_id,
                    "type": DISPOSAL_CENTER,
                    "top_left": [element.top_left[0], element.top_left[1]],
                    "bottom_right": [element.bottom_right[0], element.bottom_right[1]],
                    "color": element.color,
                }
            )

    return {
        "version": 1,
        "canvas": {
            "width": width,
            "height": height,
            "background": "white",
        },
        "elements": serialized_elements,
    }


def load_elements(json_data: Dict, width: int, height: int) -> List[WallElement]:
    elements: List[WallElement] = []

    for raw in json_data.get("elements", []):
        element_id = int(raw.get("id", len(elements) + 1))
        element_type = raw.get("type")

        if element_type == LINE_WALL:
            points = raw.get("points", [])
            if len(points) != 2:
                continue

            x0 = _clamp(int(points[0][0]), 0, width - 1)
            y0 = _clamp(int(points[0][1]), 0, height - 1)
            x1 = _clamp(int(points[1][0]), 0, width - 1)
            y1 = _clamp(int(points[1][1]), 0, height - 1)
            thickness = int(raw.get("thickness", DEFAULT_LINE_THICKNESS))

            elements.append(
                WallElement(
                    element_id=element_id,
                    element_type=LINE_WALL,
                    color=raw.get("color", DEFAULT_COLOR),
                    thickness=thickness,
                    points=[(x0, y0), (x1, y1)],
                )
            )
        elif element_type == RECT_WALL:
            top_left = raw.get("top_left")
            bottom_right = raw.get("bottom_right")
            if not top_left or not bottom_right:
                continue

            x0 = _clamp(int(top_left[0]), 0, width - 1)
            y0 = _clamp(int(top_left[1]), 0, height - 1)
            x1 = _clamp(int(bottom_right[0]), 0, width - 1)
            y1 = _clamp(int(bottom_right[1]), 0, height - 1)
            normalized_top_left, normalized_bottom_right = _normalize_rect(x0, y0, x1, y1)

            elements.append(
                WallElement(
                    element_id=element_id,
                    element_type=RECT_WALL,
                    color=raw.get("color", DEFAULT_COLOR),
                    top_left=normalized_top_left,
                    bottom_right=normalized_bottom_right,
                )
            )
        elif element_type == START_AREA:
            top_left = raw.get("top_left")
            bottom_right = raw.get("bottom_right")
            if not top_left or not bottom_right:
                continue

            x0 = _clamp(int(top_left[0]), 0, width - 1)
            y0 = _clamp(int(top_left[1]), 0, height - 1)
            x1 = _clamp(int(bottom_right[0]), 0, width - 1)
            y1 = _clamp(int(bottom_right[1]), 0, height - 1)
            normalized_top_left, normalized_bottom_right = _normalize_rect(x0, y0, x1, y1)

            elements = [e for e in elements if e.element_type != START_AREA]
            elements.append(
                WallElement(
                    element_id=element_id,
                    element_type=START_AREA,
                    color=raw.get("color", DEFAULT_START_AREA_COLOR),
                    top_left=normalized_top_left,
                    bottom_right=normalized_bottom_right,
                )
            )
        elif element_type == DISPOSAL_CENTER:
            top_left = raw.get("top_left")
            bottom_right = raw.get("bottom_right")
            if not top_left or not bottom_right:
                continue

            x0 = _clamp(int(top_left[0]), 0, width - 1)
            y0 = _clamp(int(top_left[1]), 0, height - 1)
            x1 = _clamp(int(bottom_right[0]), 0, width - 1)
            y1 = _clamp(int(bottom_right[1]), 0, height - 1)
            normalized_top_left, normalized_bottom_right = _normalize_rect(x0, y0, x1, y1)

            elements = [e for e in elements if e.element_type != DISPOSAL_CENTER]
            elements.append(
                WallElement(
                    element_id=element_id,
                    element_type=DISPOSAL_CENTER,
                    color=raw.get("color", DEFAULT_DISPOSAL_CENTER_COLOR),
                    top_left=normalized_top_left,
                    bottom_right=normalized_bottom_right,
                )
            )

    return elements


def render_elements_to_image(width: int, height: int, elements: List[WallElement]) -> np.ndarray:
    """将墙体元素渲染为 PNG 图片（白色背景 + 黑色墙体 + 彩色标记）。"""
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    for element in elements:
        color = _parse_hex_color(element.color)
        if element.element_type == LINE_WALL and element.points:
            cv2.line(
                image,
                element.points[0],
                element.points[1],
                color,
                element.thickness,
                cv2.LINE_AA,
            )
        elif element.element_type == RECT_WALL and element.top_left and element.bottom_right:
            cv2.rectangle(
                image,
                element.top_left,
                element.bottom_right,
                color,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
        elif element.element_type == START_AREA and element.top_left and element.bottom_right:
            # 起点区域：空心矩形（蓝色）
            cv2.rectangle(
                image,
                element.top_left,
                element.bottom_right,
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        elif element.element_type == DISPOSAL_CENTER and element.top_left and element.bottom_right:
            # 处置中心：红色实心填充
            cv2.rectangle(
                image,
                element.top_left,
                element.bottom_right,
                (0, 0, 255),  # BGR 红色
                thickness=-1,
                lineType=cv2.LINE_8,
            )

    return image


class EditorState:
    def __init__(self) -> None:
        self.canvas_width = DEFAULT_CANVAS_WIDTH
        self.canvas_height = DEFAULT_CANVAS_HEIGHT
        self.elements: List[WallElement] = []
        self.current_tool = SELECT
        self.first_point: Optional[Tuple[int, int]] = None
        self.next_id = 1
        self.selected_id: Optional[int] = None  # 当前选中的元素 ID

    def get_selected(self) -> Optional[WallElement]:
        if self.selected_id is None:
            return None
        for e in self.elements:
            if e.element_id == self.selected_id:
                return e
        return None

    def resize_canvas(self, new_width: int, new_height: int) -> None:
        """
        调整画布尺寸，以左上角为锚点。
        尺寸变小时，裁剪超出边界的元素：
          - 矩形类：完全在外部的删除，部分超出的截断到边界
          - 线段墙：完全在外部的删除，部分超出的截断到边界
        """
        new_width = max(1, int(new_width))
        new_height = max(1, int(new_height))
        if new_width == self.canvas_width and new_height == self.canvas_height:
            return

        # 旧尺寸到新尺寸：元素完全保留（以左上角为锚，坐标不需要偏移）
        # 只需要裁剪掉超出新边界的部分
        new_elements: List[WallElement] = []
        for elem in self.elements:
            if elem.element_type == LINE_WALL and elem.points:
                p0 = list(elem.points[0])
                p1 = list(elem.points[1])
                # 线段裁剪到边界（Sutherland-Cohen 简化版，因为是轴对齐裁剪）
                p0[0] = _clamp(p0[0], 0, new_width - 1)
                p0[1] = _clamp(p0[1], 0, new_height - 1)
                p1[0] = _clamp(p1[0], 0, new_width - 1)
                p1[1] = _clamp(p1[1], 0, new_height - 1)
                # 裁剪后退化成点了就丢弃
                if p0[0] == p1[0] and p0[1] == p1[1]:
                    continue
                elem.points = [tuple(p0), tuple(p1)]
                new_elements.append(elem)
            elif elem.top_left is not None and elem.bottom_right is not None:
                tl_x = _clamp(elem.top_left[0], 0, new_width - 1)
                tl_y = _clamp(elem.top_left[1], 0, new_height - 1)
                br_x = _clamp(elem.bottom_right[0], 0, new_width - 1)
                br_y = _clamp(elem.bottom_right[1], 0, new_height - 1)
                # 裁剪后面积为 0 就丢弃
                if br_x <= tl_x or br_y <= tl_y:
                    continue
                elem.top_left = (tl_x, tl_y)
                elem.bottom_right = (br_x, br_y)
                new_elements.append(elem)
            else:
                # 未知类型，保留
                new_elements.append(elem)

        self.elements = new_elements
        self.canvas_width = new_width
        self.canvas_height = new_height

        # 如果选中的元素被删了，清空选中
        if self.selected_id is not None:
            if not any(e.element_id == self.selected_id for e in self.elements):
                self.selected_id = None

    def reset_first_point(self) -> None:
        self.first_point = None

    def add_element(self, x0: int, y0: int, x1: int, y1: int) -> Optional[WallElement]:
        if self.current_tool == LINE_WALL:
            if x0 == x1 and y0 == y1:
                return None
            element = WallElement(
                element_id=self.next_id,
                element_type=LINE_WALL,
                points=[(x0, y0), (x1, y1)],
                thickness=DEFAULT_LINE_THICKNESS,
            )
            self.elements.append(element)
            self.next_id += 1
            return element

        top_left, bottom_right = _normalize_rect(x0, y0, x1, y1)
        if top_left == bottom_right:
            return None

        if self.current_tool == RECT_WALL:
            element = WallElement(
                element_id=self.next_id,
                element_type=RECT_WALL,
                top_left=top_left,
                bottom_right=bottom_right,
            )
            self.elements.append(element)
            self.next_id += 1
            return element

        if self.current_tool == START_AREA:
            self.elements = [e for e in self.elements if e.element_type != START_AREA]
            element = WallElement(
                element_id=self.next_id,
                element_type=START_AREA,
                color=DEFAULT_START_AREA_COLOR,
                top_left=top_left,
                bottom_right=bottom_right,
            )
        else:
            self.elements = [e for e in self.elements if e.element_type != DISPOSAL_CENTER]
            element = WallElement(
                element_id=self.next_id,
                element_type=DISPOSAL_CENTER,
                color=DEFAULT_DISPOSAL_CENTER_COLOR,
                top_left=top_left,
                bottom_right=bottom_right,
            )

        self.elements.append(element)
        self.next_id += 1
        return element

    def replace_from_json(self, data: Dict) -> Tuple[int, int]:
        canvas = data.get("canvas", {})
        json_width = int(canvas.get("width", self.canvas_width))
        json_height = int(canvas.get("height", self.canvas_height))

        # 用 JSON 里的尺寸更新画布，加载元素时不再裁剪
        self.canvas_width = json_width
        self.canvas_height = json_height

        self.elements = load_elements(data, json_width, json_height)
        self.next_id = max((element.element_id for element in self.elements), default=0) + 1
        self.reset_first_point()

        return json_width, json_height

    def to_json(self) -> Dict:
        return serialize_elements(self.canvas_width, self.canvas_height, self.elements)


def run_pyside6_app() -> bool:
    try:
        from PySide6.QtCore import QPoint, QRect, Qt, QTimer
        from PySide6.QtGui import (
            QAction,
            QBrush,
            QColor,
            QFont,
            QFontDatabase,
            QPainter,
            QPen,
        )
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QRadioButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        return False

    def configure_chinese_font(app: QApplication) -> None:
        font_paths = [
            "/mnt/c/Windows/Fonts/msyh.ttc",
            "/mnt/c/Windows/Fonts/msyhbd.ttc",
            "/mnt/c/Windows/Fonts/simhei.ttf",
            "/mnt/c/Windows/Fonts/simsun.ttc",
            "/mnt/c/Windows/Fonts/Deng.ttf",
            "/mnt/c/Windows/Fonts/msjhl.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]

        for font_path in font_paths:
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id < 0:
                continue

            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                app.setFont(QFont(families[0], 10))
                return

    class CanvasWidget(QWidget):
        def __init__(self, state: EditorState, set_status) -> None:
            super().__init__()
            self.state = state
            self.set_status = set_status
            self.preview_point: Optional[Tuple[int, int]] = None
            self.setFixedSize(state.canvas_width, state.canvas_height)
            self.setAutoFillBackground(True)
            self.setMouseTracking(True)

        def paintEvent(self, event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.fillRect(self.rect(), QBrush(QColor("white")))

            for element in self.state.elements:
                self._paint_element(painter, element)

            if self.state.first_point is not None and self.preview_point is not None:
                preview_color = QColor("#444444")
                x0, y0 = self.state.first_point
                x1, y1 = self.preview_point
                if self.state.current_tool == LINE_WALL:
                    pen = QPen(preview_color, DEFAULT_LINE_THICKNESS, Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(QPoint(x0, y0), QPoint(x1, y1))
                elif _is_rect_like_tool(self.state.current_tool):
                    pen = QPen(preview_color, 1, Qt.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    top_left, bottom_right = _normalize_rect(x0, y0, x1, y1)
                    rect = QRect(
                        top_left[0],
                        top_left[1],
                        bottom_right[0] - top_left[0],
                        bottom_right[1] - top_left[1],
                    )
                    painter.drawRect(rect)

        def _find_element_at(self, x: int, y: int) -> Optional[WallElement]:
            """从后往前（上层优先）查找点击位置的元素。"""
            for element in reversed(self.state.elements):
                if _hit_test(element, x, y):
                    return element
            return None

        def mousePressEvent(self, event) -> None:  # noqa: N802
            x = _clamp(int(event.position().x()), 0, self.state.canvas_width - 1)
            y = _clamp(int(event.position().y()), 0, self.state.canvas_height - 1)

            # 右键：删除点击位置的元素
            if event.button() == Qt.RightButton:
                hit = self._find_element_at(x, y)
                if hit is not None:
                    self.state.elements.remove(hit)
                    if self.state.selected_id == hit.element_id:
                        self.state.selected_id = None
                    self.set_status(f"已删除元素 #{hit.element_id}，当前总数 {len(self.state.elements)}")
                    self.update()
                return

            if event.button() != Qt.LeftButton:
                return

            # 选择工具：点击选中元素
            if self.state.current_tool == SELECT:
                hit = self._find_element_at(x, y)
                if hit is not None:
                    self.state.selected_id = hit.element_id
                    self.set_status(f"已选中元素 #{hit.element_id}（{hit.element_type}）")
                else:
                    self.state.selected_id = None
                    self.set_status("已取消选择")
                self.update()
                if hasattr(self, "on_selection_changed"):
                    self.on_selection_changed()
                return

            # 绘制工具：正常的两点绘制逻辑
            if self.state.first_point is None:
                self.state.first_point = (x, y)
                self.preview_point = (x, y)
                self.set_status(f"已记录首点 ({x}, {y})，请点击第二个点确认")
                self.update()
                return

            x0, y0 = self.state.first_point
            element = self.state.add_element(x0, y0, x, y)
            self.state.reset_first_point()
            self.preview_point = None

            if element is None:
                self.set_status("元素无效（长度/面积为 0），请重试")
            elif element.element_type == START_AREA:
                self.set_status(f"已设置起点区域（元素 #{element.element_id}），当前总数 {len(self.state.elements)}")
            elif element.element_type == DISPOSAL_CENTER:
                self.set_status(f"已设置 DisposalCenter（元素 #{element.element_id}），当前总数 {len(self.state.elements)}")
            else:
                self.set_status(f"已添加元素 #{element.element_id}，当前总数 {len(self.state.elements)}")
            self.update()

        def mouseMoveEvent(self, event) -> None:  # noqa: N802
            if self.state.first_point is None:
                return
            x = _clamp(int(event.position().x()), 0, self.state.canvas_width - 1)
            y = _clamp(int(event.position().y()), 0, self.state.canvas_height - 1)
            self.preview_point = (x, y)
            self.update()

        def _paint_element(self, painter: QPainter, element: WallElement) -> None:
            is_selected = (self.state.selected_id is not None
                           and element.element_id == self.state.selected_id)

            if element.element_type == LINE_WALL and element.points:
                thickness = element.thickness + 4 if is_selected else element.thickness
                color = QColor("#FF4500") if is_selected else QColor(element.color)
                pen = QPen(color, thickness, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(
                    QPoint(element.points[0][0], element.points[0][1]),
                    QPoint(element.points[1][0], element.points[1][1]),
                )
            elif element.element_type == RECT_WALL and element.top_left and element.bottom_right:
                c = QColor("#FF4500") if is_selected else QColor(element.color)
                painter.setPen(QPen(c, 1, Qt.SolidLine))
                painter.setBrush(QBrush(c))
                rect = QRect(
                    element.top_left[0],
                    element.top_left[1],
                    element.bottom_right[0] - element.top_left[0],
                    element.bottom_right[1] - element.top_left[1],
                )
                painter.drawRect(rect)
            elif element.element_type == START_AREA and element.top_left and element.bottom_right:
                c = QColor("#FF4500") if is_selected else QColor(element.color)
                fill = QColor(c)
                fill.setAlpha(70)
                painter.setPen(QPen(c, 3 if is_selected else 2, Qt.SolidLine))
                painter.setBrush(QBrush(fill))
                rect = QRect(
                    element.top_left[0],
                    element.top_left[1],
                    element.bottom_right[0] - element.top_left[0],
                    element.bottom_right[1] - element.top_left[1],
                )
                painter.drawRect(rect)
            elif element.element_type == DISPOSAL_CENTER and element.top_left and element.bottom_right:
                c = QColor("#FF4500") if is_selected else QColor(element.color)
                fill = QColor(c)
                fill.setAlpha(70)
                painter.setPen(QPen(c, 3 if is_selected else 2, Qt.SolidLine))
                painter.setBrush(QBrush(fill))
                rect = QRect(
                    element.top_left[0],
                    element.top_left[1],
                    element.bottom_right[0] - element.top_left[0],
                    element.bottom_right[1] - element.top_left[1],
                )
                painter.drawRect(rect)

    class MapWallEditorWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.state = EditorState()
            self.setWindowTitle("Map Wall Editor")

            container = QWidget()
            layout = QVBoxLayout(container)

            # ===== 第一行工具栏：工具按钮 + 文件操作 =====
            toolbar = QHBoxLayout()
            self.select_button = QRadioButton("选择")
            self.line_button = QRadioButton("线段墙 (6px)")
            self.rect_button = QRadioButton("矩形墙")
            self.start_area_button = QRadioButton("起点区域")
            self.disposal_center_button = QRadioButton("DisposalCenter")
            self.select_button.setChecked(True)
            self.select_button.toggled.connect(self._select_select_tool)
            self.line_button.toggled.connect(self._select_line_tool)
            self.rect_button.toggled.connect(self._select_rect_tool)
            self.start_area_button.toggled.connect(self._select_start_area_tool)
            self.disposal_center_button.toggled.connect(self._select_disposal_center_tool)

            save_button = QPushButton("保存 JSON")
            load_button = QPushButton("加载 JSON")
            export_png_button = QPushButton("导出 PNG")
            save_button.clicked.connect(self.save_json)
            load_button.clicked.connect(self.load_json_file)
            export_png_button.clicked.connect(self.export_png)

            toolbar.addWidget(self.select_button)
            toolbar.addWidget(self.line_button)
            toolbar.addWidget(self.rect_button)
            toolbar.addWidget(self.start_area_button)
            toolbar.addWidget(self.disposal_center_button)
            toolbar.addSpacing(20)
            toolbar.addWidget(save_button)
            toolbar.addWidget(load_button)
            toolbar.addWidget(export_png_button)
            toolbar.addStretch()

            # 画布尺寸调整
            toolbar.addWidget(QLabel("画布宽:"))
            self.canvas_width_spin = self._make_spin_box()
            self.canvas_width_spin.setRange(100, 10000)
            self.canvas_width_spin.setValue(DEFAULT_CANVAS_WIDTH)
            toolbar.addWidget(self.canvas_width_spin)
            toolbar.addWidget(QLabel("高:"))
            self.canvas_height_spin = self._make_spin_box()
            self.canvas_height_spin.setRange(100, 10000)
            self.canvas_height_spin.setValue(DEFAULT_CANVAS_HEIGHT)
            toolbar.addWidget(self.canvas_height_spin)
            self.apply_canvas_size_button = QPushButton("应用尺寸")
            self.apply_canvas_size_button.clicked.connect(self._apply_canvas_size)
            toolbar.addWidget(self.apply_canvas_size_button)

            # ===== 第二行：属性面板 =====
            props_bar = QHBoxLayout()
            self.props_label = QLabel("未选中元素")
            props_bar.addWidget(self.props_label)

            # 线段墙属性
            self.props_line_x1_spin = self._make_spin_box()
            self.props_line_y1_spin = self._make_spin_box()
            self.props_line_x2_spin = self._make_spin_box()
            self.props_line_y2_spin = self._make_spin_box()
            self.props_line_len_label = QLabel("长度: 0")
            self.props_line_widgets = [
                QLabel("x1:"), self.props_line_x1_spin,
                QLabel("y1:"), self.props_line_y1_spin,
                QLabel("x2:"), self.props_line_x2_spin,
                QLabel("y2:"), self.props_line_y2_spin,
                self.props_line_len_label,
            ]
            for w in self.props_line_widgets:
                w.hide()
                props_bar.addWidget(w)

            # 矩形类属性（矩形墙/起点区域/DisposalCenter）——对角两点坐标
            self.props_rect_x1_spin = self._make_spin_box()
            self.props_rect_y1_spin = self._make_spin_box()
            self.props_rect_x2_spin = self._make_spin_box()
            self.props_rect_y2_spin = self._make_spin_box()
            self.props_rect_widgets = [
                QLabel("x1:"), self.props_rect_x1_spin,
                QLabel("y1:"), self.props_rect_y1_spin,
                QLabel("x2:"), self.props_rect_x2_spin,
                QLabel("y2:"), self.props_rect_y2_spin,
            ]
            for w in self.props_rect_widgets:
                w.hide()
                props_bar.addWidget(w)

            # 删除按钮
            self.delete_button = QPushButton("删除选中")
            self.delete_button.clicked.connect(self._delete_selected)
            self.delete_button.hide()
            props_bar.addWidget(self.delete_button)

            props_bar.addStretch()

            self.status_label = QLabel("就绪：请选择工具并左键两次添加元素（右键可删除墙体）")
            self.canvas = CanvasWidget(self.state, self.set_status)
            self.canvas.on_selection_changed = self._on_selection_changed

            layout.addLayout(toolbar)
            layout.addLayout(props_bar)
            layout.addWidget(self.status_label)
            layout.addWidget(self.canvas)
            layout.addStretch(1)  # 画布缩小后，多余空间留底部，避免工具栏被撑开
            self.setCentralWidget(container)

            quit_action = QAction("Exit", self)
            quit_action.triggered.connect(self.close)

        def _make_spin_box(self):
            from PySide6.QtWidgets import QSpinBox
            spin = QSpinBox()
            spin.setRange(-10000, 10000)
            spin.setMaximumWidth(70)
            return spin

        def _select_select_tool(self, checked: bool) -> None:
            if checked:
                self.state.current_tool = SELECT
                self.state.reset_first_point()
                self.canvas.preview_point = None
                self.canvas.update()
                self.set_status("当前工具：选择（左键选中元素，右键删除元素）")

        def _select_line_tool(self, checked: bool) -> None:
            if checked:
                self.state.current_tool = LINE_WALL
                self.state.reset_first_point()
                self.canvas.preview_point = None
                self.canvas.update()
                self.set_status("当前工具：线段墙 (6px)")

        def _select_rect_tool(self, checked: bool) -> None:
            if checked:
                self.state.current_tool = RECT_WALL
                self.state.reset_first_point()
                self.canvas.preview_point = None
                self.canvas.update()
                self.set_status("当前工具：矩形墙")

        def _select_start_area_tool(self, checked: bool) -> None:
            if checked:
                self.state.current_tool = START_AREA
                self.state.reset_first_point()
                self.canvas.preview_point = None
                self.canvas.update()
                self.set_status("当前工具：起点区域（两次点击确定矩形，会替换已有起点区域）")

        def _select_disposal_center_tool(self, checked: bool) -> None:
            if checked:
                self.state.current_tool = DISPOSAL_CENTER
                self.state.reset_first_point()
                self.canvas.preview_point = None
                self.canvas.update()
                self.set_status("当前工具：DisposalCenter（两次点击确定矩形，会替换已有 DisposalCenter）")

        def set_status(self, text: str) -> None:
            self.status_label.setText(text)

        def _apply_canvas_size(self) -> None:
            new_w = self.canvas_width_spin.value()
            new_h = self.canvas_height_spin.value()
            old_w = self.state.canvas_width
            old_h = self.state.canvas_height
            if new_w == old_w and new_h == old_h:
                return
            self.state.resize_canvas(new_w, new_h)
            self.canvas.setFixedSize(new_w, new_h)
            self._on_selection_changed()
            self.canvas.update()
            # 等布局更新完成后再调整窗口大小，避免慢一拍
            QTimer.singleShot(0, self.adjustSize)
            note = "（尺寸减小，元素已裁剪）" if (new_w < old_w or new_h < old_h) else ""
            self.set_status(f"画布尺寸已调整为 {new_w}×{new_h}{note}，当前元素数 {len(self.state.elements)}")

        def _hide_all_props(self) -> None:
            """隐藏所有属性输入框。"""
            for w in self.props_line_widgets:
                w.hide()
            for w in self.props_rect_widgets:
                w.hide()
            self.delete_button.hide()

        def _show_line_props(self, element: WallElement) -> None:
            self.props_label.setText(f"线段墙 #{element.element_id}")
            # 先断开信号避免循环
            for spin in [self.props_line_x1_spin, self.props_line_y1_spin,
                         self.props_line_x2_spin, self.props_line_y2_spin]:
                try:
                    spin.valueChanged.disconnect()
                except RuntimeError:
                    pass
            x1, y1 = element.points[0]
            x2, y2 = element.points[1]
            self.props_line_x1_spin.setValue(x1)
            self.props_line_y1_spin.setValue(y1)
            self.props_line_x2_spin.setValue(x2)
            self.props_line_y2_spin.setValue(y2)
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            self.props_line_len_label.setText(f"长度: {length:.1f}")
            for w in self.props_line_widgets:
                w.show()
            # 重新连接信号
            self.props_line_x1_spin.valueChanged.connect(lambda v: self._update_line_point(0, 0, v))
            self.props_line_y1_spin.valueChanged.connect(lambda v: self._update_line_point(0, 1, v))
            self.props_line_x2_spin.valueChanged.connect(lambda v: self._update_line_point(1, 0, v))
            self.props_line_y2_spin.valueChanged.connect(lambda v: self._update_line_point(1, 1, v))
            self.delete_button.show()

        def _show_rect_props(self, element: WallElement) -> None:
            type_names = {
                RECT_WALL: "矩形墙",
                START_AREA: "起点区域",
                DISPOSAL_CENTER: "DisposalCenter",
            }
            self.props_label.setText(f"{type_names.get(element.element_type, '元素')} #{element.element_id}")
            for spin in [self.props_rect_x1_spin, self.props_rect_y1_spin,
                         self.props_rect_x2_spin, self.props_rect_y2_spin]:
                try:
                    spin.valueChanged.disconnect()
                except RuntimeError:
                    pass
            x1 = element.top_left[0]
            y1 = element.top_left[1]
            x2 = element.bottom_right[0]
            y2 = element.bottom_right[1]
            self.props_rect_x1_spin.setValue(x1)
            self.props_rect_y1_spin.setValue(y1)
            self.props_rect_x2_spin.setValue(x2)
            self.props_rect_y2_spin.setValue(y2)
            for w in self.props_rect_widgets:
                w.show()
            self.props_rect_x1_spin.valueChanged.connect(lambda v: self._update_rect_corner(0, 0, v))
            self.props_rect_y1_spin.valueChanged.connect(lambda v: self._update_rect_corner(0, 1, v))
            self.props_rect_x2_spin.valueChanged.connect(lambda v: self._update_rect_corner(1, 0, v))
            self.props_rect_y2_spin.valueChanged.connect(lambda v: self._update_rect_corner(1, 1, v))
            self.delete_button.show()

        def _on_selection_changed(self) -> None:
            self._hide_all_props()
            elem = self.state.get_selected()
            if elem is None:
                self.props_label.setText("未选中元素")
                return
            if elem.element_type == LINE_WALL:
                self._show_line_props(elem)
            elif elem.top_left is not None:
                self._show_rect_props(elem)

        def _update_line_point(self, pt_idx: int, coord_idx: int, value: int) -> None:
            elem = self.state.get_selected()
            if elem is None or elem.element_type != LINE_WALL or not elem.points:
                return
            p = list(elem.points[pt_idx])
            p[coord_idx] = value
            elem.points[pt_idx] = tuple(p)
            # 更新长度显示
            x1, y1 = elem.points[0]
            x2, y2 = elem.points[1]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            self.props_line_len_label.setText(f"长度: {length:.1f}")
            self.canvas.update()

        def _update_rect_corner(self, corner_idx: int, coord_idx: int, value: int) -> None:
            """修改矩形的一个角坐标。corner_idx: 0=top_left, 1=bottom_right。coord_idx: 0=x, 1=y。"""
            elem = self.state.get_selected()
            if elem is None or not elem.top_left or not elem.bottom_right:
                return
            tl = list(elem.top_left)
            br = list(elem.bottom_right)
            if corner_idx == 0:
                tl[coord_idx] = value
            else:
                br[coord_idx] = value
            elem.top_left = (min(tl[0], br[0]), min(tl[1], br[1]))
            elem.bottom_right = (max(tl[0], br[0]), max(tl[1], br[1]))
            self.canvas.update()

        def _delete_selected(self) -> None:
            elem = self.state.get_selected()
            if elem is None:
                return
            self.state.elements.remove(elem)
            self.state.selected_id = None
            self._on_selection_changed()
            self.canvas.update()
            self.set_status(f"已删除元素 #{elem.element_id}，当前总数 {len(self.state.elements)}")

        def save_json(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "保存 JSON", "", "JSON files (*.json)")
            if not path:
                return

            if not path.lower().endswith(".json"):
                path += ".json"

            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_json(), f, indent=2, ensure_ascii=False)
            self.set_status(f"JSON 已保存：{path}")

        def load_json_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "加载 JSON", "", "JSON files (*.json)")
            if not path:
                return

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                QMessageBox.critical(self, "加载失败", f"无法读取 JSON：{exc}")
                return

            old_w, old_h = self.state.canvas_width, self.state.canvas_height
            json_width, json_height = self.state.replace_from_json(data)

            # 画布尺寸变化时，调整画布大小和窗口大小
            if json_width != old_w or json_height != old_h:
                self.canvas.setFixedSize(json_width, json_height)
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, self.adjustSize)

            self.state.selected_id = None
            self._on_selection_changed()
            self.canvas.preview_point = None
            self.canvas.update()
            # 同步尺寸输入框
            self.canvas_width_spin.setValue(json_width)
            self.canvas_height_spin.setValue(json_height)
            size_note = f"（画布尺寸：{json_width}×{json_height}）" if (json_width != old_w or json_height != old_h) else ""
            self.set_status(f"已加载 {len(self.state.elements)} 个元素{size_note}：{path}")

        def export_png(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "导出 PNG", "", "PNG files (*.png)")
            if not path:
                return

            if not path.lower().endswith(".png"):
                path += ".png"

            image = render_elements_to_image(
                self.state.canvas_width,
                self.state.canvas_height,
                self.state.elements,
            )
            cv2.imwrite(path, image)
            self.set_status(f"PNG 已导出：{path}")

    app = QApplication([])
    configure_chinese_font(app)
    window = MapWallEditorWindow()
    window.show()
    app.exec()
    return True


def run_tkinter_fallback_app() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    use_chinese_text = sys.platform.startswith("win")
    labels = {
        "select": "选择" if use_chinese_text else "Select",
        "line_wall": "线段墙 (6px)" if use_chinese_text else "Line Wall (6px)",
        "rect_wall": "矩形墙" if use_chinese_text else "Rect Wall",
        "start_area": "起点区域" if use_chinese_text else "Start area",
        "disposal_center": "DisposalCenter" if use_chinese_text else "Disposal center",
        "delete_selected": "删除选中" if use_chinese_text else "Delete selected",
        "no_selection": "未选中元素" if use_chinese_text else "No selection",
        "deleted": "已删除元素" if use_chinese_text else "Deleted element",
        "selected": "已选中" if use_chinese_text else "Selected",
        "length": "长度" if use_chinese_text else "Length",
        "save_json": "保存 JSON" if use_chinese_text else "Save JSON",
        "load_json": "加载 JSON" if use_chinese_text else "Load JSON",
        "export_png": "导出 PNG" if use_chinese_text else "Export PNG",
        "png_exported": "PNG 已导出" if use_chinese_text else "PNG exported",
        "export_title": "导出 PNG" if use_chinese_text else "Export PNG",
        "ready": "就绪：请选择工具并左键两次添加元素"
        if use_chinese_text
        else "Ready: choose a tool and left-click twice",
        "current_tool": "当前工具" if use_chinese_text else "Current tool",
        "first_point": "已记录首点" if use_chinese_text else "First point",
        "second_point": "请点击第二个点确认" if use_chinese_text else "click the second point",
        "invalid": "元素无效（长度/面积为 0）" if use_chinese_text else "Invalid element: zero length/area",
        "added": "已添加元素" if use_chinese_text else "Added element",
        "total": "当前总数" if use_chinese_text else "total",
        "save_title": "保存 JSON" if use_chinese_text else "Save JSON",
        "load_title": "加载 JSON" if use_chinese_text else "Load JSON",
        "json_saved": "JSON 已保存" if use_chinese_text else "JSON saved",
        "load_failed": "加载失败" if use_chinese_text else "Load failed",
        "cannot_read": "无法读取 JSON" if use_chinese_text else "Cannot read JSON",
        "size_mismatch": "尺寸不匹配" if use_chinese_text else "Size mismatch",
        "canvas_size": "画布尺寸" if use_chinese_text else "canvas size",
        "canvas_width": "宽:" if use_chinese_text else "W:",
        "canvas_height": "高:" if use_chinese_text else "H:",
        "apply_size": "应用尺寸" if use_chinese_text else "Apply size",
        "size_applied": "画布尺寸已调整为" if use_chinese_text else "Canvas resized to",
        "elements_clipped": "（尺寸减小，元素已裁剪）" if use_chinese_text else " (clipped)",
        "loaded": "已加载" if use_chinese_text else "Loaded",
        "elements": "个元素" if use_chinese_text else "elements",
        "line_walls": "线段墙" if use_chinese_text else "line walls",
        "rect_walls_count": "矩形墙" if use_chinese_text else "rect walls",
    }

    class TkCanvasApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.state = EditorState()
            self.preview_item: Optional[int] = None
            self._suppress_props_callback = False  # 防止属性输入框回调循环

            root.title("Map Wall Editor")

            # ===== 第一行工具栏 =====
            toolbar = tk.Frame(root, padx=8, pady=8)
            toolbar.pack(fill=tk.X)

            self.tool_var = tk.StringVar(value=SELECT)
            tk.Radiobutton(
                toolbar,
                text=labels["select"],
                variable=self.tool_var,
                value=SELECT,
                command=self._set_tool,
                indicatoron=False,
                width=8,
            ).pack(side=tk.LEFT, padx=2)
            tk.Radiobutton(
                toolbar,
                text=labels["line_wall"],
                variable=self.tool_var,
                value=LINE_WALL,
                command=self._set_tool,
                indicatoron=False,
                width=14,
            ).pack(side=tk.LEFT, padx=2)
            tk.Radiobutton(
                toolbar,
                text=labels["rect_wall"],
                variable=self.tool_var,
                value=RECT_WALL,
                command=self._set_tool,
                indicatoron=False,
                width=10,
            ).pack(side=tk.LEFT, padx=2)
            tk.Radiobutton(
                toolbar,
                text=labels["start_area"],
                variable=self.tool_var,
                value=START_AREA,
                command=self._set_tool,
                indicatoron=False,
                width=12,
            ).pack(side=tk.LEFT, padx=2)
            tk.Radiobutton(
                toolbar,
                text=labels["disposal_center"],
                variable=self.tool_var,
                value=DISPOSAL_CENTER,
                command=self._set_tool,
                indicatoron=False,
                width=12,
            ).pack(side=tk.LEFT, padx=2)
            tk.Button(toolbar, text=labels["save_json"], command=self.save_json).pack(side=tk.LEFT, padx=6)
            tk.Button(toolbar, text=labels["load_json"], command=self.load_json_file).pack(side=tk.LEFT, padx=6)
            tk.Button(toolbar, text=labels["export_png"], command=self.export_png).pack(side=tk.LEFT, padx=6)

            # 画布尺寸调整
            tk.Label(toolbar, text=labels["canvas_width"]).pack(side=tk.RIGHT, padx=2)
            self.canvas_width_var = tk.StringVar(value=str(DEFAULT_CANVAS_WIDTH))
            self.canvas_width_entry = tk.Entry(toolbar, textvariable=self.canvas_width_var, width=6)
            self.canvas_width_entry.pack(side=tk.RIGHT, padx=2)
            tk.Label(toolbar, text=labels["canvas_height"]).pack(side=tk.RIGHT, padx=2)
            self.canvas_height_var = tk.StringVar(value=str(DEFAULT_CANVAS_HEIGHT))
            self.canvas_height_entry = tk.Entry(toolbar, textvariable=self.canvas_height_var, width=6)
            self.canvas_height_entry.pack(side=tk.RIGHT, padx=2)
            tk.Button(toolbar, text=labels["apply_size"], command=self._apply_canvas_size).pack(side=tk.RIGHT, padx=6)

            # ===== 第二行：属性面板 =====
            props_bar = tk.Frame(root, padx=8, pady=4)
            props_bar.pack(fill=tk.X)

            self.props_label_var = tk.StringVar(value=labels["no_selection"])
            tk.Label(props_bar, textvariable=self.props_label_var, width=20, anchor="w").pack(side=tk.LEFT)

            # 线段墙属性
            self.props_line_vars = {k: tk.StringVar(value="0") for k in ["x1", "y1", "x2", "y2"]}
            self.props_line_len_var = tk.StringVar(value="")
            self.props_line_entries = {}
            self.props_line_labels = {}
            for i, key in enumerate(["x1", "y1", "x2", "y2"]):
                lbl = tk.Label(props_bar, text=f"{key}:")
                ent = tk.Entry(props_bar, textvariable=self.props_line_vars[key], width=6)
                ent.bind("<Return>", lambda e, k=key: self._on_line_prop_change(k))
                self.props_line_labels[key] = lbl
                self.props_line_entries[key] = ent
            self.props_len_label = tk.Label(props_bar, textvariable=self.props_line_len_var)

            # 矩形类属性——对角两点坐标
            self.props_rect_vars = {k: tk.StringVar(value="0") for k in ["x1", "y1", "x2", "y2"]}
            self.props_rect_entries = {}
            self.props_rect_labels = {}
            for key in ["x1", "y1", "x2", "y2"]:
                lbl = tk.Label(props_bar, text=f"{key}:")
                ent = tk.Entry(props_bar, textvariable=self.props_rect_vars[key], width=6)
                ent.bind("<Return>", lambda e, k=key: self._on_rect_prop_change(k))
                self.props_rect_labels[key] = lbl
                self.props_rect_entries[key] = ent

            # 删除按钮
            self.delete_btn = tk.Button(props_bar, text=labels["delete_selected"], command=self._delete_selected)

            self._hide_all_props()

            self.status_var = tk.StringVar(value=labels["ready"])
            tk.Label(root, textvariable=self.status_var, anchor="w", padx=8).pack(fill=tk.X)

            self.canvas = tk.Canvas(
                root,
                width=self.state.canvas_width,
                height=self.state.canvas_height,
                bg="white",
                highlightthickness=1,
                highlightbackground="#dddddd",
            )
            self.canvas.pack(padx=8, pady=8)
            self.canvas.bind("<Button-1>", self.on_left_click)
            self.canvas.bind("<Button-3>", self.on_right_click)
            self.canvas.bind("<Motion>", self.on_mouse_move)

        def _find_element_at(self, x: int, y: int) -> Optional[WallElement]:
            for element in reversed(self.state.elements):
                if _hit_test(element, x, y):
                    return element
            return None

        def _set_tool(self) -> None:
            self.state.current_tool = self.tool_var.get()
            self.state.reset_first_point()
            self._clear_preview()
            tool = self.state.current_tool
            if tool == SELECT:
                tool_name = labels["select"]
            elif tool == LINE_WALL:
                tool_name = labels["line_wall"]
            elif tool == RECT_WALL:
                tool_name = labels["rect_wall"]
            elif tool == START_AREA:
                tool_name = labels["start_area"]
            else:
                tool_name = labels["disposal_center"]
            self.status_var.set(f"{labels['current_tool']}: {tool_name}")
            self._redraw_all()

        def _clear_preview(self) -> None:
            if self.preview_item is not None:
                self.canvas.delete(self.preview_item)
                self.preview_item = None

        def _draw_element(self, element: WallElement) -> None:
            is_selected = (self.state.selected_id is not None
                           and element.element_id == self.state.selected_id)
            sel_color = "#FF4500"

            if element.element_type == LINE_WALL and element.points:
                color = sel_color if is_selected else element.color
                width = element.thickness + 4 if is_selected else element.thickness
                self.canvas.create_line(
                    element.points[0][0],
                    element.points[0][1],
                    element.points[1][0],
                    element.points[1][1],
                    fill=color,
                    width=width,
                )
            elif element.element_type == RECT_WALL and element.top_left and element.bottom_right:
                color = sel_color if is_selected else element.color
                self.canvas.create_rectangle(
                    element.top_left[0],
                    element.top_left[1],
                    element.bottom_right[0],
                    element.bottom_right[1],
                    outline=color,
                    fill=color,
                )
            elif element.element_type == START_AREA and element.top_left and element.bottom_right:
                color = sel_color if is_selected else element.color
                w = 3 if is_selected else 2
                self.canvas.create_rectangle(
                    element.top_left[0],
                    element.top_left[1],
                    element.bottom_right[0],
                    element.bottom_right[1],
                    outline=color,
                    width=w,
                )
            elif element.element_type == DISPOSAL_CENTER and element.top_left and element.bottom_right:
                color = sel_color if is_selected else element.color
                w = 3 if is_selected else 2
                self.canvas.create_rectangle(
                    element.top_left[0],
                    element.top_left[1],
                    element.bottom_right[0],
                    element.bottom_right[1],
                    outline=color,
                    width=w,
                )

        def _redraw_all(self) -> None:
            self.canvas.delete("all")
            for element in self.state.elements:
                self._draw_element(element)
            self._clear_preview()

        def on_right_click(self, event: tk.Event) -> None:
            x = _clamp(int(event.x), 0, self.state.canvas_width - 1)
            y = _clamp(int(event.y), 0, self.state.canvas_height - 1)
            hit = self._find_element_at(x, y)
            if hit is not None:
                self.state.elements.remove(hit)
                if self.state.selected_id == hit.element_id:
                    self.state.selected_id = None
                    self._update_props_panel()
                self._redraw_all()
                self.status_var.set(f"{labels['deleted']} #{hit.element_id}; {labels['total']} {len(self.state.elements)}")

        def on_left_click(self, event: tk.Event) -> None:
            x = _clamp(int(event.x), 0, self.state.canvas_width - 1)
            y = _clamp(int(event.y), 0, self.state.canvas_height - 1)

            # 选择工具
            if self.state.current_tool == SELECT:
                hit = self._find_element_at(x, y)
                if hit is not None:
                    self.state.selected_id = hit.element_id
                else:
                    self.state.selected_id = None
                self._redraw_all()
                self._update_props_panel()
                return

            # 绘制工具
            if self.state.first_point is None:
                self.state.first_point = (x, y)
                self.status_var.set(f"{labels['first_point']}: ({x}, {y}); {labels['second_point']}")
                return

            x0, y0 = self.state.first_point
            element = self.state.add_element(x0, y0, x, y)
            self.state.reset_first_point()
            self._clear_preview()

            if element is None:
                self.status_var.set(labels["invalid"])
                return

            self._redraw_all()
            self.status_var.set(f"{labels['added']} #{element.element_id}; {labels['total']} {len(self.state.elements)}")

        def on_mouse_move(self, event: tk.Event) -> None:
            if self.state.first_point is None:
                return

            x = _clamp(int(event.x), 0, self.state.canvas_width - 1)
            y = _clamp(int(event.y), 0, self.state.canvas_height - 1)
            self._clear_preview()

            if self.state.current_tool == LINE_WALL:
                self.preview_item = self.canvas.create_line(
                    self.state.first_point[0],
                    self.state.first_point[1],
                    x,
                    y,
                    fill="#444444",
                    width=DEFAULT_LINE_THICKNESS,
                    dash=(4, 2),
                )
            elif _is_rect_like_tool(self.state.current_tool):
                self.preview_item = self.canvas.create_rectangle(
                    self.state.first_point[0],
                    self.state.first_point[1],
                    x,
                    y,
                    outline="#444444",
                    dash=(4, 2),
                )

        def _hide_all_props(self) -> None:
            for key in ["x1", "y1", "x2", "y2"]:
                self.props_line_labels[key].pack_forget()
                self.props_line_entries[key].pack_forget()
            self.props_len_label.pack_forget()
            for key in ["x1", "y1", "x2", "y2"]:
                self.props_rect_labels[key].pack_forget()
                self.props_rect_entries[key].pack_forget()
            self.delete_btn.pack_forget()

        def _update_props_panel(self) -> None:
            self._hide_all_props()
            elem = self.state.get_selected()
            if elem is None:
                self.props_label_var.set(labels["no_selection"])
                return

            self._suppress_props_callback = True
            if elem.element_type == LINE_WALL and elem.points:
                type_names = {LINE_WALL: labels["line_wall"]}
                self.props_label_var.set(f"{type_names.get(elem.element_type, '')} #{elem.element_id}")
                x1, y1 = elem.points[0]
                x2, y2 = elem.points[1]
                self.props_line_vars["x1"].set(str(x1))
                self.props_line_vars["y1"].set(str(y1))
                self.props_line_vars["x2"].set(str(x2))
                self.props_line_vars["y2"].set(str(y2))
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                self.props_line_len_var.set(f"{labels['length']}: {length:.1f}")
                for key in ["x1", "y1", "x2", "y2"]:
                    self.props_line_labels[key].pack(side=tk.LEFT, padx=2)
                    self.props_line_entries[key].pack(side=tk.LEFT, padx=2)
                self.props_len_label.pack(side=tk.LEFT, padx=8)
            elif elem.top_left is not None and elem.bottom_right is not None:
                type_names = {
                    RECT_WALL: labels["rect_wall"],
                    START_AREA: labels["start_area"],
                    DISPOSAL_CENTER: labels["disposal_center"],
                }
                self.props_label_var.set(f"{type_names.get(elem.element_type, '')} #{elem.element_id}")
                x1 = elem.top_left[0]
                y1 = elem.top_left[1]
                x2 = elem.bottom_right[0]
                y2 = elem.bottom_right[1]
                self.props_rect_vars["x1"].set(str(x1))
                self.props_rect_vars["y1"].set(str(y1))
                self.props_rect_vars["x2"].set(str(x2))
                self.props_rect_vars["y2"].set(str(y2))
                for key in ["x1", "y1", "x2", "y2"]:
                    self.props_rect_labels[key].pack(side=tk.LEFT, padx=2)
                    self.props_rect_entries[key].pack(side=tk.LEFT, padx=2)
            self.delete_btn.pack(side=tk.LEFT, padx=10)
            self._suppress_props_callback = False

        def _on_line_prop_change(self, key: str) -> None:
            if self._suppress_props_callback:
                return
            elem = self.state.get_selected()
            if elem is None or elem.element_type != LINE_WALL or not elem.points:
                return
            try:
                val = int(self.props_line_vars[key].get())
            except ValueError:
                return
            pt_idx = 0 if key in ("x1", "y1") else 1
            coord_idx = 0 if key in ("x1", "x2") else 1
            p = list(elem.points[pt_idx])
            p[coord_idx] = val
            elem.points[pt_idx] = tuple(p)
            # 更新长度
            x1, y1 = elem.points[0]
            x2, y2 = elem.points[1]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            self.props_line_len_var.set(f"{labels['length']}: {length:.1f}")
            self._redraw_all()

        def _on_rect_prop_change(self, key: str) -> None:
            if self._suppress_props_callback:
                return
            elem = self.state.get_selected()
            if elem is None or not elem.top_left or not elem.bottom_right:
                return
            try:
                val = int(self.props_rect_vars[key].get())
            except ValueError:
                return
            corner_idx = 0 if key in ("x1", "y1") else 1
            coord_idx = 0 if key in ("x1", "x2") else 1
            tl = list(elem.top_left)
            br = list(elem.bottom_right)
            if corner_idx == 0:
                tl[coord_idx] = val
            else:
                br[coord_idx] = val
            # 重新归一化，保证 top_left 始终在左上
            elem.top_left = (min(tl[0], br[0]), min(tl[1], br[1]))
            elem.bottom_right = (max(tl[0], br[0]), max(tl[1], br[1]))
            self._redraw_all()

        def _delete_selected(self) -> None:
            elem = self.state.get_selected()
            if elem is None:
                return
            self.state.elements.remove(elem)
            self.state.selected_id = None
            self._update_props_panel()
            self._redraw_all()
            self.status_var.set(f"{labels['deleted']} #{elem.element_id}; {labels['total']} {len(self.state.elements)}")

        def save_json(self) -> None:
            path = filedialog.asksaveasfilename(
                title=labels["save_title"],
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
            )
            if not path:
                return

            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_json(), f, indent=2, ensure_ascii=False)
            self.status_var.set(f"{labels['json_saved']}: {path}")

        def load_json_file(self) -> None:
            path = filedialog.askopenfilename(title=labels["load_title"], filetypes=[("JSON files", "*.json")])
            if not path:
                return

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                messagebox.showerror(labels["load_failed"], f"{labels['cannot_read']}: {exc}")
                return

            old_w, old_h = self.state.canvas_width, self.state.canvas_height
            json_width, json_height = self.state.replace_from_json(data)

            # 画布尺寸变化时，调整画布大小和窗口大小
            if json_width != old_w or json_height != old_h:
                self.canvas.config(width=json_width, height=json_height)
                self.root.update_idletasks()
                self.root.geometry("")

            self.state.selected_id = None
            self._update_props_panel()
            self._redraw_all()
            # 同步尺寸输入框
            self.canvas_width_var.set(str(json_width))
            self.canvas_height_var.set(str(json_height))
            size_note = (
                f"（{labels['canvas_size']}: {json_width}×{json_height}）"
                if (json_width != old_w or json_height != old_h)
                else ""
            )
            self.status_var.set(
                f"{labels['loaded']} {len(self.state.elements)} {labels['elements']}{size_note}: {path}"
            )

        def export_png(self) -> None:
            path = filedialog.asksaveasfilename(
                title=labels["export_title"],
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
            )
            if not path:
                return

            image = render_elements_to_image(
                self.state.canvas_width,
                self.state.canvas_height,
                self.state.elements,
            )
            cv2.imwrite(path, image)
            self.status_var.set(f"{labels['png_exported']}: {path}")

        def _apply_canvas_size(self) -> None:
            try:
                new_w = int(self.canvas_width_var.get())
                new_h = int(self.canvas_height_var.get())
            except ValueError:
                return
            old_w = self.state.canvas_width
            old_h = self.state.canvas_height
            if new_w == old_w and new_h == old_h:
                return
            if new_w < 100 or new_h < 100:
                return
            self.state.resize_canvas(new_w, new_h)
            self.canvas.config(width=new_w, height=new_h)
            # 让顶层窗口自适应画布大小
            self.root.update_idletasks()
            self.root.geometry("")
            self._update_props_panel()
            self._redraw_all()
            clipped = labels["elements_clipped"] if (new_w < old_w or new_h < old_h) else ""
            self.status_var.set(
                f"{labels['size_applied']} {new_w}×{new_h}{clipped}; {labels['total']} {len(self.state.elements)}"
            )

    root = tk.Tk()
    TkCanvasApp(root)
    root.resizable(False, False)
    root.mainloop()


def main() -> None:
    if not run_pyside6_app():
        run_tkinter_fallback_app()


if __name__ == "__main__":
    main()

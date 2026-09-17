import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.tools import map_wall_editor as mwe


def test_serialize_and_load_start_area_disposal_center_roundtrip() -> None:
    data = {
        "version": 1,
        "canvas": {"width": 100, "height": 80, "background": "white"},
        "elements": [
            {
                "id": 1,
                "type": "start_area",
                "top_left": [10, 10],
                "bottom_right": [50, 40],
                "color": "#1E90FF",
            },
            {
                "id": 2,
                "type": "disposal_center",
                "top_left": [60, 20],
                "bottom_right": [90, 60],
                "color": "#228B22",
            },
        ],
    }
    elements = mwe.load_elements(data, width=100, height=80)
    assert len(elements) == 2
    assert elements[0].element_type == mwe.START_AREA
    assert elements[0].top_left == (10, 10)
    assert elements[0].bottom_right == (50, 40)
    assert elements[1].element_type == mwe.DISPOSAL_CENTER

    out = mwe.serialize_elements(100, 80, elements)
    assert out["elements"] == data["elements"]


def test_load_elements_keeps_last_singleton_when_duplicates() -> None:
    """起点区域与处置中心都是单例：重复时只保留最后一个。"""
    data = {
        "version": 1,
        "canvas": {"width": 100, "height": 80, "background": "white"},
        "elements": [
            {
                "id": 1,
                "type": "start_area",
                "top_left": [0, 0],
                "bottom_right": [10, 10],
                "color": "#1E90FF",
            },
            {
                "id": 2,
                "type": "start_area",
                "top_left": [20, 20],
                "bottom_right": [40, 40],
                "color": "#0000FF",
            },
            {
                "id": 3,
                "type": "disposal_center",
                "top_left": [50, 50],
                "bottom_right": [60, 60],
                "color": "#228B22",
            },
            {
                "id": 4,
                "type": "disposal_center",
                "top_left": [70, 20],
                "bottom_right": [90, 70],
                "color": "#006400",
            },
        ],
    }
    elements = mwe.load_elements(data, width=100, height=80)
    assert len(elements) == 2
    assert elements[0].element_type == mwe.START_AREA
    assert elements[0].top_left == (20, 20)
    assert elements[0].bottom_right == (40, 40)
    assert elements[1].element_type == mwe.DISPOSAL_CENTER
    assert elements[1].top_left == (70, 20)
    assert elements[1].bottom_right == (90, 70)


def test_load_elements_ignores_unknown_or_legacy_types() -> None:
    """旧版 'return_area' 等未知类型会被忽略（不产生元素、不报错）。"""
    data = {
        "version": 1,
        "canvas": {"width": 100, "height": 80, "background": "white"},
        "elements": [
            {
                "id": 1,
                "type": "return_area",
                "top_left": [0, 0],
                "bottom_right": [10, 10],
                "color": "#1E90FF",
            },
            {
                "id": 2,
                "type": "line_wall",
                "points": [[0, 10], [90, 10]],
                "thickness": 6,
                "color": "#000000",
            },
        ],
    }
    elements = mwe.load_elements(data, width=100, height=80)
    assert len(elements) == 1
    assert elements[0].element_type == mwe.LINE_WALL


def test_render_elements_to_image_draws_walls_only() -> None:
    """只有墙体参与渲染为障碍；起点区域空心、处置中心红块不当作黑色障碍。"""
    elements = [
        mwe.WallElement(1, mwe.LINE_WALL, points=[(0, 10), (99, 10)]),
        mwe.WallElement(
            2,
            mwe.START_AREA,
            color=mwe.DEFAULT_START_AREA_COLOR,
            top_left=(10, 20),
            bottom_right=(40, 40),
        ),
        mwe.WallElement(
            3,
            mwe.DISPOSAL_CENTER,
            color=mwe.DEFAULT_DISPOSAL_CENTER_COLOR,
            top_left=(60, 20),
            bottom_right=(90, 60),
        ),
    ]
    img = mwe.render_elements_to_image(100, 80, elements)
    # 线段墙在 (50, 10) 处画了黑色像素
    assert img[10, 50, 0] < 100
    # 起点区域是空心矩形：内部保持白色
    assert img[30, 25, 0] > 200
    # 处置中心是红色实心填充（BGR 中红色通道高、蓝色通道低）
    assert img[40, 75, 2] > 200
    assert img[40, 75, 0] < 50


def _line(p0, p1, eid=1, thickness=mwe.DEFAULT_LINE_THICKNESS):
    return mwe.WallElement(eid, mwe.LINE_WALL, points=[tuple(p0), tuple(p1)], thickness=thickness)


def _rect(tl, br, eid=1, etype=mwe.RECT_WALL):
    return mwe.WallElement(eid, etype, top_left=tuple(tl), bottom_right=tuple(br))


def test_add_element_line_requires_two_distinct_points() -> None:
    state = mwe.EditorState()
    state.current_tool = mwe.LINE_WALL
    assert state.add_element(10, 10, 10, 10) is None
    el = state.add_element(10, 10, 90, 90)
    assert el is not None
    assert el.points == [(10, 10), (90, 90)]


def test_add_element_rect_normalizes() -> None:
    state = mwe.EditorState()
    state.current_tool = mwe.RECT_WALL
    el = state.add_element(90, 10, 10, 90)
    assert el is not None
    assert el.top_left == (10, 10)
    assert el.bottom_right == (90, 90)


def test_add_singleton_replaces_existing() -> None:
    state = mwe.EditorState()
    state.current_tool = mwe.START_AREA
    state.add_element(0, 0, 10, 10)
    state.add_element(20, 20, 40, 40)
    assert len(state.elements) == 1
    assert state.elements[0].top_left == (20, 20)

    state.current_tool = mwe.DISPOSAL_CENTER
    state.add_element(0, 50, 10, 60)
    assert len(state.elements) == 2
    assert state.elements[1].element_type == mwe.DISPOSAL_CENTER


def test_editor_state_selection_and_delete() -> None:
    state = mwe.EditorState()
    state.current_tool = mwe.LINE_WALL
    el = state.add_element(10, 10, 90, 90)
    assert el is not None
    state.selected_id = el.element_id
    assert state.get_selected() is el

    # 删除（GUI 侧直接移出列表并清空选中）
    state.elements.remove(el)
    assert state.get_selected() is None

    # 序号从现存元素最大值之后继续
    state.current_tool = mwe.RECT_WALL
    el2 = state.add_element(0, 0, 5, 5)
    assert el2 is not None
    assert el2.element_id == el.element_id + 1


def test_editor_state_resize_clamps_and_trims() -> None:
    state = mwe.EditorState()
    state.current_tool = mwe.LINE_WALL
    line = state.add_element(0, 50, 200, 150)
    state.current_tool = mwe.RECT_WALL
    big = state.add_element(5, 5, 200, 100)
    inside = state.add_element(10, 10, 20, 20)
    assert line is not None and big is not None and inside is not None

    state.resize_canvas(100, 100)
    # 超出部分被夹回画布边界
    assert line.points == [(0, 50), (99, 99)]
    assert big.top_left == (5, 5)
    assert big.bottom_right == (99, 99)
    assert inside.top_left == (10, 10)
    assert inside.bottom_right == (20, 20)
    assert len(state.elements) == 3

    # 缩到 1x1 时所有元素退化为点/线，全部被丢弃
    state.resize_canvas(1, 1)
    assert state.elements == []

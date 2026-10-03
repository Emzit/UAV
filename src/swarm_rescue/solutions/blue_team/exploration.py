"""第一阶段的多机分区覆盖：只选择目标，不读取传感器或控制电机。

前沿、A* 和通信租约留给后续阶段。本模块仍保持文档规定的公共接口。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import ExplorationConfig
from .contracts import ExplorationDecision, ExplorationInput, NavigationGoal, WorldPoint


@dataclass
class _CoveragePoint:
    position: WorldPoint
    status: str = "PENDING"
    created_step: int = -1
    blocked_until: int = -1
    reached_streak: int = 0
    best_distance: float = math.inf
    last_progress_step: int = -1


class ExplorationManager:
    """为一架无人机维护确定性的蛇形巡航状态。"""

    def __init__(self, config: ExplorationConfig) -> None:
        if config.coverage_lane_max_spacing_px <= 0:
            raise ValueError("coverage lane spacing must be positive")
        if config.coverage_goal_hold_steps < 1:
            raise ValueError("coverage goal hold steps must be positive")
        if config.coverage_stall_steps < 1 or config.coverage_retry_steps < 1:
            raise ValueError("coverage timing values must be positive")
        self._config = config
        self.reset(None)

    def reset(self, world_size: Optional[WorldPoint]) -> None:
        self._world_size = world_size
        self._layout_key: Optional[Tuple[WorldPoint, int, int]] = None
        self._points: List[_CoveragePoint] = []
        self._active_index: Optional[int] = None
        self._last_step = -1
        self._last_decision = self._decision("HOLD", "not_initialized")

    def update(
        self,
        data: ExplorationInput,
        grid: Optional[Any] = None,
        routes: Optional[Any] = None,
    ) -> ExplorationDecision:
        """每步调用一次；grid/routes 是下一阶段的兼容参数。"""
        del grid, routes
        if data.step == self._last_step:
            return self._last_decision
        if data.step < self._last_step:
            raise ValueError("exploration step must be monotonically increasing")
        if self._last_step >= 0 and data.step > self._last_step + 1:
            # 控制器执行抓弹/返航时不会调用探索器；不要把暂停算作卡住。
            self._pause_progress(data.step)
        self._last_step = data.step

        if data.carrying or data.has_bomb_task:
            self._pause_progress(data.step)
            return self._publish(self._decision("SUSPENDED", "bomb_task_active"))
        if not self._config.exploration_enabled:
            self._pause_progress(data.step)
            return self._publish(self._decision("LOCAL_FALLBACK", "disabled"))
        if not self._valid_pose(data):
            self._pause_progress(data.step)
            return self._publish(self._decision("LOCAL_FALLBACK", "no_reliable_world_pose"))

        size = self._valid_size(data.world_size)
        if size is None:
            return self._publish(self._decision("LOCAL_FALLBACK", "invalid_world_size"))
        count = data.drone_count if isinstance(data.drone_count, int) else 1
        count = max(1, count)
        drone_id = data.drone_id if isinstance(data.drone_id, int) else 0
        if drone_id < 0 or drone_id >= count:
            return self._publish(self._decision("LOCAL_FALLBACK", "invalid_drone_id"))

        layout_key = (size, drone_id, count)
        if layout_key != self._layout_key:
            self._build_layout(size, drone_id, count, data.pose.position)
            self._layout_key = layout_key
        if not self._points:
            return self._publish(self._decision("LOCAL_FALLBACK", "no_coverage_points"))

        position = data.pose.position
        assert position is not None
        for _ in range(len(self._points)):
            index = self._next_available_index(data.step)
            if index is None:
                if all(point.status == "VISITED" for point in self._points):
                    return self._publish(self._decision("EXHAUSTED", "coverage_complete"))
                return self._publish(self._decision("HOLD", "all_points_temporarily_blocked"))

            point = self._points[index]
            distance = math.hypot(
                position[0] - point.position[0],
                position[1] - point.position[1],
            )
            if distance <= self._config.coverage_goal_radius_px:
                point.reached_streak += 1
                if point.reached_streak >= self._config.coverage_goal_hold_steps:
                    point.status = "VISITED"
                    self._active_index = None
                    continue
            else:
                point.reached_streak = 0

            if point.last_progress_step < 0:
                point.best_distance = distance
                point.last_progress_step = data.step
            elif point.best_distance - distance >= self._config.coverage_min_progress_px:
                point.best_distance = distance
                point.last_progress_step = data.step
            elif data.step - point.last_progress_step >= self._config.coverage_stall_steps:
                self._block(index, data.step)
                continue

            point.status = "ACTIVE"
            if point.created_step < 0:
                point.created_step = data.step
            goal = NavigationGoal(
                goal_id="coverage:world:d{}:p{}".format(drone_id, index),
                kind="EXPLORE_COVERAGE",
                position=point.position,
                tolerance=self._config.coverage_goal_radius_px,
                priority=0.5,
                created_step=point.created_step,
            )
            return self._publish(ExplorationDecision(
                status="ACTIVE",
                goal=goal,
                frontier_id=None,
                expected_gain=0.5,
                estimated_cost=1.0,
                lease_until_step=-1,
                reason="coverage_waypoint",
            ))

        return self._publish(self._decision("HOLD", "no_available_point"))

    def invalidate_route(self, goal_id: str, step: int) -> None:
        """规划器判定当前目标不可达时，暂时跳过它。"""
        if self._active_index is None:
            return
        if self._layout_key is None:
            return
        expected = "coverage:world:d{}:p{}".format(
            self._layout_key[1], self._active_index
        )
        if goal_id == expected:
            self._block(self._active_index, step)

    def lease_payload(self) -> Optional[Dict[str, object]]:
        """第一阶段没有前沿租约，不改变现有炸弹通信协议。"""
        return None

    def debug_snapshot(self) -> Dict[str, object]:
        return {
            "layout_key": self._layout_key,
            "active_index": self._active_index,
            "points": tuple((p.position, p.status, p.blocked_until)
                            for p in self._points),
            "last_decision": self._last_decision.status,
        }

    def _build_layout(
        self, size: WorldPoint, drone_id: int, count: int, position: WorldPoint
    ) -> None:
        width, height = size
        margin = self._config.coverage_margin_px
        if width <= 2 * margin or height <= 2 * margin:
            self._points = []
            self._active_index = None
            return
        strip_width = (width - 2 * margin) / count
        lane_count = max(2, math.ceil(strip_width /
                                      self._config.coverage_lane_max_spacing_px))
        strip_left = -width / 2 + margin + drone_id * strip_width
        lane_x = [strip_left + strip_width * (j + 0.5) / lane_count
                  for j in range(lane_count)]
        y_low = -height / 2 + margin
        y_high = height / 2 - margin
        candidates: List[List[WorldPoint]] = []
        for ordered_x in (lane_x, list(reversed(lane_x))):
            for first_low in (True, False):
                path: List[WorldPoint] = []
                for lane_index, x in enumerate(ordered_x):
                    low_first = first_low if lane_index % 2 == 0 else not first_low
                    ys = (y_low, y_high) if low_first else (y_high, y_low)
                    path.extend(((x, ys[0]), (x, ys[1])))
                candidates.append(path)
        route = min(candidates, key=lambda path: math.hypot(
            path[0][0] - position[0], path[0][1] - position[1]))
        self._points = [_CoveragePoint(point) for point in route]
        self._active_index = None

    def _next_available_index(self, step: int) -> Optional[int]:
        if self._active_index is not None:
            point = self._points[self._active_index]
            if point.status == "ACTIVE":
                return self._active_index
        for index, point in enumerate(self._points):
            if point.status == "VISITED":
                continue
            if point.blocked_until > step:
                continue
            if point.status == "BLOCKED_UNTIL":
                point.status = "PENDING"
                point.created_step = -1
                point.last_progress_step = -1
                point.best_distance = math.inf
            self._active_index = index
            return index
        self._active_index = None
        return None

    def _block(self, index: int, step: int) -> None:
        point = self._points[index]
        point.status = "BLOCKED_UNTIL"
        point.blocked_until = step + self._config.coverage_retry_steps
        point.last_progress_step = -1
        point.best_distance = math.inf
        point.reached_streak = 0
        self._active_index = None

    def _pause_progress(self, step: int) -> None:
        if self._active_index is not None:
            self._points[self._active_index].last_progress_step = step
            self._points[self._active_index].reached_streak = 0

    def _valid_pose(self, data: ExplorationInput) -> bool:
        pose = data.pose
        return (
            pose.valid
            and pose.frame_id == "world"
            and pose.position is not None
            and pose.heading is not None
            and data.step - pose.step in (0, 1, 2)
            and math.isfinite(pose.position_variance)
            and pose.position_variance <= self._config.pose_variance_max_px2
            and math.isfinite(pose.heading_variance)
            and pose.heading_variance
            <= self._config.pose_heading_variance_max_rad2
            and math.isfinite(pose.heading)
            and all(math.isfinite(v) for v in pose.position)
        )

    @staticmethod
    def _valid_size(size: Optional[WorldPoint]) -> Optional[WorldPoint]:
        if size is None or len(size) != 2:
            return None
        try:
            width, height = float(size[0]), float(size[1])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            return None
        return width, height

    @staticmethod
    def _decision(status: str, reason: str) -> ExplorationDecision:
        return ExplorationDecision(
            status=status,
            goal=None,
            frontier_id=None,
            expected_gain=0.0,
            estimated_cost=1.0,
            lease_until_step=-1,
            reason=reason,
        )

    def _publish(self, decision: ExplorationDecision) -> ExplorationDecision:
        self._last_decision = decision
        return decision

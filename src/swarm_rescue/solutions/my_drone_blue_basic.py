"""蓝队基础控制器：多机搜索、抓取、通信去冲突和回收。"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Optional

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import (
    DroneSemanticSensor,
)
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.solutions.my_drone_rescue_example import (
    MyDroneRescueExample,
    SemanticHit,
    WorldPoint,
)


class MyDroneBlueBasic(MyDroneRescueExample):
    """在官方蓝方示例之上实现一套可直接运行的多机基础策略。

    主要增强：
    1. 所有无人机都执行搜索和搬运，不再只有 0 号机工作；
    2. 记忆短期看到的炸弹和回收区，短暂丢失视野后仍可导航；
    3. 通过局部广播共享目标，并用无人机 id 解决目标冲突；
    4. 不再使用与某张地图绑定的固定回收坐标。

    抓取、近距离追踪、Lidar 避障和卡死恢复继续复用官方示例中已经
    验证过的实现。
    """

    _MESSAGE_KIND = "blue-basic-v1"

    # 同一个炸弹在不同传感器射线上会得到略有区别的表面坐标。
    # 因此目标匹配不能用坐标完全相等，而要使用距离阈值。
    _BOMB_CLUSTER_RADIUS = 32.0
    _CLAIM_MATCH_RADIUS = 58.0
    _CLEARED_MATCH_RADIUS = 68.0

    # 记忆和通信都是短期信息。过期后主动丢弃，避免永久追逐旧目标。
    _BOMB_MEMORY_TTL = 360
    _CLAIM_TTL = 45
    _LOCAL_SIGHTING_BROADCAST_TTL = 30
    _CLEARED_BROADCAST_TTL = 50
    _MAX_BOMB_MEMORIES = 32

    # 到达记忆坐标却仍看不到炸弹时，原地扫描一小段时间再放弃。
    _MEMORY_REACQUIRE_RADIUS = 28.0
    _MEMORY_SEARCH_STEPS = 12
    _MEMORY_SEARCH_ROTATION = 0.55

    # 多次观测回收区时做低通融合，降低 GPS 和语义测距噪声。
    _DISPOSAL_FILTER_ALPHA = 0.25

    # 探索时保存稀疏轨迹；抓到炸弹后反向沿轨迹返回。这样即使回收区和
    # 炸弹之间隔着长墙，也不需要读取地图真值或硬编码墙体坐标。
    _BREADCRUMB_SPACING = 28.0
    _BREADCRUMB_REACHED_RADIUS = 36.0
    _MAX_BREADCRUMBS = 600

    def __init__(
        self,
        identifier: Optional[int] = None,
        misc_data: Optional[MiscData] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)

        self._blue_step = 0
        self._home_position: Optional[WorldPoint] = None
        self._known_disposal: Optional[WorldPoint] = None

        # 元素格式为 (世界坐标, 本机最后更新时间)。
        self._bomb_memories: list[tuple[WorldPoint, int]] = []
        # peer_id -> (目标坐标, 本机收到消息的时刻, 是否正在携带)。
        self._peer_claims: dict[int, tuple[WorldPoint, int, bool]] = {}

        self._target_bomb: Optional[WorldPoint] = None
        self._target_last_seen_step = -1
        self._last_local_sighting: Optional[WorldPoint] = None
        self._last_local_sighting_step = -1
        self._last_cleared_bomb: Optional[WorldPoint] = None
        self._last_cleared_step = -1
        self._cleared_memories: list[tuple[WorldPoint, int]] = []
        self._memory_search_steps = 0
        self._was_carrying = False
        self._mode = "explore"
        self._breadcrumbs: list[WorldPoint] = []
        self._return_waypoints: list[WorldPoint] = []

        # 错开各机探索周期，并让奇偶编号无人机偏向不同转向方向。
        my_id = int(self.identifier or 0)
        cycle = self._EXPLORE_STRAIGHT_STEPS + self._EXPLORE_TURN_STEPS
        self._explore_offset = (my_id * 19) % cycle
        self._explore_tick = self._explore_offset
        self._preferred_turn_sign = 1.0 if my_id % 2 == 0 else -1.0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def define_message_for_all(self) -> dict[str, Any]:
        """广播少量任务状态；通信被禁用时算法仍可独立运行。"""
        position = self._point_from_sensor(self.measured_gps_position())
        sighting: Optional[WorldPoint] = None
        if (
            self._last_local_sighting is not None
            and self._blue_step - self._last_local_sighting_step
            <= self._LOCAL_SIGHTING_BROADCAST_TTL
            and not self._is_recently_cleared(self._last_local_sighting)
        ):
            sighting = self._last_local_sighting

        cleared: Optional[WorldPoint] = None
        if (
            self._last_cleared_bomb is not None
            and self._blue_step - self._last_cleared_step
            <= self._CLEARED_BROADCAST_TTL
        ):
            cleared = self._last_cleared_bomb

        return {
            "kind": self._MESSAGE_KIND,
            "id": int(self.identifier or 0),
            "step": self._blue_step,
            "position": position,
            "mode": self._mode,
            "carrying": bool(self.grasped_bombs()),
            "claim": self._target_bomb,
            "sighting": sighting,
            "cleared": cleared,
            "disposal": self._known_disposal,
        }

    def control(self) -> CommandsDict:
        self._blue_step += 1
        self._remember_home_position()
        self._ingest_peer_messages()
        visible_bombs = self._observe_local_semantics()
        self._prune_expired_knowledge()

        carrying = bool(self.grasped_bombs())
        just_grabbed = carrying and not self._was_carrying
        if self._was_carrying and not carrying:
            self._mark_current_target_cleared()
            self._breadcrumbs = []
            self._return_waypoints = []
            self._record_breadcrumb(force=True)
        elif just_grabbed:
            self._record_breadcrumb(force=True)
            self._return_waypoints = list(self._breadcrumbs)
        elif not carrying:
            self._record_breadcrumb()
        self._was_carrying = carrying

        cmd = self._empty_command()
        if carrying:
            self._mode = "carry"
            self._memory_search_steps = 0
            return self._control_carry(cmd)

        visible_target = self._choose_visible_bomb(visible_bombs)
        if visible_target is not None:
            self._mode = "chase"
            self._target_bomb = visible_target[1]
            self._target_last_seen_step = self._blue_step
            self._memory_search_steps = 0
            return self._control_chase_bomb(cmd, visible_target)

        if self._target_bomb is not None and self._claimed_by_preferred_peer(
            self._target_bomb
        ):
            self._target_bomb = None
            self._memory_search_steps = 0

        if self._target_bomb is None:
            self._target_bomb = self._choose_remembered_bomb()
            if self._target_bomb is not None:
                self._target_last_seen_step = self._blue_step

        if self._target_bomb is not None:
            self._mode = "navigate"
            return self._control_to_remembered_bomb(cmd)

        self._mode = "explore"
        self._memory_search_steps = 0
        return self._control_explore(cmd)

    # ------------------------------------------------------------------
    # 多机任务分工与探索
    # ------------------------------------------------------------------

    def _is_active_drone(self) -> bool:
        """蓝方基础策略让全部无人机参与任务。"""
        return True

    def _reset_explore_state(self) -> None:
        super()._reset_explore_state()
        self._explore_tick = self._explore_offset

    def _explore_turn_rate(self) -> float:
        path_min = self._lidar_path_min()
        if path_min is not None and self._lidar_is_blocking(path_min, None):
            return self._IDLE_SPIN * self._lidar_turn_sign()
        return self._IDLE_SPIN * self._preferred_turn_sign

    # ------------------------------------------------------------------
    # 本地观测和短期地图
    # ------------------------------------------------------------------

    def _observe_local_semantics(self) -> list[SemanticHit]:
        semantic = self.semantic_values()
        if semantic is None:
            return []

        visible: list[SemanticHit] = []
        for det in semantic:
            if det.entity_type == DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER:
                point = self._semantic_to_world(det)
                if point is not None:
                    self._remember_disposal(point)
                continue

            if det.entity_type != DroneSemanticSensor.TypeEntity.BOMB:
                continue
            if bool(det.grasped):
                continue
            distance = float(det.distance)
            if not math.isfinite(distance):
                continue
            point = self._semantic_to_world(det)
            if point is None or self._is_recently_cleared(point):
                continue

            self._remember_bomb(point)
            self._last_local_sighting = point
            self._last_local_sighting_step = self._blue_step
            self._append_clustered_hit(visible, (det, point))

        return visible

    def _append_clustered_hit(
        self, visible: list[SemanticHit], candidate: SemanticHit
    ) -> None:
        """将同一炸弹产生的多条语义射线合并为一个候选目标。"""
        _, point = candidate
        for index, old in enumerate(visible):
            old_det, old_point = old
            if self._distance(point, old_point) > self._BOMB_CLUSTER_RADIUS:
                continue
            if float(candidate[0].distance) < float(old_det.distance):
                visible[index] = candidate
            return
        visible.append(candidate)

    def _remember_bomb(self, point: WorldPoint) -> None:
        for index, (old_point, _) in enumerate(self._bomb_memories):
            if self._distance(point, old_point) <= self._BOMB_CLUSTER_RADIUS:
                # 适度平滑坐标，同时把信息刷新为当前时刻。
                fused = (
                    0.6 * old_point[0] + 0.4 * point[0],
                    0.6 * old_point[1] + 0.4 * point[1],
                )
                self._bomb_memories[index] = (fused, self._blue_step)
                return
        self._bomb_memories.append((point, self._blue_step))
        self._bomb_memories.sort(key=lambda item: item[1], reverse=True)
        del self._bomb_memories[self._MAX_BOMB_MEMORIES :]

    def _remove_bomb_memory_near(self, point: WorldPoint) -> None:
        self._bomb_memories = [
            item
            for item in self._bomb_memories
            if self._distance(item[0], point) > self._CLEARED_MATCH_RADIUS
        ]

    def _remember_disposal(self, point: WorldPoint) -> None:
        if self._known_disposal is None:
            self._known_disposal = point
            return
        alpha = self._DISPOSAL_FILTER_ALPHA
        self._known_disposal = (
            (1.0 - alpha) * self._known_disposal[0] + alpha * point[0],
            (1.0 - alpha) * self._known_disposal[1] + alpha * point[1],
        )

    def _remember_home_position(self) -> None:
        if self._home_position is not None:
            return
        self._home_position = self._point_from_sensor(
            self.measured_gps_position()
        )

    # ------------------------------------------------------------------
    # 通信和目标去冲突
    # ------------------------------------------------------------------

    def _ingest_peer_messages(self) -> None:
        for _, raw in self.communicator.received_messages:
            if not isinstance(raw, dict):
                continue
            if raw.get("kind") != self._MESSAGE_KIND:
                continue
            peer_id = raw.get("id")
            if not isinstance(peer_id, int):
                continue
            if peer_id == int(self.identifier or 0):
                continue

            disposal = self._valid_point(raw.get("disposal"))
            if disposal is not None:
                self._remember_disposal(disposal)

            cleared = self._valid_point(raw.get("cleared"))
            if cleared is not None:
                self._remember_cleared(cleared)
                self._remove_bomb_memory_near(cleared)
                if (
                    self._target_bomb is not None
                    and self._distance(self._target_bomb, cleared)
                    <= self._CLEARED_MATCH_RADIUS
                ):
                    self._target_bomb = None

            sighting = self._valid_point(raw.get("sighting"))
            if sighting is not None and not self._is_recently_cleared(sighting):
                self._remember_bomb(sighting)

            claim = self._valid_point(raw.get("claim"))
            if claim is None:
                self._peer_claims.pop(peer_id, None)
            else:
                carrying = bool(raw.get("carrying", False))
                self._peer_claims[peer_id] = (
                    claim,
                    self._blue_step,
                    carrying,
                )

    def _choose_visible_bomb(
        self, visible: list[SemanticHit]
    ) -> Optional[SemanticHit]:
        if not visible:
            return None

        # 优先保持当前目标，避免连续帧之间来回切换。
        if self._target_bomb is not None:
            current = sorted(
                visible,
                key=lambda hit: self._distance(hit[1], self._target_bomb),
            )[0]
            if (
                self._distance(current[1], self._target_bomb)
                <= self._CLAIM_MATCH_RADIUS
                and not self._claimed_by_preferred_peer(current[1])
            ):
                return current

        for candidate in sorted(visible, key=lambda hit: float(hit[0].distance)):
            if not self._claimed_by_preferred_peer(candidate[1]):
                return candidate
        return None

    def _choose_remembered_bomb(self) -> Optional[WorldPoint]:
        candidates = [
            point
            for point, _ in self._bomb_memories
            if not self._claimed_by_preferred_peer(point)
            and not self._is_recently_cleared(point)
        ]
        if not candidates:
            return None

        position = self._point_from_sensor(self.measured_gps_position())
        if position is None:
            return candidates[0]
        return min(candidates, key=lambda point: self._distance(position, point))

    def _claimed_by_preferred_peer(self, point: WorldPoint) -> bool:
        my_id = int(self.identifier or 0)
        for peer_id, (claim, received_step, carrying) in self._peer_claims.items():
            if self._blue_step - received_step > self._CLAIM_TTL:
                continue
            if self._distance(point, claim) > self._CLAIM_MATCH_RADIUS:
                continue
            # 已抓到炸弹的无人机拥有最高优先级；其余冲突由较小 id 获胜。
            if carrying or peer_id < my_id:
                return True
        return False

    def _prune_expired_knowledge(self) -> None:
        self._bomb_memories = [
            item
            for item in self._bomb_memories
            if self._blue_step - item[1] <= self._BOMB_MEMORY_TTL
            and not self._is_recently_cleared(item[0])
        ]
        self._peer_claims = {
            peer_id: value
            for peer_id, value in self._peer_claims.items()
            if self._blue_step - value[1] <= self._CLAIM_TTL
        }
        self._cleared_memories = [
            item
            for item in self._cleared_memories
            if self._blue_step - item[1] <= self._CLEARED_BROADCAST_TTL
        ]
        if (
            self._target_bomb is not None
            and self._blue_step - self._target_last_seen_step
            > self._BOMB_MEMORY_TTL
            and not self.grasped_bombs()
        ):
            self._target_bomb = None

    # ------------------------------------------------------------------
    # 记忆目标导航和投放
    # ------------------------------------------------------------------

    def _control_carry(self, cmd: CommandsDict) -> CommandsDict:
        """优先沿探索轨迹返航，看到回收区后交给精确投放逻辑。"""
        cmd["grasper"] = 1
        direct = self._find_closest_semantic(
            DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER
        )
        if direct is not None:
            self._remember_disposal(direct[1])
            return super()._control_carry(cmd)

        waypoint = self._next_return_waypoint()
        if waypoint is None:
            # 轨迹不可用时仍保留官方逻辑，并使用已记忆的回收区或出生点。
            return super()._control_carry(cmd)

        position = self._point_from_sensor(self.measured_gps_position())
        if position is None or self.measured_compass_angle() is None:
            return cmd
        distance = self._distance(position, waypoint)

        if self._run_lidar_escape_if_needed(cmd):
            cmd["grasper"] = 1
            return cmd
        self._move_toward(
            waypoint[0],
            waypoint[1],
            cmd,
            use_lidar=True,
            lidar_clearance=distance,
        )
        cmd["grasper"] = 1
        return cmd

    def _record_breadcrumb(self, *, force: bool = False) -> None:
        position = self._point_from_sensor(self.measured_gps_position())
        if position is None:
            return
        if (
            not force
            and self._breadcrumbs
            and self._distance(position, self._breadcrumbs[-1])
            < self._BREADCRUMB_SPACING
        ):
            return
        if (
            force
            and self._breadcrumbs
            and self._distance(position, self._breadcrumbs[-1]) < 3.0
        ):
            self._breadcrumbs[-1] = position
        else:
            self._breadcrumbs.append(position)

        if len(self._breadcrumbs) > self._MAX_BREADCRUMBS:
            # 始终保留第一个出生点，只移除第二老的普通轨迹点。
            del self._breadcrumbs[1]

    def _next_return_waypoint(self) -> Optional[WorldPoint]:
        position = self._point_from_sensor(self.measured_gps_position())
        if position is None:
            return None
        while self._return_waypoints:
            candidate = self._return_waypoints[-1]
            if (
                self._distance(position, candidate)
                > self._BREADCRUMB_REACHED_RADIUS
            ):
                return candidate
            self._return_waypoints.pop()
        return None

    def _control_to_remembered_bomb(self, cmd: CommandsDict) -> CommandsDict:
        target = self._target_bomb
        if target is None:
            return self._control_explore(cmd)

        cmd["grasper"] = 1
        position = self._point_from_sensor(self.measured_gps_position())
        heading = self.measured_compass_angle()
        if position is None or heading is None:
            return self._control_explore(cmd)

        distance = self._distance(position, target)
        if distance <= self._MEMORY_REACQUIRE_RADIUS:
            self._memory_search_steps += 1
            cmd["rotation"] = (
                self._MEMORY_SEARCH_ROTATION * self._preferred_turn_sign
            )
            if self._memory_search_steps >= self._MEMORY_SEARCH_STEPS:
                self._remove_bomb_memory_near(target)
                self._target_bomb = None
                self._memory_search_steps = 0
            return cmd

        self._memory_search_steps = 0
        if self._run_lidar_escape_if_needed(cmd):
            cmd["grasper"] = 1
            return cmd
        self._move_toward(
            target[0],
            target[1],
            cmd,
            use_lidar=True,
            lidar_clearance=distance,
        )
        cmd["grasper"] = 1
        return cmd

    def _mark_current_target_cleared(self) -> None:
        if self._target_bomb is None:
            return
        self._last_cleared_bomb = self._target_bomb
        self._last_cleared_step = self._blue_step
        self._remember_cleared(self._target_bomb)
        self._remove_bomb_memory_near(self._target_bomb)
        self._target_bomb = None
        self._memory_search_steps = 0

    def _disposal_target(self) -> WorldPoint:
        direct = self._find_closest_semantic(
            DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER
        )
        if direct is not None:
            self._remember_disposal(direct[1])
            return direct[1]
        if self._known_disposal is not None:
            return self._known_disposal
        # 内置地图把回收区放在出生区附近。未观测到回收区时先回出生点，
        # 进入语义传感器范围后便会改用真实观测，不依赖具体地图坐标。
        if self._home_position is not None:
            return self._home_position
        return (0.0, 0.0)

    # ------------------------------------------------------------------
    # 数据校验工具
    # ------------------------------------------------------------------

    def _is_recently_cleared(self, point: WorldPoint) -> bool:
        for cleared, step in self._cleared_memories:
            if self._blue_step - step > self._CLEARED_BROADCAST_TTL:
                continue
            if self._distance(point, cleared) <= self._CLEARED_MATCH_RADIUS:
                return True
        return False

    def _remember_cleared(self, point: WorldPoint) -> None:
        for index, (old_point, _) in enumerate(self._cleared_memories):
            if self._distance(point, old_point) <= self._CLEARED_MATCH_RADIUS:
                self._cleared_memories[index] = (point, self._blue_step)
                return
        self._cleared_memories.append((point, self._blue_step))

    @staticmethod
    def _distance(first: WorldPoint, second: WorldPoint) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _point_from_sensor(value: Any) -> Optional[WorldPoint]:
        if value is None:
            return None
        try:
            x = float(value[0])
            y = float(value[1])
        except (IndexError, TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return (x, y)

    @staticmethod
    def _valid_point(value: Any) -> Optional[WorldPoint]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None
        if not isinstance(value[0], Real) or not isinstance(value[1], Real):
            return None
        x = float(value[0])
        y = float(value[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return (x, y)

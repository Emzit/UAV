"""
蓝队示例控制器（team_mode: rescue）。

本文件用于提交示例，目标是：
1) 给选手一个可直接运行的最小救援策略；
2) 用中文说明常用 API 与返回值。

示例流程：搜索炸弹 -> 抓取 -> 运往回收区 -> 释放。

----------------------------
每个选手必须实现的核心方法
----------------------------
- define_message_for_all() -> Any
- control() -> CommandsDict

----------------------------
执行器命令 API（control 返回字典）
----------------------------
- "forward": float，范围 [-1, 1]
- "lateral": float，范围 [-1, 1]
- "rotation": float，范围 [-1, 1]
- "grasper": int，取值 {0, 1}
  - 1: 保持抓取/尝试抓取
  - 0: 释放当前抓取物

----------------------------
传感器便捷 API（比赛控制逻辑建议使用）
----------------------------
- measured_gps_position() -> np.ndarray | None
- measured_compass_angle() -> float | None
- measured_velocity() -> np.ndarray | None
- measured_angular_velocity() -> float | None
- semantic_values() -> list[DroneSemanticSensor.Data] | None
  Data 字段如下：
  - distance: float
  - angle: float
  - entity_type: DroneSemanticSensor.TypeEntity
    (WALL / BOMB / DISPOSAL_CENTER / DRONE / OTHER)
  - grasped: bool
- lidar_values() -> np.ndarray | None
- lidar_rays_angles() -> np.ndarray
- gps_values() -> np.ndarray | None
- compass_values() -> float | None
- odometer_values() -> np.ndarray | None

----------------------------
通信 API
----------------------------
- define_message_for_all(): 返回广播消息
- communicator.received_messages -> list[(sender_communicator, msg)]
- communicator_is_disabled() -> bool

----------------------------
状态与辅助 API
----------------------------
- drone_health -> int
- elapsed_timestep -> int
- elapsed_walltime -> float
- size_area -> tuple | None
- grasped_bombs() -> list[Bomb]
- carried_bombs_count() -> int（救援模式通常为 0）
- *_is_disabled() -> bool
- gps()/compass()/odometer()/lidar()/semantic() -> 传感器对象（高级用法）

说明：
- `true_*` 方法仅用于调试/日志，禁止在 `control()` 中使用；评测（competition mode）下在 `control()` 内调用会抛出 `GroundTruthInControlError`。
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.utils import normalize_angle

SemanticHit = Tuple[DroneSemanticSensor.Data, Tuple[float, float]]
WorldPoint = Tuple[float, float]


class MyDroneRescueExample(DroneAbstract):
    """
    蓝队最小救援示例：仅 id==0 执行完整任务，其余无人机原地慢转。

    状态机（由 control 依次判断）：
      idle -> carry -> chase -> explore
    """

    # --- 任务分工 ---
    _ACTIVE_DRONE_ID = 0
    _IDLE_SPIN = 0.35

    # --- 导航 ---
    _DISPOSAL_APPROACH: WorldPoint = (0.0, -165.0)
    _ARRIVE_RADIUS = 10.0
    _HEADING_TOLERANCE = 0.25

    # --- 运弹 / 处置 ---
    _DISPOSAL_PUSH_TRIGGER_DIST = 28.0
    _DISPOSAL_PUSH_STEPS = 18
    _DISPOSAL_PUSH_FORWARD = 0.75
    _DISPOSAL_SPIN_RELEASE_STEPS = 14
    _BOMB_FRONT_ALIGN_TOL = 0.16
    _BOMB_FRONT_ALIGN_ROT = 0.75
    _BOMB_GRAB_APPROACH_SPEED = 0.48
    _BOMB_GRAB_MIN_SPEED = 0.22

    # --- 探索 / 脱困 ---
    _EXPLORE_STRAIGHT_STEPS = 90
    _EXPLORE_TURN_STEPS = 28
    _EXPLORE_FORWARD = 0.7

    # --- Lidar 避障（阈值参考 my_drone_place_example）---
    _LIDAR_BLOCK_DIST = 35.0
    _LIDAR_SLOW_DIST = 52.0
    _LIDAR_PATH_CONE = math.pi / 9
    _LIDAR_SIDE_CONE = math.pi / 4
    _LIDAR_TARGET_MARGIN = 10.0
    _LIDAR_AVOID_ROTATION = 0.85
    _LIDAR_BLOCK_TRIGGER_STEPS = 6
    _LIDAR_ESCAPE_BACK_STEPS = 10
    _LIDAR_ESCAPE_TURN_STEPS = 16
    _LIDAR_ESCAPE_STRAFE_STEPS = 10
    _LIDAR_ESCAPE_BACK_SPEED = -0.55
    _LIDAR_ESCAPE_STRAFE_SPEED = 0.65

    def __init__(
        self,
        identifier: Optional[int] = None,
        misc_data: Optional[MiscData] = None,
        **kwargs,
    ):
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)
        self._explore_tick = 0
        self._carry_push_steps = 0
        self._carry_spin_release_steps_left = 0
        self._carry_spin_sign = 1.0
        self._lidar_blocked_steps = 0
        self._escape_back_steps_left = 0
        self._escape_turn_steps_left = 0
        self._escape_strafe_steps_left = 0
        self._escape_turn_sign = 1.0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def define_message_for_all(self) -> Any:
        return None

    def control(self) -> CommandsDict:
        cmd = self._empty_command()
        if not self._is_active_drone():
            return self._control_idle(cmd)
        if self.grasped_bombs():
            return self._control_carry(cmd)
        bomb = self._find_closest_semantic(
            DroneSemanticSensor.TypeEntity.BOMB,
            require_not_grasped=True,
        )
        if bomb is None:
            return self._control_explore(cmd)
        return self._control_chase_bomb(cmd, bomb)

    # ------------------------------------------------------------------
    # 控制阶段
    # ------------------------------------------------------------------

    def _is_active_drone(self) -> bool:
        return int(self.identifier or 0) == self._ACTIVE_DRONE_ID

    def _control_idle(self, cmd: CommandsDict) -> CommandsDict:
        my_id = int(self.identifier or 0)
        cmd["rotation"] = self._IDLE_SPIN if my_id % 2 == 0 else -self._IDLE_SPIN
        return cmd

    def _control_carry(self, cmd: CommandsDict) -> CommandsDict:
        cmd["grasper"] = 1
        target = self._disposal_target()
        near = self._within_radius(target, self._DISPOSAL_PUSH_TRIGGER_DIST)
        if self._should_start_carry_spin_release(near):
            self._start_carry_spin_release(self._lidar_turn_sign())

        if self._carry_spin_release_steps_left > 0:
            return self._control_carry_spin_release(cmd)

        if (not near) and self._run_lidar_escape_if_needed(cmd):
            return cmd

        if near or self._carry_push_steps > 0:
            return self._control_carry_push(cmd, target)

        arrived = self._move_toward(*target, cmd)
        if arrived:
            cmd["grasper"] = 1
        return cmd

    def _control_carry_push(self, cmd: CommandsDict, target: WorldPoint) -> CommandsDict:
        # 炸弹挂在机身后方：先推进处置区再释放，避免落在区外。
        if self._carry_spin_release_steps_left > 0:
            return self._control_carry_spin_release(cmd)

        if self._lidar_front_blocked():
            self._start_carry_spin_release(self._lidar_turn_sign())
            return self._control_carry_spin_release(cmd)

        self._carry_push_steps += 1
        self._move_toward(*target, cmd)
        cmd["forward"] = max(float(cmd.get("forward", 0.0)), self._DISPOSAL_PUSH_FORWARD)
        cmd["grasper"] = 1
        if self._carry_push_steps >= self._DISPOSAL_PUSH_STEPS:
            cmd["grasper"] = 0
        return cmd

    def _start_carry_spin_release(self, turn_sign: float) -> None:
        self._carry_spin_sign = 1.0 if turn_sign >= 0 else -1.0
        self._carry_spin_release_steps_left = self._DISPOSAL_SPIN_RELEASE_STEPS
        self._carry_push_steps = 0

    def _control_carry_spin_release(self, cmd: CommandsDict) -> CommandsDict:
        self._set_motion(
            cmd,
            forward=0.0,
            lateral=0.0,
            rotation=self._LIDAR_AVOID_ROTATION * self._carry_spin_sign,
        )
        cmd["grasper"] = 1
        self._carry_spin_release_steps_left -= 1
        if self._carry_spin_release_steps_left <= 0:
            cmd["grasper"] = 0
        return cmd

    def _should_start_carry_spin_release(self, near_disposal: bool) -> bool:
        if not near_disposal or self._carry_spin_release_steps_left > 0:
            return False
        return self._lidar_front_blocked()

    def _control_explore(self, cmd: CommandsDict) -> CommandsDict:
        self._carry_push_steps = 0
        if self._run_lidar_escape_if_needed(cmd):
            return cmd

        cycle = self._EXPLORE_STRAIGHT_STEPS + self._EXPLORE_TURN_STEPS
        phase = self._explore_tick % cycle
        self._explore_tick += 1

        if phase < self._EXPLORE_STRAIGHT_STEPS:
            cmd["forward"] = self._EXPLORE_FORWARD
            self._apply_lidar_avoidance(cmd)
        else:
            cmd["forward"] = 0.0
            cmd["rotation"] = self._explore_turn_rate()
        return cmd

    def _control_chase_bomb(self, cmd: CommandsDict, bomb: SemanticHit) -> CommandsDict:
        self._carry_push_steps = 0
        self._reset_explore_state()
        bomb_det, _ = bomb
        cmd["grasper"] = 1
        if self._run_lidar_escape_if_needed(cmd):
            return cmd
        self._approach_bomb_frontally(cmd, bomb_det)
        return cmd

    def _approach_bomb_frontally(
        self, cmd: CommandsDict, bomb_det: DroneSemanticSensor.Data
    ) -> None:
        """正面对齐后再接近炸弹，减少侧向抓取导致的回程不稳。"""
        ang = float(bomb_det.angle)
        dist = float(bomb_det.distance)
        if not math.isfinite(ang) or not math.isfinite(dist):
            return

        if abs(ang) > self._BOMB_FRONT_ALIGN_TOL:
            self._set_motion(
                cmd,
                forward=0.0,
                lateral=0.0,
                rotation=self._BOMB_FRONT_ALIGN_ROT if ang > 0 else -self._BOMB_FRONT_ALIGN_ROT,
            )
            return

        self._set_motion(
            cmd,
            forward=max(
            self._BOMB_GRAB_MIN_SPEED,
            min(self._BOMB_GRAB_APPROACH_SPEED, dist / 70.0),
            ),
            lateral=0.0,
            rotation=0.0,
        )
        self._apply_lidar_avoidance(cmd, path_clearance=dist)

    # ------------------------------------------------------------------
    # 语义感知
    # ------------------------------------------------------------------

    def _find_closest_semantic(
        self,
        entity_type: DroneSemanticSensor.TypeEntity,
        *,
        require_not_grasped: bool = False,
    ) -> Optional[SemanticHit]:
        semantic = self.semantic_values()
        if semantic is None:
            return None

        best: Optional[DroneSemanticSensor.Data] = None
        best_dist = float("inf")
        for obj in semantic:
            if obj.entity_type != entity_type:
                continue
            if require_not_grasped and bool(obj.grasped):
                continue
            d = float(obj.distance)
            if not math.isfinite(d) or d >= best_dist:
                continue
            best_dist = d
            best = obj

        if best is None:
            return None
        world = self._semantic_to_world(best)
        return (best, world) if world is not None else None

    def _semantic_to_world(
        self, det: DroneSemanticSensor.Data
    ) -> Optional[WorldPoint]:
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if pos is None or heading is None:
            return None
        bearing = float(heading) + float(det.angle)
        return (
            float(pos[0]) + float(det.distance) * math.cos(bearing),
            float(pos[1]) + float(det.distance) * math.sin(bearing),
        )

    def _disposal_target(self) -> WorldPoint:
        hit = self._find_closest_semantic(
            DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER
        )
        return hit[1] if hit is not None else self._DISPOSAL_APPROACH

    # ------------------------------------------------------------------
    # 导航
    # ------------------------------------------------------------------

    def _move_toward(
        self,
        target_x: float,
        target_y: float,
        cmd: CommandsDict,
        *,
        use_lidar: bool = False,
        lidar_clearance: Optional[float] = None,
    ) -> bool:
        """朝世界坐标 target 行驶；对齐后返回 False，到达返回 True。"""
        pos = self.measured_gps_position()
        heading = self.measured_compass_angle()
        if pos is None or heading is None:
            return False

        dx = target_x - float(pos[0])
        dy = target_y - float(pos[1])
        dist = math.hypot(dx, dy)
        if dist < self._ARRIVE_RADIUS:
            cmd["forward"] = 0.0
            cmd["lateral"] = 0.0
            cmd["rotation"] = 0.0
            return True

        desired = math.atan2(dy, dx)
        turn_err = normalize_angle(desired - float(heading))
        if abs(turn_err) > self._HEADING_TOLERANCE:
            cmd["rotation"] = 0.8 if turn_err > 0 else -0.8
            cmd["forward"] = 0.0
            cmd["lateral"] = 0.0
            return False

        cmd["rotation"] = 0.0
        cmd["lateral"] = 0.0
        cmd["forward"] = min(1.0, dist / 80.0 + 0.25)
        if use_lidar:
            clearance = dist if lidar_clearance is None else lidar_clearance
            self._apply_lidar_avoidance(cmd, path_clearance=clearance)
        return False

    def _within_radius(self, target: WorldPoint, radius: float) -> bool:
        pos = self.measured_gps_position()
        if pos is None:
            return False
        return math.hypot(float(pos[0]) - target[0], float(pos[1]) - target[1]) < radius

    # ------------------------------------------------------------------
    # 探索 / 脱困
    # ------------------------------------------------------------------

    def _reset_explore_state(self) -> None:
        self._explore_tick = 0
        self._lidar_blocked_steps = 0

    def _explore_turn_rate(self) -> float:
        spin = self._IDLE_SPIN
        path_min = self._lidar_path_min()
        if path_min is not None and self._lidar_is_blocking(path_min, None):
            spin *= self._lidar_turn_sign()
        return spin

    # ------------------------------------------------------------------
    # Lidar 避障
    # ------------------------------------------------------------------

    def _lidar_sector_min(self, angle_min: float, angle_max: float) -> Optional[float]:
        values = self.lidar_values()
        if values is None:
            return None
        best = float("inf")
        found = False
        for dist, ang in zip(values, self.lidar_rays_angles()):
            if not math.isfinite(dist):
                continue
            a = float(ang)
            if angle_min <= a <= angle_max:
                found = True
                best = min(best, float(dist))
        return best if found else None

    def _lidar_path_min(self) -> Optional[float]:
        return self._lidar_sector_min(-self._LIDAR_PATH_CONE, self._LIDAR_PATH_CONE)

    def _lidar_is_blocking(
        self, path_min: float, path_clearance: Optional[float]
    ) -> bool:
        """前方是墙（而非导航目标本身）时才需要避障。"""
        if path_min >= self._LIDAR_BLOCK_DIST:
            return False
        if path_clearance is None:
            return True
        return path_min < path_clearance - self._LIDAR_TARGET_MARGIN

    def _lidar_turn_sign(self) -> float:
        left = self._lidar_sector_min(0.08, self._LIDAR_SIDE_CONE)
        right = self._lidar_sector_min(-self._LIDAR_SIDE_CONE, -0.08)
        left_d = left if left is not None else float("inf")
        right_d = right if right is not None else float("inf")
        return 1.0 if left_d >= right_d else -1.0

    def _lidar_front_blocked(self) -> bool:
        path_min = self._lidar_path_min()
        return path_min is not None and path_min < self._LIDAR_BLOCK_DIST

    def _apply_lidar_avoidance(
        self,
        cmd: CommandsDict,
        *,
        path_clearance: Optional[float] = None,
    ) -> None:
        path_min = self._lidar_path_min()
        if path_min is None:
            self._lidar_blocked_steps = 0
            return
        fwd = float(cmd.get("forward", 0.0))
        blocked = self._lidar_is_blocking(path_min, path_clearance)
        if fwd <= 0.0 or not blocked:
            self._lidar_blocked_steps = 0
            return

        turn = self._lidar_turn_sign()
        self._lidar_blocked_steps += 1
        if (
            self._lidar_blocked_steps >= self._LIDAR_BLOCK_TRIGGER_STEPS
            and not self._lidar_escape_active()
        ):
            self._start_lidar_escape(turn)
            self._set_motion(
                cmd,
                forward=0.0,
                lateral=float(cmd.get("lateral", 0.0)),
                rotation=self._LIDAR_AVOID_ROTATION * turn,
            )
            return

        if path_min < self._LIDAR_BLOCK_DIST:
            self._set_motion(
                cmd,
                forward=0.0,
                lateral=float(cmd.get("lateral", 0.0)),
                rotation=self._LIDAR_AVOID_ROTATION * turn,
            )
            return

        span = self._LIDAR_SLOW_DIST - self._LIDAR_BLOCK_DIST
        scale = max(0.0, min(1.0, (path_min - self._LIDAR_BLOCK_DIST) / span))
        cmd["forward"] = fwd * scale
        if scale < 0.55:
            cmd["rotation"] = 0.45 * turn

    def _lidar_escape_active(self) -> bool:
        return (
            self._escape_back_steps_left > 0
            or self._escape_turn_steps_left > 0
            or self._escape_strafe_steps_left > 0
        )

    def _start_lidar_escape(self, turn_sign: float) -> None:
        self._escape_turn_sign = 1.0 if turn_sign >= 0 else -1.0
        self._escape_back_steps_left = self._LIDAR_ESCAPE_BACK_STEPS
        self._escape_turn_steps_left = self._LIDAR_ESCAPE_TURN_STEPS
        self._escape_strafe_steps_left = self._LIDAR_ESCAPE_STRAFE_STEPS
        self._lidar_blocked_steps = 0

    def _run_lidar_escape_if_needed(self, cmd: CommandsDict) -> bool:
        if not self._lidar_escape_active():
            return False
        if self._escape_back_steps_left > 0:
            self._escape_back_steps_left -= 1
            self._set_motion(
                cmd,
                forward=self._LIDAR_ESCAPE_BACK_SPEED,
                lateral=0.0,
                rotation=0.0,
            )
            return True
        if self._escape_turn_steps_left > 0:
            self._escape_turn_steps_left -= 1
            self._set_motion(
                cmd,
                forward=0.0,
                lateral=0.0,
                rotation=self._LIDAR_AVOID_ROTATION * self._escape_turn_sign,
            )
            return True
        if self._escape_strafe_steps_left > 0:
            self._escape_strafe_steps_left -= 1
            self._set_motion(
                cmd,
                forward=0.25,
                lateral=self._LIDAR_ESCAPE_STRAFE_SPEED * self._escape_turn_sign,
                rotation=0.0,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_command() -> CommandsDict:
        return {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0,
        }

    @staticmethod
    def _set_motion(
        cmd: CommandsDict, *, forward: float, lateral: float, rotation: float
    ) -> None:
        cmd["forward"] = forward
        cmd["lateral"] = lateral
        cmd["rotation"] = rotation

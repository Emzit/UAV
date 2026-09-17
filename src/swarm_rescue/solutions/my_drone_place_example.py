"""
红队示例控制器（team_mode: place）。

策略概要：
- 探索巡航 + Lidar 避障（忽略已放置炸弹，红队与炸弹无碰撞）+ 雷达脱困
- 周期性尝试放置炸弹
- 0 号机汇总全队库存后提交探索地图并结束回合

必实现方法：define_message_for_all()、control()。
传感器与执行器 API 见 DroneAbstract 文档及蓝队示例 my_drone_rescue_example.py。
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData

WorldPoint = Tuple[float, float]
TeamMessage = Tuple[int, Any, Any, int]  # (drone_id, gps, heading, inventory)


class MyDronePlaceExample(DroneAbstract):
    """红队最小示例：巡航放弹 + 0 号机提交探索矩阵。"""

    # --- 任务分工 ---
    _LEADER_DRONE_ID = 0
    _PLACE_BOMB_EVERY_STEPS = 60
    _INVENTORY_LOG_EVERY_STEPS = 60

    # --- 探索 / 脱困（与 my_drone_rescue_example 一致）---
    _EXPLORE_STRAIGHT_STEPS = 90
    _EXPLORE_TURN_STEPS = 28
    _EXPLORE_FORWARD = 0.7
    _ROAM_SPIN = 0.35

    # --- Lidar 避障 ---
    _LIDAR_BLOCK_DIST = 35.0
    _LIDAR_SLOW_DIST = 52.0
    _LIDAR_PATH_CONE = math.pi / 9
    _LIDAR_SIDE_CONE = math.pi / 4
    _LIDAR_SIDE_GAP = 0.08
    _LIDAR_TARGET_MARGIN = 10.0
    _LIDAR_AVOID_ROTATION = 0.85
    _LIDAR_BLOCK_TRIGGER_STEPS = 6
    _LIDAR_ESCAPE_BACK_STEPS = 10
    _LIDAR_ESCAPE_TURN_STEPS = 16
    _LIDAR_ESCAPE_STRAFE_STEPS = 10
    _LIDAR_ESCAPE_BACK_SPEED = -0.55
    _LIDAR_ESCAPE_STRAFE_SPEED = 0.65
    _LIDAR_BOMB_ANGLE_TOL = math.pi / 18
    _LIDAR_BOMB_DIST_TOL = 18.0

    def __init__(
        self,
        identifier: Optional[int] = None,
        misc_data: Optional[MiscData] = None,
        **kwargs,
    ):
        super().__init__(
            identifier=identifier,
            misc_data=misc_data,
            display_lidar_graph=False,
            **kwargs,
        )
        self._steps = 0
        self._explore_tick = 0
        self._lidar_blocked_steps = 0
        self._escape_back_steps_left = 0
        self._escape_turn_steps_left = 0
        self._escape_strafe_steps_left = 0
        self._escape_turn_sign = 1.0
        self._exploration_submitted = False
        self._peer_inventory: dict[int, int] = {}

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def define_message_for_all(self) -> TeamMessage:
        """广播本机 id、位姿与剩余可放置炸弹数。"""
        return (
            self.identifier,
            self.measured_gps_position(),
            self.measured_compass_angle(),
            self.carried_bombs_count(),
        )

    def control(self) -> CommandsDict:
        self._steps += 1
        cmd = self._control_roam()
        self._maybe_place_bomb(cmd)
        if self._is_leader():
            self._leader_try_submit_exploration_map()
        return cmd

    # ------------------------------------------------------------------
    # 放置炸弹
    # ------------------------------------------------------------------

    def _maybe_place_bomb(self, cmd: CommandsDict) -> None:
        if self._steps % self._PLACE_BOMB_EVERY_STEPS != 0:
            return
        if self.carried_bombs_count() > 0:
            cmd["place_bomb"] = 1

    # ------------------------------------------------------------------
    # 探索地图提交（仅 0 号机）
    # ------------------------------------------------------------------

    def _is_leader(self) -> bool:
        return int(self.identifier or 0) == self._LEADER_DRONE_ID

    def _leader_try_submit_exploration_map(self) -> None:
        if self._exploration_submitted:
            return

        self._sync_peer_inventory_from_messages()

        if self._steps % self._INVENTORY_LOG_EVERY_STEPS == 0:
            self._log_team_inventory_status()

        if not self._all_teammates_report_zero_inventory():
            return
        if self.size_area is None:
            return

        self._log_team_inventory_status()
        print("[Drone0] 全队炸弹已放完，提交探索地图并结束回合")
        self._submit_random_exploration_map()
        self._exploration_submitted = True

    def _submit_random_exploration_map(self) -> None:
        w, h = self.size_area
        grid = np.random.randint(0, 2, size=(h, w), dtype=np.uint8)
        self.submit_exploration_map(grid)

    def _sync_peer_inventory_from_messages(self) -> None:
        self._peer_inventory[int(self.identifier)] = self.carried_bombs_count()
        for _, msg in self.communicator.received_messages:
            if not isinstance(msg, tuple) or len(msg) < 4:
                continue
            drone_id, inv = msg[0], msg[3]
            if isinstance(drone_id, int) and isinstance(inv, (int, float)):
                self._peer_inventory[drone_id] = int(inv)

    def _team_size(self) -> Optional[int]:
        if self._misc_data is None:
            return None
        return self._misc_data.number_drones

    def _all_teammates_report_zero_inventory(self) -> bool:
        n = self._team_size()
        if n is None:
            return False
        for i in range(n):
            if i not in self._peer_inventory or self._peer_inventory[i] != 0:
                return False
        return True

    def _log_team_inventory_status(self) -> None:
        n = self._team_size()
        if n is None:
            print(
                f"[Drone0] 提交等待: number_drones 未知, "
                f"已记录={dict(sorted(self._peer_inventory.items()))}"
            )
            return

        missing: list[int] = []
        not_zero: list[int] = []
        lines: list[str] = []
        for i in range(n):
            if i not in self._peer_inventory:
                lines.append(f"  #{i}: 未收到消息")
                missing.append(i)
            else:
                inv = self._peer_inventory[i]
                lines.append(f"  #{i}: {inv}")
                if inv != 0:
                    not_zero.append(i)

        ready = self._all_teammates_report_zero_inventory()
        in_range = len(self.communicator.comms_in_range)
        print(
            f"[Drone0] 库存汇总 step={self.elapsed_timestep} "
            f"已确认={len(self._peer_inventory)}/{n} "
            f"通信范围内={in_range} 可提交={ready}\n"
            + "\n".join(lines)
        )
        if missing:
            print(f"[Drone0] 仍缺消息: {missing}")
        if not_zero:
            print(f"[Drone0] 仍有库存: {not_zero}")

    # ------------------------------------------------------------------
    # 巡航（探索 + 脱困 + Lidar）
    # ------------------------------------------------------------------

    def _control_roam(self) -> CommandsDict:
        cmd = self._empty_command()
        if self._run_lidar_escape_if_needed(cmd):
            return cmd

        cycle = self._EXPLORE_STRAIGHT_STEPS + self._EXPLORE_TURN_STEPS
        if self._explore_tick % cycle < self._EXPLORE_STRAIGHT_STEPS:
            cmd["forward"] = self._EXPLORE_FORWARD
            self._apply_lidar_avoidance(cmd)
        else:
            cmd["rotation"] = self._explore_turn_rate()
        self._explore_tick += 1
        return cmd

    def _roam_spin_sign(self) -> float:
        return self._ROAM_SPIN if int(self.identifier or 0) % 2 == 0 else -self._ROAM_SPIN

    def _explore_turn_rate(self) -> float:
        spin = self._roam_spin_sign()
        path_min = self._lidar_path_min()
        if path_min is not None and self._lidar_is_blocking(path_min, None):
            spin *= self._lidar_turn_sign()
        return spin

    # ------------------------------------------------------------------
    # Lidar 避障（红队：已放置炸弹不计入障碍）
    # ------------------------------------------------------------------

    def _lidar_path_min(self) -> Optional[float]:
        return self._lidar_sector_min(-self._LIDAR_PATH_CONE, self._LIDAR_PATH_CONE)

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
            if not (angle_min <= a <= angle_max):
                continue
            d = float(dist)
            if self._lidar_ray_hits_bomb(d, a):
                continue
            found = True
            best = min(best, d)
        return best if found else None

    def _lidar_ray_hits_bomb(self, dist: float, ang: float) -> bool:
        """语义确认为 BOMB 的 Lidar 读数忽略（放置阶段无碰撞）。"""
        semantic = self.semantic_values()
        if semantic is None:
            return False
        for obj in semantic:
            if obj.entity_type != DroneSemanticSensor.TypeEntity.BOMB:
                continue
            if bool(obj.grasped):
                continue
            if abs(float(obj.angle) - ang) > self._LIDAR_BOMB_ANGLE_TOL:
                continue
            if abs(float(obj.distance) - dist) <= self._LIDAR_BOMB_DIST_TOL:
                return True
        return False

    def _lidar_is_blocking(
        self, path_min: float, path_clearance: Optional[float]
    ) -> bool:
        if path_min >= self._LIDAR_BLOCK_DIST:
            return False
        if path_clearance is None:
            return True
        return path_min < path_clearance - self._LIDAR_TARGET_MARGIN

    def _lidar_turn_sign(self) -> float:
        left = self._lidar_sector_min(self._LIDAR_SIDE_GAP, self._LIDAR_SIDE_CONE)
        right = self._lidar_sector_min(-self._LIDAR_SIDE_CONE, -self._LIDAR_SIDE_GAP)
        left_d = left if left is not None else float("inf")
        right_d = right if right is not None else float("inf")
        return 1.0 if left_d >= right_d else -1.0

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
            cmd["forward"] = 0.0
            cmd["rotation"] = self._LIDAR_AVOID_ROTATION * turn
            return

        if path_min < self._LIDAR_BLOCK_DIST:
            cmd["forward"] = 0.0
            cmd["rotation"] = self._LIDAR_AVOID_ROTATION * turn
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
            cmd["forward"] = self._LIDAR_ESCAPE_BACK_SPEED
            cmd["lateral"] = 0.0
            cmd["rotation"] = 0.0
            return True
        if self._escape_turn_steps_left > 0:
            self._escape_turn_steps_left -= 1
            cmd["forward"] = 0.0
            cmd["lateral"] = 0.0
            cmd["rotation"] = self._LIDAR_AVOID_ROTATION * self._escape_turn_sign
            return True
        if self._escape_strafe_steps_left > 0:
            self._escape_strafe_steps_left -= 1
            cmd["forward"] = 0.25
            cmd["lateral"] = self._LIDAR_ESCAPE_STRAFE_SPEED * self._escape_turn_sign
            cmd["rotation"] = 0.0
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
            "place_bomb": 0,
        }

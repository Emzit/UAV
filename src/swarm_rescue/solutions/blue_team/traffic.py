"""Local right-of-way and deadlock recovery without simulator ground truth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .contracts import PoseEstimate, WorldPoint


@dataclass(frozen=True)
class PeerMotion:
    peer_id: int
    position: WorldPoint
    carrying: bool
    received_step: int
    heading: Optional[float] = None


@dataclass(frozen=True)
class TrafficAction:
    forward: float
    lateral: float
    rotation: float
    reason: str


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class TrafficManager:
    """Carrier-first traffic rule with a bounded, lidar-guarded escape."""

    def __init__(self, own_id: int) -> None:
        self.own_id = own_id
        self._yield_until = -1
        self._yield_started = -1
        self._recovery_until = -1
        self._recovery_started = -1
        self._reference_position: Optional[WorldPoint] = None
        self._reference_goal: Optional[WorldPoint] = None
        self._reference_mode = ""
        self._reference_step = -1
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_heading = 0.0

    def reset(self) -> None:
        self.__init__(self.own_id)

    @staticmethod
    def _sector_min(distances: Optional[Sequence[float]],
                    angles: Optional[Sequence[float]],
                    center: float, half_width: float = 0.35) -> float:
        if distances is None or angles is None:
            return math.inf
        values = []
        for raw_distance, raw_angle in zip(distances, angles):
            try:
                distance, angle = float(raw_distance), float(raw_angle)
            except (TypeError, ValueError):
                continue
            if (math.isfinite(distance) and
                    abs(_wrap(angle - center)) <= half_width):
                values.append(distance)
        return min(values, default=math.inf)

    def _escape_motion(self, distances: Optional[Sequence[float]],
                       angles: Optional[Sequence[float]], step: int,
                       reason: str) -> TrafficAction:
        if distances is None or angles is None:
            turn = -0.6 if self.own_id % 2 == 0 else 0.6
            return TrafficAction(0.0, 0.0, turn, reason + "_no_lidar")
        right = self._sector_min(distances, angles, -math.pi / 2)
        left = self._sector_min(distances, angles, math.pi / 2)
        rear = self._sector_min(distances, angles, math.pi)
        # Both head-on drones naturally separate if each takes its own right.
        # Switch sides after a prolonged yield instead of endlessly repeating
        # an ineffective movement against a wall.
        prefer_right = not (
            reason == "yield" and self._yield_started >= 0 and
            step - self._yield_started >= 20
        )
        side = -1.0 if prefer_right else 1.0
        best = right if prefer_right else left
        other = left if prefer_right else right
        if best < 55.0 and other >= 55.0:
            side = -side
            best = other
        if best >= 55.0:
            return TrafficAction(-0.22 if rear >= 50.0 else 0.0,
                                 0.62 * side, 0.0, reason)
        if rear >= 55.0:
            return TrafficAction(-0.48, 0.0, 0.0, reason)
        return TrafficAction(0.0, 0.0, 0.7 * side, reason + "_rotate")

    def _is_stalled(self, pose: PoseEstimate, step: int, mode: str,
                    goal: Optional[WorldPoint],
                    odometry_values: Optional[Sequence[float]]) -> bool:
        if (not pose.valid or pose.position is None or goal is None or
                math.dist(pose.position, goal) <= 60.0 or
                mode not in ("carry", "navigate", "explore", "chase")):
            self._reference_position = None
            self._odom_x = self._odom_y = self._odom_heading = 0.0
            return False
        if (self._reference_position is None or self._reference_goal is None or
                self._reference_mode != mode or
                math.dist(self._reference_goal, goal) > 35.0):
            self._reference_position = pose.position
            self._reference_goal = goal
            self._reference_mode = mode
            self._reference_step = step
            self._odom_x = self._odom_y = self._odom_heading = 0.0
            return False
        try:
            d, alpha, theta = (float(odometry_values[i]) for i in range(3))
            use_odom = (all(math.isfinite(v) for v in (d, alpha, theta))
                        and -0.6 <= d <= 40.0)
        except (IndexError, TypeError, ValueError):
            use_odom = False
        if use_odom:
            bearing = self._odom_heading + alpha
            self._odom_x += max(0.0, d) * math.cos(bearing)
            self._odom_y += max(0.0, d) * math.sin(bearing)
            self._odom_heading = _wrap(self._odom_heading + theta)
        moved = (math.hypot(self._odom_x, self._odom_y) >= 16.0
                 if use_odom else
                 math.dist(pose.position, self._reference_position) >= 12.0)
        if moved:
            self._reference_position = pose.position
            self._reference_step = step
            self._odom_x = self._odom_y = self._odom_heading = 0.0
            return False
        if step - self._reference_step >= 24:
            self._reference_position = pose.position
            self._reference_step = step
            self._odom_x = self._odom_y = self._odom_heading = 0.0
            return True
        return False

    def update(self, *, pose: PoseEstimate, step: int, carrying: bool,
               mode: str, goal: Optional[WorldPoint],
               peers: Iterable[PeerMotion],
               drone_hits: Iterable[tuple[float, float]],
               lidar_distances: Optional[Sequence[float]],
               lidar_angles: Optional[Sequence[float]],
               odometry_values: Optional[Sequence[float]] = None,
               desired_forward: float = 0.0) -> Optional[TrafficAction]:
        hits = tuple((float(a), float(d)) for a, d in drone_hits
                     if math.isfinite(float(a)) and math.isfinite(float(d))
                     and float(d) >= 0.0)
        conflict = False
        known_near = False
        if pose.position is not None and pose.heading is not None and pose.valid:
            for peer in peers:
                age = step - peer.received_step
                if not 0 <= age <= 8:
                    continue
                distance = math.dist(pose.position, peer.position)
                if distance > 92.0:
                    continue
                bearing = _wrap(math.atan2(
                    peer.position[1] - pose.position[1],
                    peer.position[0] - pose.position[0],
                ) - pose.heading)
                if abs(bearing) > 1.25 and distance > 45.0:
                    continue
                corroborated = any(
                    abs(_wrap(angle - bearing)) < 0.55 and
                    abs(hit_distance - distance) < 45.0
                    for angle, hit_distance in hits
                )
                if not corroborated and distance > 60.0:
                    continue
                known_near = True
                if peer.carrying and not carrying:
                    conflict = True
                    break
                if peer.carrying == carrying and peer.peer_id < self.own_id:
                    # Nearby spawning drones or a same-direction convoy do
                    # not need right-of-way arbitration.
                    if not carrying and step < 18:
                        continue
                    if peer.heading is not None:
                        reverse_bearing = _wrap(bearing + math.pi - peer.heading
                                                + pose.heading)
                        head_on = (
                            abs(_wrap(peer.heading - pose.heading)) > 2.2 and
                            abs(bearing) < 1.1 and
                            abs(reverse_bearing) < 1.1
                        )
                    else:
                        head_on = (distance < 48.0 and abs(bearing) < 0.9
                                   and desired_forward > 0.1)
                    if head_on:
                        conflict = True
                        break
        if (not known_near and step >= 18 and desired_forward > 0.1 and any(
            abs(angle) < 0.8 and distance < 52.0
            for angle, distance in hits
        )):
            conflict = True

        if conflict:
            if step > self._yield_until + 1:
                self._yield_started = step
            self._yield_until = step + 8
        if step <= self._yield_until:
            self._reference_position = None
            self._odom_x = self._odom_y = self._odom_heading = 0.0
            return self._escape_motion(lidar_distances, lidar_angles,
                                       step, "yield")

        if self._is_stalled(pose, step, mode, goal, odometry_values):
            self._recovery_started = step
            self._recovery_until = step + 14
        if step <= self._recovery_until:
            if step - self._recovery_started >= 8:
                front = self._sector_min(lidar_distances, lidar_angles, 0.0)
                if front >= 55.0:
                    turn = -0.5 if self.own_id % 2 == 0 else 0.5
                    return TrafficAction(0.22, 0.0, turn, "recover_advance")
            return self._escape_motion(lidar_distances, lidar_angles,
                                       step, "recover")
        return None

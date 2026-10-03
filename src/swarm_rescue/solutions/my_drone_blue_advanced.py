"""Integrated blue controller using only contestant-visible sensors/state."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Optional

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import (
    DroneSemanticSensor,
)
from swarm_rescue.solutions.blue_team.contracts import (
    ExplorationDecision, PoseEstimate, WorldPoint,
)
from swarm_rescue.solutions.blue_team.frontier import FrontierExplorer
from swarm_rescue.solutions.blue_team.localization import LocalizationManager
from swarm_rescue.solutions.blue_team.occupancy_grid import OccupancyGrid
from swarm_rescue.solutions.blue_team.path_planner import GridPathPlanner
from swarm_rescue.solutions.blue_team.return_path import ReturnNavigator
from swarm_rescue.solutions.blue_team.task_allocation import CostAllocator
from swarm_rescue.solutions.blue_team.traffic import PeerMotion, TrafficManager
from swarm_rescue.solutions.my_drone_blue_basic import MyDroneBlueBasic, SemanticHit


class MyDroneBlueAdvanced(MyDroneBlueBasic):
    """Task-aware search, conservative routing, and reward-checked disposal."""

    _CARRY_CLEARANCE = 45.0
    _BID_SWITCH_MARGIN = 40

    def __init__(self, identifier: Optional[int] = None, misc_data: Any = None,
                 **kwargs: Any) -> None:
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)
        self._localizer = LocalizationManager(int(self.identifier or 0))
        self._grid = OccupancyGrid(self.size_area or (1000.0, 1000.0))
        self._planner = GridPathPlanner(self._grid)
        self._frontier = FrontierExplorer()
        self._allocator = CostAllocator(self._grid, self._planner)
        self._return_nav = ReturnNavigator()
        self._use_fused = False
        self._verified_release = False
        self._peer_bids: dict[int, tuple[WorldPoint, int, int]] = {}
        self._peer_explore: dict[int, tuple[WorldPoint, int]] = {}
        self._peer_motion: dict[int, PeerMotion] = {}
        self._traffic = TrafficManager(int(self.identifier or 0))
        self._last_traffic_reason: Optional[str] = None
        self._planned_goal: Optional[WorldPoint] = None
        self._planned_route: tuple[WorldPoint, ...] = ()
        self._waypoint_index = 0
        self._route_step = -100
        self._current_bid: Optional[int] = None
        self._carry_reward_baseline = float(self.reward)
        self._dock_target: Optional[WorldPoint] = None
        self._dock_spin_steps = 0
        self._dock_retry = 0
        self._dock_last_seen_step = -100
        self._carry_route: tuple[WorldPoint, ...] = ()
        self._carry_route_index = 0
        self._carry_route_goal: Optional[WorldPoint] = None
        self._carry_route_step = -100
        self._carry_route_revision = -1
        self._carry_route_status = "UNPLANNED"

    def measured_gps_position(self):
        if getattr(self, "_use_fused", False):
            pose = self._localizer.latest
            return pose.position if pose.valid else None
        return super().measured_gps_position()

    def measured_compass_angle(self):
        if getattr(self, "_use_fused", False):
            pose = self._localizer.latest
            return pose.heading if pose.valid else None
        return super().measured_compass_angle()

    def define_message_for_all(self) -> dict[str, Any]:
        message = super().define_message_for_all()
        message["cost_milli"] = self._current_bid
        pose = self._localizer.latest
        message["heading"] = pose.heading if pose.valid else None
        goal = self._frontier.goal
        message["explore"] = (
            goal.position if goal is not None and self._mode == "explore" else None
        )
        return message

    def control(self) -> CommandsDict:
        step = self._blue_step + 1
        odometry = self.odometer_values()
        # Bypass our pose accessors while acquiring unmodified sensor readings.
        pose = self._localizer.update(
            step, super().measured_gps_position(),
            super().measured_compass_angle(), odometry,
        )
        self._use_fused = True
        if self._localizer.reanchored:
            self._reset_world_memory()
        if step % 5 == 1 and pose.valid:
            self._grid.update_from_scan(
                pose, self.lidar_values(), self.lidar_rays_angles(),
                self.semantic_values() or (), step,
            )

        carrying = bool(self.grasped_bombs())
        if carrying and not self._was_carrying:
            self._carry_reward_baseline = float(self.reward)
            if (pose.position is not None and
                    (self._target_bomb is None or
                     math.dist(self._target_bomb, pose.position) > 45.0)):
                self._target_bomb = pose.position
            self._dock_target = None
            self._dock_spin_steps = 0
            self._dock_retry = 0
            self._reset_carry_route()
        self._verified_release = (
            self._was_carrying and not carrying and
            float(self.reward) > self._carry_reward_baseline + 1e-6
        )
        if self._was_carrying and not carrying:
            self._return_nav.reset()
            self._reset_carry_route()
            self._dock_target = None
            self._dock_spin_steps = 0
        elif carrying and not self._was_carrying:
            self._return_nav.begin(self._breadcrumbs, self._grid, step)

        command = super().control()
        if carrying and self._duplicate_carrier_should_yield():
            # The simulator permits two graspers on one bomb. The higher-ID
            # carrier must explicitly detach, or the two motors fight forever.
            command.update(forward=0.0, lateral=0.0, rotation=0.0,
                           grasper=0)
            self._current_bid = None
            return command
        if not carrying and self._peer_held_bomb_is_near():
            # A target already held by another drone must not be grabbed by
            # our still-armed chase/navigate command during the message lag.
            command["grasper"] = 0
        command = self._apply_traffic(command, carrying, odometry)
        self._current_bid = (
            self._bid(self._target_bomb)
            if self._target_bomb is not None and not carrying else None
        )
        # Low health is handled as a soft risk constraint; full return-to-base
        # is still higher priority when a bomb is carried.
        if self.drone_health <= 15 and command["forward"] > 0:
            command["forward"] = min(command["forward"], 0.58)
        return command

    def _reset_world_memory(self) -> None:
        self._grid = OccupancyGrid(self.size_area or (1000.0, 1000.0))
        self._planner = GridPathPlanner(self._grid)
        self._allocator = CostAllocator(self._grid, self._planner)
        self._frontier.reset()
        self._return_nav.reset()
        self._planned_route = ()
        self._planned_goal = None
        self._waypoint_index = 0
        self._breadcrumbs = []
        self._return_waypoints = []
        self._bomb_memories = []
        self._target_bomb = None
        self._known_disposal = None
        self._home_position = None
        self._traffic.reset()
        self._dock_target = None
        self._dock_spin_steps = 0
        self._dock_retry = 0
        self._dock_last_seen_step = -100
        self._reset_carry_route()

    def _reset_carry_route(self) -> None:
        self._carry_route = ()
        self._carry_route_index = 0
        self._carry_route_goal = None
        self._carry_route_step = -100
        self._carry_route_revision = -1
        self._carry_route_status = "UNPLANNED"

    def _duplicate_carrier_should_yield(self) -> bool:
        """Use one stable owner if two drones are attached to one bomb."""
        if self._target_bomb is None:
            return False
        own_id = int(self.identifier or 0)
        own_position = self._localizer.latest.position
        for peer_id, (claim, received, carrying) in self._peer_claims.items():
            if (peer_id >= own_id or not carrying or
                    self._blue_step - received > 5 or
                    math.dist(claim, self._target_bomb) > 48.0):
                continue
            motion = self._peer_motion.get(peer_id)
            if (motion is not None and own_position is not None and
                    math.dist(motion.position, own_position) > 70.0):
                continue
            return True
        return False

    def _peer_held_bomb_is_near(self) -> bool:
        for hit in self.semantic_values() or ():
            if (hit.entity_type == DroneSemanticSensor.TypeEntity.BOMB and
                    bool(hit.grasped) and float(hit.distance) < 55.0):
                return True
        position = self._localizer.latest.position
        if position is None:
            return False
        for peer in self._peer_motion.values():
            if (peer.carrying and self._blue_step - peer.received_step <= 5 and
                    math.dist(position, peer.position) < 58.0):
                return True
        return False

    def _mark_current_target_cleared(self) -> None:
        if self._verified_release:
            super()._mark_current_target_cleared()
        else:
            # A failed drop is not a verified disposal: preserve memories.
            self._target_bomb = None
            self._memory_search_steps = 0

    def _ingest_peer_messages(self) -> None:
        super()._ingest_peer_messages()
        for _, raw in self.communicator.received_messages:
            if not isinstance(raw, dict) or raw.get("kind") != self._MESSAGE_KIND:
                continue
            peer_id = raw.get("id")
            if not isinstance(peer_id, int) or peer_id == int(self.identifier or 0):
                continue
            position = self._valid_point(raw.get("position"))
            if position is not None:
                raw_heading = raw.get("heading")
                try:
                    heading = float(raw_heading) if raw_heading is not None else None
                except (TypeError, ValueError):
                    heading = None
                if heading is not None and not math.isfinite(heading):
                    heading = None
                self._peer_motion[peer_id] = PeerMotion(
                    peer_id, position, bool(raw.get("carrying", False)),
                    self._blue_step, heading,
                )
            claim = self._valid_point(raw.get("claim"))
            bid = raw.get("cost_milli")
            if claim is not None and isinstance(bid, int) and 0 <= bid <= 1000:
                self._peer_bids[peer_id] = (claim, bid, self._blue_step)
            elif claim is None:
                self._peer_bids.pop(peer_id, None)
            explore = self._valid_point(raw.get("explore"))
            if explore is not None:
                self._peer_explore[peer_id] = (explore, self._blue_step)
            else:
                self._peer_explore.pop(peer_id, None)
        self._peer_bids = {i: value for i, value in self._peer_bids.items()
                           if self._blue_step - value[2] <= self._CLAIM_TTL}
        self._peer_explore = {i: value for i, value in self._peer_explore.items()
                              if self._blue_step - value[1] <= 30}
        self._peer_motion = {i: value for i, value in self._peer_motion.items()
                             if self._blue_step - value.received_step <= 8}

    def _traffic_goal(self) -> Optional[WorldPoint]:
        if self._mode == "carry":
            return self._known_disposal or self._home_position
        if self._mode in ("navigate", "chase"):
            return self._target_bomb
        if self._mode == "explore":
            if self._frontier.goal is not None:
                return self._frontier.goal.position
            if (self._last_explore_decision is not None and
                    self._last_explore_decision.goal is not None):
                return self._last_explore_decision.goal.position
        return None

    def _apply_traffic(self, command: CommandsDict, carrying: bool,
                       odometry: Any) -> CommandsDict:
        previous_reason = self._last_traffic_reason
        self._last_traffic_reason = None
        # Do not interrupt the disposal-center push/release sequence.
        if carrying and self._dock_target is not None:
            return command
        drone_hits = []
        for hit in self.semantic_values() or ():
            if hit.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                drone_hits.append((hit.angle, hit.distance))
        action = self._traffic.update(
            pose=self._localizer.latest, step=self._blue_step,
            carrying=carrying, mode=self._mode, goal=self._traffic_goal(),
            peers=self._peer_motion.values(), drone_hits=drone_hits,
            lidar_distances=self.lidar_values(),
            lidar_angles=self.lidar_rays_angles(),
            odometry_values=odometry,
            desired_forward=float(command.get("forward", 0.0)),
        )
        if action is not None:
            if (carrying and action.reason.startswith("recover") and
                    not (previous_reason or "").startswith("recover")):
                self._reset_carry_route()
            keep_grasping = bool(command.get("grasper", 0))
            command.update(forward=action.forward, lateral=action.lateral,
                           rotation=action.rotation,
                           grasper=1 if carrying or keep_grasping else 0)
            self._last_traffic_reason = action.reason
        return command

    def _bid(self, point: Optional[WorldPoint]) -> Optional[int]:
        if point is None:
            return None
        return self._allocator.bid(
            point, self._localizer.latest, self._blue_step,
            int(self.drone_health), self._known_disposal, self._target_bomb,
        )

    def _claimed_by_preferred_peer(self, point: WorldPoint) -> bool:
        my_bid = self._bid(point)
        my_id = int(self.identifier or 0)
        for peer_id, (claim, received, carrying) in self._peer_claims.items():
            if (self._blue_step - received > self._CLAIM_TTL or
                    self._distance(point, claim) > self._CLAIM_MATCH_RADIUS):
                continue
            if carrying:
                return True
            other = self._peer_bids.get(peer_id)
            if (other is not None and
                    self._distance(other[0], point) <= self._CLAIM_MATCH_RADIUS):
                if my_bid is None or other[1] + self._BID_SWITCH_MARGIN < my_bid:
                    return True
                if (my_bid is not None and
                        abs(other[1] - my_bid) <= self._BID_SWITCH_MARGIN and
                        peer_id < my_id):
                    return True
            elif peer_id < my_id:
                return True
        return False

    def _choose_visible_bomb(self, visible: list[SemanticHit]) -> Optional[SemanticHit]:
        if self.drone_health <= 8:
            return None
        if self._scout_direction() and self._scout_is_in_transit():
            visible = [hit for hit in visible if float(hit[0].distance) <= 70.0]
        return super()._choose_visible_bomb(visible)

    def _choose_remembered_bomb(self) -> Optional[WorldPoint]:
        if self.drone_health <= 8:
            return None
        scout = bool(self._scout_direction()) and self._scout_is_in_transit()
        position = self._localizer.latest.position
        choices = [point for point, _ in self._bomb_memories
                   if not self._is_recently_cleared(point)
                   and not self._claimed_by_preferred_peer(point)
                   and (not scout or position is None or
                        math.dist(position, point) <= 70.0)]
        if not choices:
            return None
        rated = [(self._bid(point), point) for point in choices]
        return min(rated, key=lambda item: (
            item[0] if item[0] is not None else 1001,
            item[1][0], item[1][1],
        ))[1]

    def _scout_is_in_transit(self) -> bool:
        pose = self._localizer.latest
        if pose.position is None or self._home_position is None:
            return False
        return abs(pose.position[0] - self._home_position[0]) < 0.4 * self._grid.width

    def _scout_direction(self) -> float:
        count = int(self._blue_drone_count)
        if (count < 4 or int(self.identifier or 0) < math.ceil(0.6 * count)
                or self._home_position is None):
            return 0.0
        pose = self._localizer.latest
        if not pose.valid or pose.position is None:
            return 0.0
        max_steps = getattr(self._misc_data, "max_timestep_limit", None) or 1500
        if self._blue_step >= min(1400, int(0.9 * max_steps)):
            return 0.0
        width = self._grid.width
        home_x = self._home_position[0]
        direction = (1.0 if home_x < -0.15 * width else
                     -1.0 if home_x > 0.15 * width else
                     1.0 if int(self.identifier or 0) % 2 else -1.0)
        depth = direction * (pose.position[0] - home_x)
        return direction if depth < 0.90 * width else 0.0

    def _control_explore(self, cmd: CommandsDict) -> CommandsDict:
        pose = self._localizer.latest
        advance = self._scout_direction()
        goal = self._frontier.update(
            pose, self._grid, self._planner, self._blue_step,
            int(self.identifier or 0), int(self._blue_drone_count),
            (value[0] for value in self._peer_explore.values()),
            advance_direction=advance,
        )
        if goal is None:
            return super()._control_explore(cmd)
        self._last_explore_decision = ExplorationDecision(
            "ACTIVE", goal, goal.goal_id, 0.0, 0.0,
            self._blue_step + 30, "reachable_frontier",
        )
        if self._run_lidar_escape_if_needed(cmd):
            return cmd
        self._move_toward(*goal.position, cmd, use_lidar=True)
        return cmd

    def _control_chase_bomb(self, cmd: CommandsDict,
                            bomb: SemanticHit) -> CommandsDict:
        """Translate toward a nearby bomb while the grasper stays armed.

        The grasper surrounds the drone; exact frontal alignment is not
        required. A hard angle threshold made a noisy close target oscillate
        between rotate-only commands without ever entering grasping range.
        """
        self._carry_push_steps = 0
        self._reset_explore_state()
        cmd["grasper"] = 1
        hit = bomb[0]
        angle, distance = float(hit.angle), float(hit.distance)
        if not math.isfinite(angle) or not math.isfinite(distance):
            return cmd
        if distance > 55.0 and self._run_lidar_escape_if_needed(cmd):
            cmd["grasper"] = 1
            return cmd
        turn = max(-0.65, min(0.65, 0.75 * angle))
        if distance > 55.0:
            forward = 0.67 * max(0.0, math.cos(angle))
            lateral = 0.28 * math.sin(angle)
        else:
            speed = max(0.22, min(0.46, distance / 80.0 + 0.14))
            forward = speed * math.cos(angle)
            lateral = speed * math.sin(angle)
        self._set_motion(cmd, forward=forward, lateral=lateral,
                         rotation=turn)
        if distance > 28.0:
            self._apply_lidar_avoidance(cmd, path_clearance=distance)
        cmd["grasper"] = 1
        return cmd

    def _visible_disposal_surface(self) -> Optional[tuple[WorldPoint, float]]:
        """Estimate a stable patch of the visible disposal boundary."""
        points: list[WorldPoint] = []
        distances: list[float] = []
        for hit in self.semantic_values() or ():
            if hit.entity_type != DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER:
                continue
            distance = float(hit.distance)
            if not math.isfinite(distance):
                continue
            point = self._semantic_to_world(hit)
            if point is not None:
                points.append(point)
                distances.append(distance)
        if not points:
            return None
        xs = sorted(point[0] for point in points)
        ys = sorted(point[1] for point in points)
        # The median of all visible rays is less sensitive to the corner ray
        # that a nearest-hit target would select. Retries spread over the
        # visible face instead of repeatedly pushing at one blocked corner.
        fraction = (0.5 if self._dock_retry == 0 else
                    0.3 if self._dock_retry % 2 else 0.7)
        if xs[-1] - xs[0] >= ys[-1] - ys[0]:
            point = (xs[int(fraction * (len(xs) - 1))], float(median(ys)))
        else:
            point = (float(median(xs)), ys[int(fraction * (len(ys) - 1))])
        return point, min(distances)

    def _control_dock(self, cmd: CommandsDict,
                      observation: WorldPoint) -> CommandsDict:
        """Keep the bomb attached until it actually collides with disposal."""
        cmd["grasper"] = 1
        self._carry_push_steps = 0
        self._carry_spin_release_steps_left = 0
        if self._dock_target is None:
            self._dock_target = observation
        elif self._dock_spin_steps == 0:
            alpha = 0.15
            self._dock_target = (
                (1 - alpha) * self._dock_target[0] + alpha * observation[0],
                (1 - alpha) * self._dock_target[1] + alpha * observation[1],
            )
        pose = self._localizer.latest
        if pose.position is None:
            return cmd
        target_distance = math.dist(pose.position, self._dock_target)
        if target_distance > 26.0:
            self._dock_spin_steps = 0
            if pose.heading is None:
                return cmd
            desired = math.atan2(self._dock_target[1] - pose.position[1],
                                 self._dock_target[0] - pose.position[0])
            angle = math.atan2(math.sin(desired - pose.heading),
                               math.cos(desired - pose.heading))
            speed = max(0.20, min(0.55, target_distance / 75.0))
            self._set_motion(
                cmd, forward=speed * math.cos(angle),
                lateral=speed * math.sin(angle),
                rotation=max(-0.45, min(0.45, 0.35 * angle)),
            )
            cmd["grasper"] = 1
            return cmd
        # At the boundary, a carried bomb can be behind the drone. Rotating
        # while retaining the pivot joint sweeps it around the vehicle; the
        # simulator disposes it on contact. Never release after a timer.
        self._dock_spin_steps += 1
        sign = 1.0 if int(self.identifier or 0) % 2 == 0 else -1.0
        self._set_motion(cmd, forward=0.0, lateral=0.0,
                         rotation=0.65 * sign)
        cmd["grasper"] = 1
        if self._dock_spin_steps >= 42:
            self._dock_spin_steps = 0
            self._dock_retry += 1
            self._dock_target = None
        return cmd

    def _planned_carry_waypoint(self, position: WorldPoint,
                                target: WorldPoint) -> Optional[WorldPoint]:
        """Prefer a short certified route to the disposal/home target."""
        changed_goal = (self._carry_route_goal is None or
                        math.dist(target, self._carry_route_goal) > 28.0)
        elapsed = self._blue_step - self._carry_route_step
        changed_map = self._grid.revision != self._carry_route_revision
        retry_after = (120 if self._carry_route else
                       24 if self._carry_route_status == "START_BLOCKED"
                       else 80)
        if (changed_goal or elapsed >= retry_after or
                (not self._carry_route and changed_map and elapsed >= 8)):
            route = self._planner.plan(
                position, target, self._blue_step,
                clearance=self._CARRY_CLEARANCE, max_expansions=3500,
            )
            self._carry_route = (route.waypoints if route.status == "READY"
                                 else ())
            self._carry_route_index = 0
            self._carry_route_goal = target
            self._carry_route_step = self._blue_step
            self._carry_route_revision = self._grid.revision
            self._carry_route_status = route.status
        while (self._carry_route_index < len(self._carry_route) and
               math.dist(position,
                         self._carry_route[self._carry_route_index]) <= 28.0):
            self._carry_route_index += 1
        if self._carry_route_index >= len(self._carry_route):
            return None
        waypoint = self._carry_route[self._carry_route_index]
        if self._grid.check_segment(
                position, waypoint, self._CARRY_CLEARANCE,
                self._blue_step) == "SAFE":
            return waypoint
        # A newly observed obstacle invalidates the active segment. Try a
        # fresh map route next control cycle; the breadcrumb fallback handles
        # this cycle without driving through the obstacle.
        self._carry_route = ()
        self._carry_route_step = -100
        self._carry_route_status = "BLOCKED"
        return None

    def _control_carry(self, cmd: CommandsDict) -> CommandsDict:
        cmd["grasper"] = 1
        pose = self._localizer.latest
        if pose.position is None or not pose.valid:
            # Do not use uncertain world coordinates, but keep moving slowly
            # with local obstacle sensing so a disabled-sensor zone can end.
            if self._run_lidar_escape_if_needed(cmd):
                return cmd
            cmd["forward"] = 0.28
            self._apply_lidar_avoidance(cmd)
            return cmd
        visible = self._visible_disposal_surface()
        if visible is not None:
            surface, distance = visible
            self._remember_disposal(surface)
            if distance < 105.0:
                self._dock_last_seen_step = self._blue_step
                return self._control_dock(cmd, surface)
        if (self._dock_target is not None and
                self._blue_step - self._dock_last_seen_step <= 15):
            return self._control_dock(cmd, self._dock_target)
        self._dock_target = None
        self._dock_spin_steps = 0
        disposal = self._known_disposal or self._home_position
        if disposal is None:
            disposal = (0.0, 0.0)
        if self._known_disposal is not None and math.dist(
                pose.position, disposal) < 115.0:
            if self._run_lidar_escape_if_needed(cmd):
                cmd["grasper"] = 1
                return cmd
            self._move_toward(*disposal, cmd, use_lidar=True,
                              lidar_clearance=math.dist(pose.position, disposal))
            cmd["grasper"] = 1
            return cmd
        waypoint = self._planned_carry_waypoint(pose.position, disposal)
        if waypoint is None:
            waypoint = self._return_nav.next_waypoint(
                pose.position, self._grid, self._blue_step
            )
        if waypoint is None:
            waypoint = disposal
        if self._run_lidar_escape_if_needed(cmd):
            cmd["grasper"] = 1
            return cmd
        # The selected point already belongs to the certified route. Avoid
        # running a second A* merely to steer toward this waypoint.
        super()._move_toward(
            *waypoint, cmd, use_lidar=True,
            lidar_clearance=math.dist(pose.position, waypoint),
        )
        cmd["grasper"] = 1
        return cmd

    def _move_toward(self, target_x: float, target_y: float,
                     cmd: CommandsDict, *, use_lidar: bool = False,
                     lidar_clearance: Optional[float] = None) -> bool:
        target = (target_x, target_y)
        pose: PoseEstimate = self._localizer.latest
        if (self._mode == "carry" and not use_lidar and
                pose.position is not None and
                math.dist(pose.position, target) > 35.0):
            use_lidar = True
        if not use_lidar or not pose.valid or pose.position is None:
            return super()._move_toward(
                target_x, target_y, cmd, use_lidar=use_lidar,
                lidar_clearance=lidar_clearance,
            )
        distance = math.dist(pose.position, target)
        if distance < 70.0:
            return super()._move_toward(
                target_x, target_y, cmd, use_lidar=True,
                lidar_clearance=lidar_clearance,
            )
        clearance = (self._CARRY_CLEARANCE if self.grasped_bombs() else
                     30.0 if self._mode == "explore" else 59.0)
        if (self._planned_goal is None or
                math.dist(target, self._planned_goal) > 20.0 or
                self._blue_step - self._route_step > 60):
            route = self._planner.plan(
                pose.position, target, self._blue_step,
                clearance=clearance, max_expansions=500,
            )
            self._planned_goal = target
            self._planned_route = route.waypoints if route.status == "READY" else ()
            self._waypoint_index = 0
            self._route_step = self._blue_step
        while (self._waypoint_index < len(self._planned_route) and
               math.dist(pose.position,
                         self._planned_route[self._waypoint_index]) <= 28.0):
            self._waypoint_index += 1
        if self._waypoint_index < len(self._planned_route):
            waypoint = self._planned_route[self._waypoint_index]
            if self._grid.check_segment(
                pose.position, waypoint, clearance, self._blue_step
            ) == "SAFE":
                target = waypoint
            else:
                self._route_step = -100
        # If no certified path exists, local Lidar still guards the direct
        # attempt. This also allows entering newly discovered free space.
        arrived = super()._move_toward(
            *target, cmd, use_lidar=True,
            lidar_clearance=math.dist(pose.position, target),
        )
        if self.drone_health <= 30 and cmd["forward"] > 0:
            cmd["forward"] = min(cmd["forward"], 0.72)
        return arrived

"""Reachable-frontier exploration layered over deterministic stripe coverage."""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Optional, Tuple

import numpy as np

from .contracts import NavigationGoal, PoseEstimate, WorldPoint
from .occupancy_grid import FREE, UNKNOWN, GridCell, OccupancyGrid
from .path_planner import GridPathPlanner


class FrontierExplorer:
    def __init__(self) -> None:
        self._goal: Optional[NavigationGoal] = None
        self._last_refresh = -100
        self._last_progress = -1
        self._best_distance = math.inf
        self._reached = 0
        self._blocked: dict[GridCell, int] = {}
        self._counter = 0
        self._current_score = -math.inf
        self._advance_anchor: Optional[float] = None
        self._advance_anchor_step = -1
        self._sweep_sign = 0.0

    @property
    def goal(self) -> Optional[NavigationGoal]:
        return self._goal

    def reset(self) -> None:
        self.__init__()

    def update(self, pose: PoseEstimate, grid: OccupancyGrid,
               planner: GridPathPlanner, step: int, drone_id: int,
               drone_count: int, peer_goals: Iterable[WorldPoint] = (),
               advance_direction: float = 0.0) -> Optional[NavigationGoal]:
        if (not pose.valid or pose.frame_id != "world" or pose.position is None
                or pose.position_variance > 400.0 or
                int(np.count_nonzero(grid.states == FREE)) < 20):
            return None
        current_cell = grid.world_to_cell(pose.position)
        if current_cell is None or grid.cell_status(current_cell, step) != FREE:
            return None
        if advance_direction:
            progress = advance_direction * pose.position[0]
            if self._advance_anchor is None:
                self._advance_anchor = progress
                self._advance_anchor_step = step
                self._sweep_sign = -1.0 if drone_id % 2 else 1.0
            elif progress >= self._advance_anchor + 75.0:
                self._advance_anchor = progress
                self._advance_anchor_step = step
                # Once around a wall, return to this drone's assigned half
                # instead of letting all bypassing scouts remain on one side.
                self._sweep_sign = -1.0 if drone_id % 2 else 1.0
            elif step - self._advance_anchor_step >= 110:
                # Repeatedly pushing into the same wall is unproductive.
                # Sweep the opposite end of an obstacle instead.
                self._sweep_sign *= -1.0
                self._advance_anchor = progress
                self._advance_anchor_step = step
                self._goal = None
        if self._goal is not None:
            distance = math.dist(pose.position, self._goal.position)
            if distance <= 30.0:
                self._reached += 1
                if self._reached >= 3:
                    cell = grid.world_to_cell(self._goal.position)
                    if cell is not None:
                        self._blocked[cell] = step + 80
                    self._goal = None
            else:
                self._reached = 0
            if self._goal is not None:
                if self._best_distance - distance >= 10:
                    self._best_distance, self._last_progress = distance, step
                elif step - self._last_progress >= 45:
                    cell = grid.world_to_cell(self._goal.position)
                    if cell is not None:
                        self._blocked[cell] = step + 120
                    self._goal = None
                elif step - self._last_refresh < 20:
                    return self._goal
        if step - self._last_refresh < 20 and self._goal is None:
            return None
        self._last_refresh = step
        self._blocked = {cell: until for cell, until in self._blocked.items()
                         if until > step}
        blocked_points = tuple(grid.cell_center(cell) for cell in self._blocked)

        reachable = self._reachable(current_cell, grid, step)
        frontier = {cell for cell in reachable if any(
            0 <= cell[0] + dr < grid.rows and 0 <= cell[1] + dc < grid.cols
            and grid.states[cell[0] + dr, cell[1] + dc] == UNKNOWN
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )}
        clusters = self._clusters(frontier)
        peers = tuple(peer_goals)
        best: Optional[Tuple[float, WorldPoint, int]] = None
        # The straight-line route cost is an optimistic bound on the true
        # route cost. Rank all candidates cheaply, then stop running A* once
        # no remaining candidate can beat the best certified route.
        candidates_to_plan: list[Tuple[float, int, WorldPoint, float]] = []
        strip_width = (grid.width - 90.0) / max(1, drone_count)
        strip_center = -grid.width / 2 + 45.0 + (drone_id + 0.5) * strip_width
        route_scale = max(grid.width, grid.height)
        for cluster in sorted(clusters, key=len, reverse=True)[:8]:
            if len(cluster) < 3:
                continue
            # Large open-room frontiers may form one ring. Sampling actual
            # frontier cells, not its centroid, keeps drones spread apart.
            frontier_choices = sorted(
                cluster,
                key=lambda cell: (
                    abs(grid.cell_center(cell)[0] - strip_center),
                    math.dist(grid.cell_center(cell), pose.position), cell,
                ),
            )
            stride = max(1, len(frontier_choices) // 8)
            # A wall may make the strip-nearest frontier unreachable. Include
            # both vertical extremes so scouts can discover either wall end.
            candidates = (frontier_choices[:5]
                          + frontier_choices[::stride][:8]
                          + sorted(cluster, key=lambda c: c[0])[:2]
                          + sorted(cluster, key=lambda c: -c[0])[:2])
            seen: set[GridCell] = set()
            for frontier_cell in candidates:
                edge = grid.cell_center(frontier_cell)
                dx, dy = edge[0] - pose.position[0], edge[1] - pose.position[1]
                distance = max(1.0, math.hypot(dx, dy))
                # Stand back two cells so the vehicle's clearance does not
                # extend into unknown cells around the actual frontier.
                point = (edge[0] - 2.5 * grid.resolution * dx / distance,
                         edge[1] - 2.5 * grid.resolution * dy / distance)
                cell = grid.world_to_cell(point)
                if (cell is None or cell not in reachable or cell in seen):
                    continue
                seen.add(cell)
                point = grid.cell_center(cell)
                if any(math.dist(point, old) < 60.0 for old in blocked_points):
                    continue
                if grid.check_segment(point, point, 30.0, step) != "SAFE":
                    continue
                gain = min(1.0, grid.unknown_count(point, 200.0) / 180.0)
                overlap = max((max(0.0, 1 - math.dist(point, peer) / 150.0)
                               for peer in peers), default=0.0)
                outside = max(0.0, abs(point[0] - strip_center) - strip_width / 2)
                strip_penalty = min(0.30, 0.7 * outside / max(1.0, grid.width))
                switch = 0.0 if self._goal is not None and math.dist(
                    point, self._goal.position) < 45 else 0.08
                base_score = (0.45 * gain - 0.20 * overlap
                              - 0.10 * grid.risk_near(point, 50.0, step)
                              - strip_penalty - switch)
                if advance_direction:
                    base_score += (0.52 * advance_direction *
                                   (point[0] - pose.position[0]) / grid.width)
                    base_score += (0.22 * self._sweep_sign *
                                   (point[1] - pose.position[1]) / grid.height)
                straight_distance = math.dist(pose.position, point)
                # Planner returns AT_GOAL with zero cost within 12 pixels.
                lower_cost = (0.0 if straight_distance < 12.0 else
                              min(1.0, straight_distance / route_scale))
                upper_score = base_score - 0.25 * lower_cost
                candidates_to_plan.append((upper_score,
                                           len(candidates_to_plan),
                                           point, base_score))
        planned = 0
        for upper_score, order, point, base_score in sorted(
                candidates_to_plan, key=lambda item: (-item[0], item[1])):
            if best is not None and upper_score + 1e-12 < best[0]:
                break
            if planned >= (6 if best is not None else 12):
                break
            route = planner.plan(pose.position, point, step,
                                 max_expansions=350, clearance=30.0)
            planned += 1
            if route.status not in ("READY", "AT_GOAL"):
                continue
            cost = min(1.0, route.geometric_length_px / route_scale)
            score = base_score - 0.25 * cost
            if (best is None or score > best[0] or
                    (score == best[0] and order < best[2])):
                best = (score, point, order)
        if best is None:
            self._goal = None
            return None
        if self._goal is not None and best[0] < self._current_score + 0.12:
            return self._goal
        self._current_score = best[0]
        self._counter += 1
        self._goal = NavigationGoal(
            "frontier:world:d{}:{}".format(drone_id, self._counter),
            "EXPLORE_FRONTIER", best[1], 30.0, 0.6, step,
        )
        self._best_distance = math.dist(pose.position, best[1])
        self._last_progress = step
        self._reached = 0
        return self._goal

    @staticmethod
    def _reachable(start: GridCell, grid: OccupancyGrid, step: int) -> set[GridCell]:
        found = {start}
        queue = deque((start,))
        while queue:
            row, col = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cell = row + dr, col + dc
                if (0 <= cell[0] < grid.rows and 0 <= cell[1] < grid.cols
                        and cell not in found and grid.cell_status(cell, step) == FREE):
                    found.add(cell)
                    queue.append(cell)
        return found

    @staticmethod
    def _clusters(cells: set[GridCell]) -> list[set[GridCell]]:
        pending = set(cells)
        result = []
        while pending:
            start = pending.pop()
            cluster = {start}
            queue = [start]
            while queue:
                row, col = queue.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        candidate = row + dr, col + dc
                        if candidate in pending:
                            pending.remove(candidate)
                            cluster.add(candidate)
                            queue.append(candidate)
            result.append(cluster)
        return result

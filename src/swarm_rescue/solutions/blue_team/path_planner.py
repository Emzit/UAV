"""Bounded eight-neighbour A* over certified-free occupancy cells."""

from __future__ import annotations

import heapq
import math
from typing import Dict, Optional, Tuple

from .contracts import PathPlan, WorldPoint
from .occupancy_grid import FREE, GridCell, OccupancyGrid


class GridPathPlanner:
    def __init__(self, grid: OccupancyGrid) -> None:
        self.grid = grid
        self._cell_cache_epoch: Optional[Tuple[int, int]] = None
        self._cell_clearance_cache: Dict[Tuple[float, GridCell], bool] = {}
        self._cell_risk_cache: Dict[GridCell, float] = {}

    def plan(self, start: WorldPoint, goal: WorldPoint, step: int,
             carrying: bool = False, max_expansions: int = 800,
             clearance: Optional[float] = None) -> PathPlan:
        clearance = (45.0 if carrying else 59.0) if clearance is None else clearance
        revision = self.grid.revision

        def fail(status: str, reason: str) -> PathPlan:
            return PathPlan(status, (), math.inf, revision, step, reason,
                            math.inf, clearance)

        start_cell = self.grid.world_to_cell(start)
        goal_cell = self.grid.world_to_cell(goal)
        if start_cell is None:
            return fail("START_BLOCKED", "start_out_of_bounds")
        if goal_cell is None:
            return fail("GOAL_OUT_OF_BOUNDS", "goal_out_of_bounds")
        if math.dist(start, goal) < 12.0:
            return PathPlan("AT_GOAL", (), 0.0, revision, step, "within_tolerance",
                            0.0, clearance)
        if self.grid.check_segment(start, goal, clearance, step) == "SAFE":
            distance = math.dist(start, goal)
            return PathPlan("READY", (goal,), distance, revision, step,
                            "certified_direct", distance, clearance)

        checked: Dict[GridCell, bool] = {}
        checked_risk: Dict[GridCell, float] = {}
        # Multiple frontier candidates are planned against the same local
        # map. Reuse certified cell clearances only within the same timestep
        # and grid revision; dynamic obstacles can expire on the next step.
        cache_enabled = 0 <= self.grid.updated_step <= step
        if cache_enabled:
            epoch = (step, revision)
            if self._cell_cache_epoch != epoch:
                self._cell_cache_epoch = epoch
                self._cell_clearance_cache.clear()
                self._cell_risk_cache.clear()

        def safe(cell: GridCell) -> bool:
            if cell not in checked:
                row, col = cell
                if self.grid.cell_status(cell, step) != FREE:
                    checked[cell] = False
                else:
                    key = (clearance, cell)
                    if cache_enabled and key in self._cell_clearance_cache:
                        checked[cell] = self._cell_clearance_cache[key]
                    else:
                        point = self.grid.cell_center(cell)
                        checked[cell] = self.grid.check_segment(
                            point, point, clearance, step
                        ) == "SAFE"
                        if cache_enabled:
                            self._cell_clearance_cache[key] = checked[cell]
            return checked[cell]

        def risk(cell: GridCell) -> float:
            if cell not in checked_risk:
                if cache_enabled and cell in self._cell_risk_cache:
                    checked_risk[cell] = self._cell_risk_cache[cell]
                else:
                    value = self.grid.risk_near(
                        self.grid.cell_center(cell), 40.0, step
                    )
                    checked_risk[cell] = value
                    if cache_enabled:
                        self._cell_risk_cache[cell] = value
            return checked_risk[cell]

        def nearest(seed: GridCell, radius: int) -> Optional[GridCell]:
            candidates = ((seed[0] + dr, seed[1] + dc)
                          for dr in range(-radius, radius + 1)
                          for dc in range(-radius, radius + 1))
            options = [cell for cell in candidates
                       if 0 <= cell[0] < self.grid.rows
                       and 0 <= cell[1] < self.grid.cols and safe(cell)]
            return min(options, key=lambda cell: math.dist(
                self.grid.cell_center(seed), self.grid.cell_center(cell)
            )) if options else None

        if not safe(start_cell):
            return fail("START_BLOCKED", "no_certified_free_start")
        exact_goal_safe = safe(goal_cell)
        if not exact_goal_safe:
            goal_cell = nearest(goal_cell, 3)
            if goal_cell is None:
                return fail("GOAL_BLOCKED", "no_certified_free_goal_nearby")
        if start_cell == goal_cell:
            proxy = self.grid.cell_center(goal_cell)
            if math.dist(start, proxy) < 12:
                return PathPlan("AT_GOAL", (), 0.0, revision, step,
                                "at_safe_proxy", 0.0, clearance)
            if self.grid.check_segment(start, proxy, clearance, step) != "SAFE":
                return fail("NO_PATH", "uncertified_safe_proxy")
            return PathPlan("READY", (proxy,), math.dist(start, proxy),
                            revision, step, "safe_proxy", math.dist(start, proxy),
                            clearance)

        def heuristic(cell: GridCell) -> float:
            dx, dy = abs(cell[1] - goal_cell[1]), abs(cell[0] - goal_cell[0])
            return self.grid.resolution * (max(dx, dy) +
                                           (math.sqrt(2) - 1) * min(dx, dy))

        queue = [(heuristic(start_cell), 0.0, start_cell)]
        costs: Dict[GridCell, float] = {start_cell: 0.0}
        parent: Dict[GridCell, GridCell] = {}
        expanded = 0
        while queue and expanded < max_expansions:
            _, cost, cell = heapq.heappop(queue)
            if cost > costs.get(cell, math.inf):
                continue
            expanded += 1
            if cell == goal_cell:
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    nxt = cell[0] + dr, cell[1] + dc
                    if (not 0 <= nxt[0] < self.grid.rows or
                            not 0 <= nxt[1] < self.grid.cols or not safe(nxt)):
                        continue
                    if dr and dc and (not safe((cell[0] + dr, cell[1])) or
                                      not safe((cell[0], cell[1] + dc))):
                        continue
                    move = self.grid.resolution * (math.sqrt(2) if dr and dc else 1.0)
                    proposed = cost + move * (1.0 + 0.35 * risk(nxt))
                    if proposed < costs.get(nxt, math.inf):
                        costs[nxt] = proposed
                        parent[nxt] = cell
                        heapq.heappush(queue, (proposed + heuristic(nxt), proposed, nxt))
        if goal_cell not in costs:
            return fail("BUDGET_EXCEEDED" if queue else "NO_PATH",
                        "astar_budget" if queue else "disconnected_known_free")

        route = [goal_cell]
        while route[-1] != start_cell:
            route.append(parent[route[-1]])
        route.reverse()
        points = [start] + [self.grid.cell_center(cell) for cell in route[1:]]
        if exact_goal_safe and self.grid.check_segment(points[-1], goal, clearance, step) == "SAFE":
            points.append(goal)
        if any(self.grid.check_segment(a, b, clearance, step) != "SAFE"
               for a, b in zip(points, points[1:])):
            return fail("NO_PATH", "uncertified_adjacent_segment")
        smoothed = [points[0]]
        index = 0
        while index < len(points) - 1:
            far = len(points) - 1
            while far > index + 1 and self.grid.check_segment(
                    points[index], points[far], clearance, step) != "SAFE":
                far -= 1
            smoothed.append(points[far])
            index = far
        waypoints = tuple(smoothed[1:])
        geometric = sum(math.dist(a, b) for a, b in zip(smoothed, smoothed[1:]))
        return PathPlan("READY", waypoints, costs[goal_cell], revision, step,
                        "known_free_astar", geometric, clearance)

    def estimate_cost(self, start: WorldPoint, goal: WorldPoint, step: int,
                      carrying: bool = False) -> Tuple[str, float, int]:
        route = self.plan(start, goal, step, carrying, max_expansions=250)
        return route.status, route.geometric_length_px, route.map_revision

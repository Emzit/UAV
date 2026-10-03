"""Comparable, health-aware local bids for bomb assignments."""

from __future__ import annotations

import math
from typing import Optional

from .contracts import PoseEstimate, WorldPoint
from .occupancy_grid import OccupancyGrid
from .path_planner import GridPathPlanner


class CostAllocator:
    """Lower cost wins; disconnected drones remain independently useful."""

    def __init__(self, grid: OccupancyGrid, planner: GridPathPlanner) -> None:
        self.grid = grid
        self.planner = planner
        self._cached: dict[tuple[object, ...], int] = {}

    def bid(self, point: WorldPoint, pose: PoseEstimate, step: int,
            health: int, disposal: Optional[WorldPoint],
            current_target: Optional[WorldPoint]) -> Optional[int]:
        if (not pose.valid or pose.frame_id != "world" or pose.position is None
                or pose.position_variance > 400.0 or health <= 8):
            return None
        key = (round(point[0] / 15), round(point[1] / 15),
               round(pose.position[0] / 20), round(pose.position[1] / 20),
               round(health / 5),
               None if disposal is None else
               (round(disposal[0] / 20), round(disposal[1] / 20)),
               None if current_target is None else
               (round(current_target[0] / 20), round(current_target[1] / 20)),
               step // 12, self.grid.revision)
        if key in self._cached:
            return self._cached[key]
        scale = max(self.grid.width, self.grid.height, 400.0)
        # A bid is only a comparison between drones, not a route to execute.
        # Running two A* searches per bid made communication/target selection
        # dominate the control budget as the shared bomb list grew.
        length = math.dist(pose.position, point)
        if self.grid.check_segment(pose.position, point, 30.0, step) != "SAFE":
            length = 1.35 * length + 0.15 * scale
        pickup = min(1.0, length / scale)
        if disposal is None:
            return_cost = 0.65
        else:
            # The disposal observation is a visible surface point, often not
            # a traversable cell. A* to that point was both expensive and
            # systematically failed near the center's physical boundary.
            length = 1.25 * math.dist(point, disposal) + 0.10 * scale
            return_cost = min(1.0, length / scale)
        health_cost = max(0.0, min(1.0, (25.0 - health) / 25.0))
        risk = self.grid.risk_near(point, 60.0, step)
        uncertainty = min(1.0, math.sqrt(pose.position_variance) / 40.0)
        switching = (1.0 if current_target is not None and
                     math.dist(point, current_target) > 58.0 else 0.0)
        cost = (0.34 * pickup + 0.18 * return_cost + 0.14 * health_cost
                + 0.12 * risk + 0.08 * uncertainty + 0.09 * switching)
        result = int(round(1000 * min(1.0, max(0.0, cost))))
        if len(self._cached) > 128:
            self._cached.clear()
        self._cached[key] = result
        return result

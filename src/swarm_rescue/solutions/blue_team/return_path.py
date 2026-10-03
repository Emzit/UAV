"""Breadcrumb return with shortcuts only across verified free space."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from .contracts import WorldPoint
from .occupancy_grid import OccupancyGrid


class ReturnNavigator:
    _CARRY_CLEARANCE = 45.0

    def __init__(self) -> None:
        self._route: list[WorldPoint] = []
        self._index = -1
        self._last_shortcut_step = -100
        self._last_shortcut_revision = -1

    def reset(self) -> None:
        self._route.clear()
        self._index = -1
        self._last_shortcut_step = -100
        self._last_shortcut_revision = -1

    def begin(self, breadcrumbs: Sequence[WorldPoint],
              grid: Optional[OccupancyGrid] = None, step: int = 0) -> None:
        route: list[WorldPoint] = []
        for point in breadcrumbs:
            if grid is not None:
                # Erase a loop only if the new observation can be joined to
                # the earlier visit through certified free space.
                for index in range(max(0, len(route) - 2)):
                    if (math.dist(point, route[index]) <= 55.0 and
                            grid.check_segment(
                                route[index], point,
                                self._CARRY_CLEARANCE, step) == "SAFE"):
                        del route[index + 1:]
                        break
            route.append(point)
        self._route = route
        self._index = len(self._route) - 1
        self._last_shortcut_step = -100
        self._last_shortcut_revision = -1

    def next_waypoint(self, position: WorldPoint, grid: OccupancyGrid,
                      step: int) -> Optional[WorldPoint]:
        old_index = self._index
        while self._index >= 0 and math.dist(
            position, self._route[self._index]
        ) <= 32.0:
            self._index -= 1
        if self._index < 0:
            return None
        if (grid.updated_step >= 0 and self._index == old_index and
                grid.revision == self._last_shortcut_revision and
                step - self._last_shortcut_step < 8):
            return self._route[self._index]
        self._last_shortcut_step = step
        self._last_shortcut_revision = grid.revision
        # Index zero is the earliest known home breadcrumb. A shortcut is
        # allowed only when the whole corridor, including clearance, is free.
        for candidate_index in range(self._index + 1):
            point = self._route[candidate_index]
            if grid.check_segment(
                    position, point, self._CARRY_CLEARANCE, step) == "SAFE":
                self._index = candidate_index
                break
        return self._route[self._index]

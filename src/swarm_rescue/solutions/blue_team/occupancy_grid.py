"""Online occupancy grid built only from permitted pose and Lidar readings."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .contracts import PoseEstimate, WorldPoint

GridCell = Tuple[int, int]
UNKNOWN, FREE, OCCUPIED = -1, 0, 1


class OccupancyGrid:
    """Conservative three-state grid; unknown cells are not certified free."""

    def __init__(self, world_size: WorldPoint, resolution: float = 20.0) -> None:
        self.width, self.height = float(world_size[0]), float(world_size[1])
        self.resolution = float(resolution)
        self.cols = math.ceil(self.width / resolution)
        self.rows = math.ceil(self.height / resolution)
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.states = np.full((self.rows, self.cols), UNKNOWN, dtype=np.int8)
        self.hit_count = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.last_hit_step = np.full((self.rows, self.cols), -100, dtype=np.int32)
        self.dynamic_until = np.full((self.rows, self.cols), -1, dtype=np.int32)
        self.revision = 0
        self.updated_step = -1
        self._clearance_offsets_cache: dict[float, tuple[GridCell, ...]] = {}

    def world_to_cell(self, point: WorldPoint) -> Optional[GridCell]:
        x, y = point
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if not (-self.width / 2 <= x < self.width / 2 and
                -self.height / 2 <= y < self.height / 2):
            return None
        return (min(self.rows - 1, int((y + self.height / 2) / self.resolution)),
                min(self.cols - 1, int((x + self.width / 2) / self.resolution)))

    def cell_center(self, cell: GridCell) -> WorldPoint:
        row, col = cell
        x0 = -self.width / 2 + col * self.resolution
        y0 = -self.height / 2 + row * self.resolution
        return ((x0 + min(self.width / 2, x0 + self.resolution)) / 2,
                (y0 + min(self.height / 2, y0 + self.resolution)) / 2)

    def segment_cells(self, start: WorldPoint, end: WorldPoint) -> Tuple[GridCell, ...]:
        """Supercover traversal, including both cells at an exact corner."""
        first, last = self.world_to_cell(start), self.world_to_cell(end)
        if first is None or last is None:
            return ()
        row, col = first
        target_row, target_col = last
        x0 = (start[0] + self.width / 2) / self.resolution
        y0 = (start[1] + self.height / 2) / self.resolution
        x1 = (end[0] + self.width / 2) / self.resolution
        y1 = (end[1] + self.height / 2) / self.resolution
        dx, dy = x1 - x0, y1 - y0
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        tx = ((col + 1 - x0) / dx if dx > 0 else
              (col - x0) / dx if dx < 0 else math.inf)
        ty = ((row + 1 - y0) / dy if dy > 0 else
              (row - y0) / dy if dy < 0 else math.inf)
        inc_x = abs(1 / dx) if dx else math.inf
        inc_y = abs(1 / dy) if dy else math.inf
        result = [first]
        seen = {first}

        def add(r: int, c: int) -> None:
            cell = (r, c)
            if 0 <= r < self.rows and 0 <= c < self.cols and cell not in seen:
                result.append(cell)
                seen.add(cell)

        for _ in range(self.rows + self.cols + 4):
            if (row, col) == (target_row, target_col):
                break
            if abs(tx - ty) < 1e-10:
                add(row, col + step_x)
                add(row + step_y, col)
                row, col = row + step_y, col + step_x
                tx += inc_x
                ty += inc_y
            elif tx < ty:
                col += step_x
                tx += inc_x
            else:
                row += step_y
                ty += inc_y
            add(row, col)
        return tuple(result)

    def _ray_end(self, start: WorldPoint, angle: float, length: float) -> WorldPoint:
        ux, uy = math.cos(angle), math.sin(angle)
        limit = length
        for p, u, half in ((start[0], ux, self.width / 2),
                           (start[1], uy, self.height / 2)):
            if u > 0:
                limit = min(limit, (half - p - 1e-6) / u)
            elif u < 0:
                limit = min(limit, (-half - p + 1e-6) / u)
        limit = max(0.0, limit)
        return start[0] + limit * ux, start[1] + limit * uy

    def update_from_scan(
        self,
        pose: PoseEstimate,
        distances: Optional[Sequence[float]],
        angles: Optional[Sequence[float]],
        semantic: Iterable[object],
        step: int,
    ) -> bool:
        if (not pose.valid or pose.frame_id != "world" or pose.position is None
                or pose.heading is None or pose.position_variance > 400.0
                or distances is None or angles is None
                or self.world_to_cell(pose.position) is None):
            return False
        free: set[GridCell] = set()
        hits: set[GridCell] = set()
        dynamic: set[GridCell] = set()
        semantic_list = tuple(semantic or ())
        for raw_distance, raw_angle in zip(distances, angles):
            try:
                distance, rel_angle = float(raw_distance), float(raw_angle)
            except (TypeError, ValueError):
                continue
            if (not math.isfinite(distance) or not math.isfinite(rel_angle)
                    or distance <= 0 or distance > 305):
                continue
            projection_var = pose.position_variance + distance ** 2 * pose.heading_variance
            if projection_var > 30.0 ** 2:
                continue
            bearing = pose.heading + rel_angle
            free_end = self._ray_end(pose.position, bearing,
                                     max(0.0, min(distance, 300.0) - 7.5))
            free.update(self.segment_cells(pose.position, free_end))
            if distance >= 295.0 or distance < 25.0:
                continue
            endpoint = self._ray_end(pose.position, bearing, distance)
            cell = self.world_to_cell(endpoint)
            if cell is None:
                continue
            entity = self._matching_entity(semantic_list, rel_angle, distance)
            if entity in ("DRONE", "BOMB"):
                dynamic.add(cell)
            elif entity != "DISPOSAL_CENTER":
                hits.add(cell)
        free.difference_update(hits | dynamic)
        changed = False
        for cell in free:
            old = int(self.states[cell])
            self.log_odds[cell] = max(-4.0, float(self.log_odds[cell]) - 0.35)
            self._refresh_state(cell)
            changed |= old != int(self.states[cell])
        for cell in hits:
            if step - int(self.last_hit_step[cell]) >= 3:
                self.hit_count[cell] = min(255, int(self.hit_count[cell]) + 1)
                self.last_hit_step[cell] = step
            if self.hit_count[cell] >= 3:
                old = int(self.states[cell])
                self.log_odds[cell] = min(4.0, float(self.log_odds[cell]) + 1.2)
                self._refresh_state(cell)
                changed |= old != int(self.states[cell])
        for cell in dynamic:
            old = self.dynamic_until[cell] >= step
            self.dynamic_until[cell] = step + 12
            changed |= not old
        self.updated_step = step
        if changed:
            self.revision += 1
        return changed

    @staticmethod
    def _matching_entity(semantic: Sequence[object], angle: float, distance: float) -> str:
        for item in semantic:
            try:
                kind = item.entity_type.name
                other_angle = float(item.angle)
                other_distance = float(item.distance)
            except (AttributeError, TypeError, ValueError):
                continue
            delta = (other_angle - angle + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) < 0.14 and abs(other_distance - distance) < max(15.0, 0.08 * distance):
                return str(kind)
        return ""

    def _refresh_state(self, cell: GridCell) -> None:
        value = float(self.log_odds[cell])
        self.states[cell] = (FREE if value <= -0.62 else
                             OCCUPIED if value >= 0.85 else UNKNOWN)

    def cell_status(self, cell: GridCell, step: int) -> int:
        if self.dynamic_until[cell] >= step:
            return OCCUPIED
        return int(self.states[cell])

    def check_segment(self, start: WorldPoint, end: WorldPoint,
                      clearance: float, step: int) -> str:
        cells = self.segment_cells(start, end)
        if not cells:
            return "BLOCKED"
        offsets = self._clearance_offsets_cache.get(clearance)
        if offsets is None:
            radius = max(0, math.ceil(clearance / self.resolution))
            offsets = tuple(
                (dr, dc)
                for dr in range(-radius, radius + 1)
                for dc in range(-radius, radius + 1)
                if math.hypot(dr, dc) * self.resolution <= clearance + 14.2
            )
            if len(self._clearance_offsets_cache) >= 16:
                self._clearance_offsets_cache.clear()
            self._clearance_offsets_cache[clearance] = offsets
        unknown = False
        for row, col in cells:
            for dr, dc in offsets:
                rr, cc = row + dr, col + dc
                if not (0 <= rr < self.rows and 0 <= cc < self.cols):
                    return "BLOCKED"
                state = self.cell_status((rr, cc), step)
                if state == OCCUPIED:
                    return "BLOCKED"
                unknown |= state == UNKNOWN
        return "UNKNOWN" if unknown else "SAFE"

    def unknown_count(self, center: WorldPoint, radius: float) -> int:
        cell = self.world_to_cell(center)
        if cell is None:
            return 0
        spread = math.ceil(radius / self.resolution)
        row, col = cell
        return sum(
            self.states[r, c] == UNKNOWN
            for r in range(max(0, row - spread), min(self.rows, row + spread + 1))
            for c in range(max(0, col - spread), min(self.cols, col + spread + 1))
            if math.hypot(r - row, c - col) * self.resolution <= radius
        )

    def risk_near(self, center: WorldPoint, radius: float, step: int) -> float:
        cell = self.world_to_cell(center)
        if cell is None:
            return 1.0
        spread = max(1, math.ceil(radius / self.resolution))
        row, col = cell
        cells = [self.cell_status((r, c), step)
                 for r in range(max(0, row - spread), min(self.rows, row + spread + 1))
                 for c in range(max(0, col - spread), min(self.cols, col + spread + 1))]
        return min(1.0, (cells.count(OCCUPIED) + 0.3 * cells.count(UNKNOWN)) /
                   max(1, len(cells)))

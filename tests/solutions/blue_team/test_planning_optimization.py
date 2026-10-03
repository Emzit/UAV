"""Safety and regression tests for bounded frontier search and plan caching."""

import math
import random
import unittest

import numpy as np

from swarm_rescue.solutions.blue_team.contracts import PathPlan, PoseEstimate
from swarm_rescue.solutions.blue_team.frontier import FrontierExplorer
from swarm_rescue.solutions.blue_team.occupancy_grid import (
    FREE, OCCUPIED, UNKNOWN, OccupancyGrid,
)
from swarm_rescue.solutions.blue_team.path_planner import GridPathPlanner


class PlanningOptimizationTests(unittest.TestCase):
    def test_open_frontier_needs_only_one_route(self):
        grid = OccupancyGrid((600.0, 600.0), 20.0)
        grid.states[7:23, 7:23] = FREE
        pose = PoseEstimate((0.0, 0.0), 0.0, 25.0, 0.01,
                            "GPS_FUSED", "world", 1, True)
        planner = GridPathPlanner(grid)
        original = planner.plan
        calls = 0

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        planner.plan = counted
        goal = FrontierExplorer().update(pose, grid, planner, 1, 9, 10)
        self.assertIsNotNone(goal)
        self.assertGreater(goal.position[0], 0.0)
        self.assertEqual(calls, 1)

    def test_frontier_tries_backup_candidates_when_early_routes_fail(self):
        grid = OccupancyGrid((600.0, 600.0), 20.0)
        grid.states[7:23, 7:23] = FREE
        pose = PoseEstimate((0.0, 0.0), 0.0, 25.0, 0.01,
                            "GPS_FUSED", "world", 1, True)
        planner = GridPathPlanner(grid)
        original = planner.plan
        calls = 0

        def fail_first_three(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls <= 3:
                return PathPlan("NO_PATH", (), math.inf, grid.revision,
                                1, "synthetic_block")
            return original(*args, **kwargs)

        planner.plan = fail_first_three
        goal = FrontierExplorer().update(pose, grid, planner, 1, 9, 10)
        self.assertIsNotNone(goal)
        self.assertGreaterEqual(calls, 4)

    def test_frontier_limits_astar_when_all_candidates_fail(self):
        grid = OccupancyGrid((600.0, 600.0), 20.0)
        grid.states[7:23, 7:23] = FREE
        pose = PoseEstimate((0.0, 0.0), 0.0, 25.0, 0.01,
                            "GPS_FUSED", "world", 1, True)
        planner = GridPathPlanner(grid)
        calls = 0

        def fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            return PathPlan("NO_PATH", (), math.inf, grid.revision,
                            1, "synthetic_block")

        planner.plan = fail
        self.assertIsNone(FrontierExplorer().update(pose, grid, planner,
                                                    1, 9, 10))
        self.assertLessEqual(calls, 12)

    def test_cached_offsets_match_original_segment_rule(self):
        grid = OccupancyGrid((400.0, 400.0), 20.0)
        rng = np.random.default_rng(7)
        grid.states[:] = rng.choice(
            [UNKNOWN, FREE, OCCUPIED], size=grid.states.shape,
            p=[0.15, 0.75, 0.10],
        )
        grid.dynamic_until[:] = rng.choice(
            [-1, 1, 3], size=grid.states.shape, p=[0.85, 0.10, 0.05],
        )

        def original_rule(start, end, clearance, step):
            cells = grid.segment_cells(start, end)
            if not cells:
                return "BLOCKED"
            radius = max(0, math.ceil(clearance / grid.resolution))
            unknown = False
            for row, col in cells:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if (math.hypot(dr, dc) * grid.resolution
                                > clearance + 14.2):
                            continue
                        rr, cc = row + dr, col + dc
                        if not (0 <= rr < grid.rows and 0 <= cc < grid.cols):
                            return "BLOCKED"
                        state = grid.cell_status((rr, cc), step)
                        if state == OCCUPIED:
                            return "BLOCKED"
                        unknown |= state == UNKNOWN
            return "UNKNOWN" if unknown else "SAFE"

        random_points = random.Random(19)
        for step in (1, 2, 4):
            for clearance in (0.0, 20.0, 30.0, 59.0, 94.0):
                for _ in range(50):
                    start = (random_points.uniform(-199, 199),
                             random_points.uniform(-199, 199))
                    end = (random_points.uniform(-199, 199),
                           random_points.uniform(-199, 199))
                    self.assertEqual(
                        grid.check_segment(start, end, clearance, step),
                        original_rule(start, end, clearance, step),
                    )

    def test_plan_cache_reuses_cells_then_invalidates_on_step_and_revision(self):
        grid = OccupancyGrid((600.0, 600.0), 20.0)
        grid.states[:] = FREE
        grid.states[3:22, 15] = OCCUPIED
        grid.updated_step = 1
        grid.revision = 1
        planner = GridPathPlanner(grid)
        original_risk = grid.risk_near
        risk_calls = 0

        def counted_risk(*args, **kwargs):
            nonlocal risk_calls
            risk_calls += 1
            return original_risk(*args, **kwargs)

        grid.risk_near = counted_risk
        args = ((-200.0, 0.0), (200.0, 0.0))
        first = planner.plan(*args, 1, clearance=30.0, max_expansions=350)
        self.assertEqual(first.status, "READY")
        self.assertGreater(risk_calls, 0)

        risk_calls = 0
        second = planner.plan(*args, 1, clearance=30.0, max_expansions=350)
        self.assertEqual(second, first)
        self.assertEqual(risk_calls, 0)

        risk_calls = 0
        planner.plan(*args, 2, clearance=30.0, max_expansions=350)
        self.assertGreater(risk_calls, 0)

        risk_calls = 0
        grid.revision += 1
        planner.plan(*args, 2, clearance=30.0, max_expansions=350)
        self.assertGreater(risk_calls, 0)

    def test_new_dynamic_obstacle_is_not_hidden_by_cached_plan(self):
        grid = OccupancyGrid((400.0, 400.0), 20.0)
        grid.states[:] = FREE
        grid.updated_step = 1
        planner = GridPathPlanner(grid)
        start, goal = (-100.0, 0.0), (100.0, 0.0)
        first = planner.plan(start, goal, 1, clearance=20.0)
        self.assertEqual(first.reason, "certified_direct")

        center_cell = grid.world_to_cell((0.0, 0.0))
        grid.dynamic_until[center_cell] = 1
        grid.revision += 1
        blocked = planner.plan(start, goal, 1, clearance=20.0)
        self.assertNotEqual(blocked.reason, "certified_direct")
        if blocked.status == "READY":
            points = (start,) + blocked.waypoints
            for a, b in zip(points, points[1:]):
                self.assertEqual(grid.check_segment(a, b, 20.0, 1), "SAFE")

        expired = planner.plan(start, goal, 2, clearance=20.0)
        self.assertEqual(expired.reason, "certified_direct")


if __name__ == "__main__":
    unittest.main()

"""分区覆盖的纯算法测试；用标准库运行，无需图形窗口。"""

import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))

from swarm_rescue.solutions.blue_team import (  # noqa: E402
    ExplorationConfig,
    ExplorationInput,
    ExplorationManager,
    PoseEstimate,
)


def make_input(step, drone_id=0, count=10, position=(-500.0, 0.0),
               size=(1250.0, 800.0), valid=True, carrying=False,
               has_bomb_task=False, frame_id="world", variance=25.0):
    pose = PoseEstimate(
        position=position,
        heading=0.0,
        position_variance=variance,
        heading_variance=0.005,
        source="GPS_DIRECT",
        frame_id=frame_id,
        step=step,
        valid=valid,
    )
    return ExplorationInput(
        pose=pose,
        step=step,
        world_size=size,
        drone_id=drone_id,
        drone_count=count,
        has_bomb_task=has_bomb_task,
        carrying=carrying,
        peer_leases=(),
    )


class ExplorationTests(unittest.TestCase):
    def test_ten_drones_get_separate_map01_strips(self):
        goals = []
        for drone_id in range(10):
            manager = ExplorationManager(ExplorationConfig())
            result = manager.update(make_input(1, drone_id=drone_id))
            self.assertEqual(result.status, "ACTIVE")
            self.assertIsNotNone(result.goal)
            goals.append(result.goal.position)
            self.assertTrue(-625 < result.goal.position[0] < 625)
            self.assertTrue(-400 < result.goal.position[1] < 400)
        self.assertEqual(len({goal[0] for goal in goals}), 10)
        self.assertLess(goals[0][0], goals[-1][0])

    def test_snake_route_alternates_vertical_direction(self):
        manager = ExplorationManager(ExplorationConfig())
        manager.update(make_input(1, count=1))
        points = [item[0] for item in manager.debug_snapshot()["points"]]
        self.assertGreaterEqual(len(points), 4)
        self.assertEqual(points[0][0], points[1][0])
        self.assertEqual(points[2][0], points[3][0])
        self.assertEqual(points[0][1], points[3][1])
        self.assertEqual(points[1][1], points[2][1])

    def test_reaching_a_waypoint_three_times_advances(self):
        manager = ExplorationManager(ExplorationConfig())
        first = manager.update(make_input(1))
        point = first.goal.position
        for step in (2, 3):
            result = manager.update(make_input(step, position=point))
            self.assertEqual(result.goal.goal_id, first.goal.goal_id)
        result = manager.update(make_input(4, position=point))
        self.assertEqual(result.status, "ACTIVE")
        self.assertNotEqual(result.goal.goal_id, first.goal.goal_id)
        self.assertEqual(manager.debug_snapshot()["points"][0][1], "VISITED")

    def test_stalled_point_is_temporarily_skipped(self):
        manager = ExplorationManager(ExplorationConfig())
        first = manager.update(make_input(1))
        for step in range(2, 46):
            manager.update(make_input(step))
        second = manager.update(make_input(46))
        self.assertEqual(second.status, "ACTIVE")
        self.assertNotEqual(second.goal.goal_id, first.goal.goal_id)
        point = manager.debug_snapshot()["points"][0]
        self.assertEqual(point[1], "BLOCKED_UNTIL")
        self.assertEqual(point[2], 166)

    def test_unusable_world_pose_or_size_falls_back(self):
        changes_list = (
            {"valid": False},
            {"position": None},
            {"frame_id": "local:d0:e0"},
            {"variance": 401.0},
            {"size": None},
        )
        for changes in changes_list:
            with self.subTest(changes=changes):
                manager = ExplorationManager(ExplorationConfig())
                decision = manager.update(make_input(1, **changes))
                self.assertEqual(decision.status, "LOCAL_FALLBACK")
                self.assertIsNone(decision.goal)

    def test_bomb_task_suspends_exploration(self):
        manager = ExplorationManager(ExplorationConfig())
        decision = manager.update(make_input(1, carrying=True))
        self.assertEqual(decision.status, "SUSPENDED")
        self.assertIsNone(decision.goal)
        decision = manager.update(make_input(2, has_bomb_task=True))
        self.assertEqual(decision.status, "SUSPENDED")

    def test_long_bomb_task_gap_does_not_mark_goal_stalled(self):
        manager = ExplorationManager(ExplorationConfig())
        first = manager.update(make_input(1))
        resumed = manager.update(make_input(200))
        self.assertEqual(resumed.goal.goal_id, first.goal.goal_id)

    def test_duplicate_step_is_idempotent_and_old_step_rejected(self):
        manager = ExplorationManager(ExplorationConfig())
        first = manager.update(make_input(5))
        duplicate = manager.update(make_input(5, position=(0.0, 0.0)))
        self.assertIs(duplicate, first)
        with self.assertRaises(ValueError):
            manager.update(make_input(4))

    def test_active_goal_keeps_creation_step(self):
        manager = ExplorationManager(ExplorationConfig())
        first = manager.update(make_input(1))
        second = manager.update(make_input(2))
        self.assertEqual(first.goal.goal_id, second.goal.goal_id)
        self.assertEqual(second.goal.created_step, 1)

    def test_invalidated_goal_is_skipped(self):
        manager = ExplorationManager(ExplorationConfig())
        first = manager.update(make_input(1))
        manager.invalidate_route(first.goal.goal_id, 1)
        second = manager.update(make_input(2))
        self.assertNotEqual(second.goal.goal_id, first.goal.goal_id)

    def test_no_lease_is_sent_in_coverage_only_phase(self):
        manager = ExplorationManager(ExplorationConfig())
        manager.update(make_input(1))
        self.assertIsNone(manager.lease_payload())
        x = manager.update(make_input(2)).goal.position[0]
        self.assertTrue(math.isfinite(x))


if __name__ == "__main__":
    unittest.main()

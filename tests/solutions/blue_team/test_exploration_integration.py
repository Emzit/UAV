"""不启动 GUI 的控制器探索分支冒烟测试。"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))

from swarm_rescue.simulation.utils.misc_data import MiscData  # noqa: E402
from swarm_rescue.solutions.my_drone_blue_basic import MyDroneBlueBasic  # noqa: E402


class ExplorationIntegrationTests(unittest.TestCase):
    def make_drone(self):
        drone = MyDroneBlueBasic(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        drone.elapsed_timestep = 2
        drone._blue_step = 1
        drone.grasped_bombs = lambda: []
        drone._run_lidar_escape_if_needed = lambda cmd: False
        drone._apply_lidar_avoidance = lambda cmd: None
        return drone

    def test_coverage_goal_reaches_existing_motion_layer(self):
        drone = self.make_drone()
        drone.measured_gps_position = lambda: (-500.0, 0.0)
        drone.measured_compass_angle = lambda: 0.0
        visited = []

        def fake_move(x, y, cmd, **kwargs):
            visited.append((x, y, kwargs))
            cmd["forward"] = 0.3
            return False

        drone._move_toward = fake_move
        result = drone._control_explore(drone._empty_command())
        self.assertEqual(drone._last_explore_decision.status, "ACTIVE")
        self.assertEqual(len(visited), 1)
        self.assertTrue(visited[0][2]["use_lidar"])
        self.assertEqual(result["forward"], 0.3)

    def test_missing_gps_uses_original_local_exploration(self):
        drone = self.make_drone()
        drone.measured_gps_position = lambda: None
        drone.measured_compass_angle = lambda: 0.0
        result = drone._control_explore(drone._empty_command())
        self.assertEqual(drone._last_explore_decision.status, "LOCAL_FALLBACK")
        self.assertGreater(result["forward"], 0.0)

    def test_main_control_enters_coverage_after_bomb_checks(self):
        drone = self.make_drone()
        drone.measured_gps_position = lambda: (-500.0, 0.0)
        drone.measured_compass_angle = lambda: 0.0
        drone._remember_home_position = lambda: None
        drone._ingest_peer_messages = lambda: None
        drone._observe_local_semantics = lambda: []
        drone._prune_expired_knowledge = lambda: None
        drone._record_breadcrumb = lambda **kwargs: None
        drone._move_toward = lambda x, y, cmd, **kwargs: cmd.update(
            forward=0.25
        ) or False
        result = drone.control()
        self.assertEqual(drone._mode, "explore")
        self.assertEqual(drone._last_explore_decision.status, "ACTIVE")
        self.assertEqual(result["forward"], 0.25)


if __name__ == "__main__":
    unittest.main()

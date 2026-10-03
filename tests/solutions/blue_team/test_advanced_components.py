"""Fast tests for sensor-only blue controller components (no GUI required)."""

import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))

from swarm_rescue.simulation.utils.misc_data import MiscData  # noqa: E402
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import (  # noqa: E402
    DroneSemanticSensor,
)
from swarm_rescue.solutions.blue_team.contracts import PoseEstimate  # noqa: E402
from swarm_rescue.solutions.blue_team.frontier import FrontierExplorer  # noqa: E402
from swarm_rescue.solutions.blue_team.localization import LocalizationManager  # noqa: E402
from swarm_rescue.solutions.blue_team.occupancy_grid import (  # noqa: E402
    FREE, OCCUPIED, OccupancyGrid,
)
from swarm_rescue.solutions.blue_team.path_planner import GridPathPlanner  # noqa: E402
from swarm_rescue.solutions.blue_team.return_path import ReturnNavigator  # noqa: E402
from swarm_rescue.solutions.blue_team.task_allocation import CostAllocator  # noqa: E402
from swarm_rescue.solutions.blue_team.traffic import PeerMotion, TrafficManager  # noqa: E402
from swarm_rescue.solutions.my_drone_blue_advanced import MyDroneBlueAdvanced  # noqa: E402
from swarm_rescue.solutions.my_drone_eval import drone_class_for_mode  # noqa: E402


class AdvancedComponentTests(unittest.TestCase):
    @staticmethod
    def _traffic_inputs(step=1, peers=(), hits=(), carrying=False):
        return dict(
            pose=PoseEstimate((0.0, 0.0), 0.0, 25.0, 0.01,
                              "GPS_FUSED", "world", step, True),
            step=step, carrying=carrying, mode="carry" if carrying else "explore",
            goal=(180.0, 0.0), peers=peers, drone_hits=hits,
            lidar_distances=(100.0, 100.0, 100.0, 100.0),
            lidar_angles=(-math.pi, -math.pi / 2, 0.0, math.pi / 2),
        )

    def test_carrier_has_priority_and_empty_drone_yields(self):
        peer = PeerMotion(1, (40.0, 0.0), True, 1)
        values = self._traffic_inputs(peers=(peer,), hits=((0.0, 28.0),))
        action = TrafficManager(2).update(**values)
        self.assertEqual(action.reason, "yield")
        self.assertLess(action.forward, 0.0)
        self.assertLess(action.lateral, 0.0)
        empty_peer = PeerMotion(1, (40.0, 0.0), False, 1)
        carrier_values = self._traffic_inputs(
            peers=(empty_peer,), hits=((0.0, 28.0),), carrying=True,
        )
        self.assertIsNone(TrafficManager(2).update(**carrier_values))

    def test_equal_priority_uses_drone_id_and_avoids_blocked_side(self):
        peer = PeerMotion(1, (40.0, 0.0), False, 25, math.pi)
        values = self._traffic_inputs(step=25, peers=(peer,),
                                      hits=((0.0, 28.0),))
        values["lidar_distances"] = (100.0, 25.0, 100.0, 100.0)
        values["desired_forward"] = 0.8
        action = TrafficManager(2).update(**values)
        self.assertGreater(action.lateral, 0.0)
        other = PeerMotion(2, (40.0, 0.0), False, 25, math.pi)
        values["peers"] = (other,)
        self.assertIsNone(TrafficManager(1).update(**values))

    def test_spawn_neighbors_and_same_direction_do_not_yield(self):
        spawn_peer = PeerMotion(1, (40.0, 0.0), False, 1, math.pi)
        values = self._traffic_inputs(peers=(spawn_peer,),
                                      hits=((0.0, 28.0),))
        values["desired_forward"] = 0.8
        self.assertIsNone(TrafficManager(2).update(**values))
        convoy = PeerMotion(1, (40.0, 0.0), False, 25, 0.0)
        values["step"] = 25
        values["peers"] = (convoy,)
        self.assertIsNone(TrafficManager(2).update(**values))

    def test_stalled_carrier_escapes_but_moving_drone_does_not(self):
        manager = TrafficManager(3)
        self.assertIsNone(manager.update(**self._traffic_inputs(
            step=1, carrying=True,
        )))
        action = manager.update(**self._traffic_inputs(
            step=25, carrying=True,
        ))
        self.assertTrue(action.reason.startswith("recover"))
        moving = TrafficManager(3)
        moving.update(**self._traffic_inputs(step=1, carrying=True))
        data = self._traffic_inputs(step=25, carrying=True)
        data["pose"] = PoseEstimate((20.0, 0.0), 0.0, 25.0, 0.01,
                                    "GPS_FUSED", "world", 25, True)
        self.assertIsNone(moving.update(**data))

    def test_odometry_detects_stall_despite_noisy_gps(self):
        manager = TrafficManager(3)
        for step in range(1, 25):
            data = self._traffic_inputs(step=step, carrying=True)
            data["pose"] = PoseEstimate(
                (15.0 if step % 2 else -15.0, 0.0), 0.0, 25.0, 0.01,
                "GPS_FUSED", "world", step, True,
            )
            data["odometry_values"] = (0.1, 0.0, 0.0)
            self.assertIsNone(manager.update(**data))
        data = self._traffic_inputs(step=25, carrying=True)
        data["odometry_values"] = (0.1, 0.0, 0.0)
        self.assertTrue(manager.update(**data).reason.startswith("recover"))

    def test_back_and_forth_motion_counts_as_no_net_progress(self):
        manager = TrafficManager(3)
        manager.update(**self._traffic_inputs(step=1, carrying=True))
        for step in range(2, 25):
            data = self._traffic_inputs(step=step, carrying=True)
            data["odometry_values"] = (
                1.0, 0.0 if step % 2 else math.pi, 0.0,
            )
            self.assertIsNone(manager.update(**data))
        data = self._traffic_inputs(step=25, carrying=True)
        data["odometry_values"] = (1.0, 0.0, 0.0)
        self.assertTrue(manager.update(**data).reason.startswith("recover"))

    def test_controller_applies_yield_without_disarming_grasper(self):
        drone = MyDroneBlueAdvanced(
            identifier=2,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        drone._blue_step = 1
        drone._mode = "explore"
        drone._localizer._latest = PoseEstimate(
            (0.0, 0.0), 0.0, 25.0, 0.01,
            "GPS_FUSED", "world", 1, True,
        )
        drone._peer_motion[1] = PeerMotion(1, (40.0, 0.0), True, 1)
        drone.semantic_values = lambda: [SimpleNamespace(
            entity_type=DroneSemanticSensor.TypeEntity.DRONE,
            angle=0.0, distance=28.0,
        )]
        drone.lidar_values = lambda: np.array((100.0, 100.0, 100.0, 100.0))
        drone.lidar_rays_angles = lambda: np.array(
            (-math.pi, -math.pi / 2, 0.0, math.pi / 2)
        )
        command = drone._empty_command()
        command.update(forward=0.8, grasper=1)
        drone._apply_traffic(command, False, np.array((0.0, 0.0, 0.0)))
        self.assertLess(command["forward"], 0.0)
        self.assertEqual(command["grasper"], 1)
        self.assertEqual(drone._last_traffic_reason, "yield")

    def test_localization_dead_reckoning_and_outlier_rejection(self):
        manager = LocalizationManager(0)
        self.assertTrue(manager.update(1, (0.0, 0.0), 0.0,
                                       (0.0, 0.0, 0.0)).valid)
        for step in range(2, 10):
            pose = manager.update(step, None, None, (2.0, 0.0, 0.0))
        self.assertAlmostEqual(pose.position[0], 16.0, delta=2.0)
        self.assertTrue(pose.valid)
        pose = manager.update(10, (500.0, 500.0), None, (0.0, 0.0, 0.0))
        self.assertLess(math.dist(pose.position, (16.0, 0.0)), 10.0)
        self.assertFalse(manager.reanchored)

    def test_unknown_and_wall_never_get_certified_shortcut(self):
        grid = OccupancyGrid((400.0, 400.0), 20.0)
        planner = GridPathPlanner(grid)
        start, goal = (-100.0, 0.0), (100.0, 0.0)
        self.assertEqual(grid.check_segment(start, goal, 20.0, 1), "UNKNOWN")
        self.assertNotEqual(planner.plan(start, goal, 1,
                                         clearance=20.0).status, "READY")
        grid.states[:] = FREE
        wall = grid.world_to_cell((0.0, 0.0))
        grid.states[wall] = OCCUPIED
        self.assertEqual(grid.check_segment(start, goal, 20.0, 1), "BLOCKED")
        route = planner.plan(start, goal, 1, clearance=20.0,
                             max_expansions=1000)
        self.assertEqual(route.status, "READY")
        self.assertGreater(route.geometric_length_px, math.dist(start, goal))

    def test_return_shortcut_requires_certified_clearance(self):
        grid = OccupancyGrid((600.0, 600.0), 20.0)
        nav = ReturnNavigator()
        nav.begin([(-100.0, 0.0), (0.0, 100.0), (100.0, 0.0)])
        first = nav.next_waypoint((100.0, 5.0), grid, 1)
        self.assertEqual(first, (0.0, 100.0))
        grid.states[:] = FREE
        second = nav.next_waypoint((100.0, 5.0), grid, 2)
        self.assertEqual(second, (-100.0, 0.0))

    def test_return_breadcrumb_loop_is_erased_only_when_safe(self):
        grid = OccupancyGrid((400.0, 400.0), 20.0)
        grid.states[:] = FREE
        path = [(-100.0, 0.0), (-50.0, 0.0), (0.0, 0.0),
                (0.0, 70.0), (-50.0, 70.0), (-90.0, 0.0),
                (100.0, 0.0)]
        nav = ReturnNavigator()
        nav.begin(path, grid, 1)
        self.assertLess(len(nav._route), len(path))
        self.assertEqual(nav._route[0], path[0])
        self.assertEqual(nav._route[-1], path[-1])

        grid.states[grid.world_to_cell((-100.0, 0.0))] = OCCUPIED
        blocked = ReturnNavigator()
        blocked.begin(path, grid, 1)
        self.assertGreaterEqual(len(blocked._route), len(nav._route))

    def test_bid_penalizes_low_health_and_cache_not_cross_drone(self):
        grid = OccupancyGrid((400.0, 400.0), 20.0)
        grid.states[:] = FREE
        allocator = CostAllocator(grid, GridPathPlanner(grid))
        near = PoseEstimate((-40.0, 0.0), 0.0, 25.0, 0.01,
                            "GPS_FUSED", "world", 1, True)
        far = PoseEstimate((-140.0, 0.0), 0.0, 25.0, 0.01,
                           "GPS_FUSED", "world", 1, True)
        target = (40.0, 0.0)
        good = allocator.bid(target, near, 1, 100, None, None)
        hurt = allocator.bid(target, near, 1, 12, None, None)
        distant = allocator.bid(target, far, 1, 100, None, None)
        self.assertLess(good, hurt)
        self.assertLess(good, distant)

    def test_bid_never_runs_astar(self):
        grid = OccupancyGrid((400.0, 400.0), 20.0)
        grid.states[:] = FREE
        planner = GridPathPlanner(grid)
        planner.plan = lambda *args, **kwargs: self.fail("bid called A*")
        allocator = CostAllocator(grid, planner)
        pose = PoseEstimate((-40.0, 0.0), 0.0, 25.0, 0.01,
                            "GPS_FUSED", "world", 1, True)
        self.assertIsInstance(allocator.bid((40.0, 0.0), pose, 1, 100,
                                            (100.0, 0.0), None), int)

    def test_one_large_frontier_ring_still_splits_left_and_right(self):
        grid = OccupancyGrid((600.0, 600.0), 20.0)
        grid.states[7:23, 7:23] = FREE
        pose = PoseEstimate((0.0, 0.0), 0.0, 25.0, 0.01,
                            "GPS_FUSED", "world", 1, True)
        planner = GridPathPlanner(grid)
        left = FrontierExplorer().update(pose, grid, planner, 1, 0, 10)
        right = FrontierExplorer().update(pose, grid, planner, 1, 9, 10)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertLess(left.position[0], 0.0)
        self.assertGreater(right.position[0], 0.0)

    def test_scout_switches_wall_end_after_no_forward_progress(self):
        grid = OccupancyGrid((600.0, 600.0), 20.0)
        grid.states[7:23, 7:23] = FREE
        pose = PoseEstimate((0.0, 0.0), 0.0, 25.0, 0.01,
                            "GPS_FUSED", "world", 1, True)
        explorer = FrontierExplorer()
        planner = GridPathPlanner(grid)
        first = explorer.update(pose, grid, planner, 1, 7, 10,
                                advance_direction=1.0)
        second = explorer.update(pose, grid, planner, 112, 7, 10,
                                 advance_direction=1.0)
        self.assertLess(first.position[1], 0.0)
        self.assertGreater(second.position[1], 0.0)
        progressed = PoseEstimate((100.0, 0.0), 0.0, 25.0, 0.01,
                                  "GPS_FUSED", "world", 140, True)
        explorer.update(progressed, grid, planner, 140, 7, 10,
                        advance_direction=1.0)
        self.assertEqual(explorer._sweep_sign, -1.0)

    def test_distant_scout_does_not_divert_for_far_near_home_bomb(self):
        drone = MyDroneBlueAdvanced(
            identifier=8,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10,
                               max_timestep_limit=2000),
        )
        drone._home_position = (-505.0, 0.0)
        drone._blue_step = 20
        drone._localizer._latest = PoseEstimate(
            (-450.0, 0.0), 0.0, 25.0, 0.01,
            "GPS_FUSED", "world", 20, True,
        )
        self.assertEqual(drone._scout_direction(), 1.0)
        far = (SimpleNamespace(distance=150.0), (-320.0, 0.0))
        self.assertIsNone(drone._choose_visible_bomb([far]))

    def test_right_side_scout_remains_active_until_near_far_boundary(self):
        drone = MyDroneBlueAdvanced(
            identifier=7,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10,
                               max_timestep_limit=2000),
        )
        drone._home_position = (-505.0, 0.0)
        drone._blue_step = 1100
        drone._localizer._latest = PoseEstimate(
            (520.0, -260.0), 0.0, 25.0, 0.01,
            "GPS_FUSED", "world", 1100, True,
        )
        self.assertEqual(drone._scout_direction(), 1.0)
        far_corner = (
            SimpleNamespace(distance=130.0), (540.0, -320.0),
        )
        self.assertEqual(drone._choose_visible_bomb([far_corner]), far_corner)

    def test_close_bomb_can_be_approached_laterally_without_spin_only(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        hit = SimpleNamespace(angle=0.6, distance=25.0)
        command = drone._control_chase_bomb(
            drone._empty_command(), (hit, (25.0, 15.0)),
        )
        self.assertEqual(command["grasper"], 1)
        self.assertGreater(command["forward"], 0.0)
        self.assertGreater(command["lateral"], 0.0)

    def test_dock_keeps_bomb_grasped_through_spin_and_retry(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        drone._localizer._latest = PoseEstimate(
            (-505.0, 178.0), math.pi / 2, 25.0, 0.01,
            "GPS_FUSED", "world", 1, True,
        )
        approach = drone._control_dock(drone._empty_command(), (-505.0, 210.0))
        self.assertGreater(approach["forward"], 0.0)
        self.assertEqual(approach["grasper"], 1)
        drone._localizer._latest = PoseEstimate(
            (-505.0, 190.0), math.pi / 2, 25.0, 0.01,
            "GPS_FUSED", "world", 2, True,
        )
        for _ in range(42):
            spin = drone._control_dock(drone._empty_command(), (-505.0, 210.0))
            self.assertEqual(spin["grasper"], 1)
        self.assertEqual(drone._dock_retry, 1)
        self.assertIsNone(drone._dock_target)

    def test_disposal_surface_uses_multiple_rays_not_nearest_corner(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        origin = (-430.0, 160.0)
        drone._use_fused = True
        drone._localizer._latest = PoseEstimate(
            origin, 0.0, 25.0, 0.01,
            "GPS_FUSED", "world", 1, True,
        )
        hits = []
        for x in (-550.0, -505.0, -455.0, -395.0):
            dx, dy = x - origin[0], 210.0 - origin[1]
            hits.append(SimpleNamespace(
                entity_type=DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER,
                angle=math.atan2(dy, dx), distance=math.hypot(dx, dy),
            ))
        drone.semantic_values = lambda: hits
        point, _ = drone._visible_disposal_surface()
        self.assertLess(point[0], -450.0)
        self.assertAlmostEqual(point[1], 210.0, delta=0.001)

    def test_entry_and_drop_verification(self):
        self.assertIs(drone_class_for_mode("rescue"), MyDroneBlueAdvanced)
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        drone._target_bomb = (50.0, 60.0)
        drone._verified_release = False
        drone._mark_current_target_cleared()
        self.assertIsNone(drone._last_cleared_bomb)
        drone._target_bomb = (50.0, 60.0)
        drone._verified_release = True
        drone._mark_current_target_cleared()
        self.assertEqual(drone._last_cleared_bomb, (50.0, 60.0))

    def test_old_reward_does_not_verify_a_new_failed_drop(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        drone._was_carrying = True
        drone._target_bomb = (50.0, 60.0)
        drone._carry_reward_baseline = 1.0
        drone.reward = 1.0
        drone.gps_is_disabled = lambda: False
        drone.gps_values = lambda: np.array((0.0, 0.0))
        drone.compass_is_disabled = lambda: False
        drone.compass_values = lambda: 0.0
        drone.odometer_values = lambda: np.array((0.0, 0.0, 0.0))
        drone.lidar_values = lambda: None
        drone.semantic_values = lambda: []
        drone.grasped_bombs = lambda: []
        drone._control_explore = lambda command: command
        drone.control()
        self.assertFalse(drone._verified_release)
        self.assertIsNone(drone._last_cleared_bomb)

    def test_controller_first_control_without_gui(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        drone.gps_is_disabled = lambda: False
        drone.gps_values = lambda: np.array((-500.0, 0.0))
        drone.compass_is_disabled = lambda: False
        drone.compass_values = lambda: 0.0
        drone.odometer_values = lambda: np.array((0.0, 0.0, 0.0))
        drone.lidar_values = lambda: None
        drone.semantic_values = lambda: []
        drone.grasped_bombs = lambda: []
        drone._run_lidar_escape_if_needed = lambda command: False
        result = drone.control()
        self.assertEqual(set(result), {"forward", "lateral", "rotation", "grasper"})
        self.assertEqual(result["grasper"], 0)
        self.assertEqual(drone.define_message_for_all()["kind"], "blue-basic-v1")

    def test_higher_id_releases_duplicate_grasp(self):
        drone = MyDroneBlueAdvanced(
            identifier=3,
            misc_data=MiscData(size_area=(600, 600), number_drones=4),
        )
        drone._blue_step = 10
        drone._target_bomb = (50.0, 40.0)
        drone._peer_claims[1] = ((52.0, 41.0), 10, True)
        drone._peer_motion[1] = PeerMotion(1, (8.0, 0.0), True, 10)
        drone._localizer._latest = PoseEstimate(
            (0.0, 0.0), 0.0, 25.0, 0.01,
            "GPS_FUSED", "world", 10, True,
        )
        self.assertTrue(drone._duplicate_carrier_should_yield())
        drone._peer_claims[1] = ((180.0, 40.0), 10, True)
        self.assertFalse(drone._duplicate_carrier_should_yield())

    def test_duplicate_carrier_command_actually_releases(self):
        drone = MyDroneBlueAdvanced(
            identifier=3,
            misc_data=MiscData(size_area=(600, 600), number_drones=4),
        )
        drone._blue_step = 11
        drone._was_carrying = True
        drone._target_bomb = (50.0, 40.0)
        drone._peer_claims[1] = ((52.0, 41.0), 11, True)
        drone._peer_motion[1] = PeerMotion(1, (8.0, 0.0), True, 11)
        drone.gps_is_disabled = lambda: False
        drone.gps_values = lambda: np.array((0.0, 0.0))
        drone.compass_is_disabled = lambda: False
        drone.compass_values = lambda: 0.0
        drone.odometer_values = lambda: np.array((0.0, 0.0, 0.0))
        drone.semantic_values = lambda: []
        drone.grasped_bombs = lambda: [object()]
        drone._control_carry = lambda cmd: {**cmd, "grasper": 1}
        self.assertEqual(drone.control()["grasper"], 0)

    def test_already_grasped_semantic_bomb_disarms_empty_drone(self):
        drone = MyDroneBlueAdvanced(
            identifier=3,
            misc_data=MiscData(size_area=(600, 600), number_drones=4),
        )
        drone.semantic_values = lambda: [SimpleNamespace(
            entity_type=DroneSemanticSensor.TypeEntity.BOMB,
            grasped=True, distance=24.0,
        )]
        self.assertTrue(drone._peer_held_bomb_is_near())

    def test_near_equal_bids_use_stable_drone_id_tie_break(self):
        drone = MyDroneBlueAdvanced(
            identifier=3,
            misc_data=MiscData(size_area=(600, 600), number_drones=5),
        )
        drone._blue_step = 10
        drone._bid = lambda point: 300
        target = (50.0, 40.0)
        drone._peer_claims[1] = (target, 10, False)
        drone._peer_bids[1] = (target, 330, 10)
        self.assertTrue(drone._claimed_by_preferred_peer(target))
        drone._peer_claims.clear()
        drone._peer_bids.clear()
        drone._peer_claims[4] = (target, 10, False)
        drone._peer_bids[4] = (target, 290, 10)
        self.assertFalse(drone._claimed_by_preferred_peer(target))
        drone._peer_bids[4] = (target, 200, 10)
        self.assertTrue(drone._claimed_by_preferred_peer(target))

    def test_carry_route_prefers_known_shorter_corridor(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(600, 600), number_drones=4),
        )
        drone._grid.states[:] = FREE
        drone._grid.states[7:19, 15] = OCCUPIED
        drone._blue_step = 1
        calls = 0
        original = drone._planner.plan

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        drone._planner.plan = counted
        start, home = (150.0, 0.0), (-150.0, 0.0)
        waypoint = drone._planned_carry_waypoint(start, home)
        self.assertIsNotNone(waypoint)
        self.assertGreater(waypoint[1], 0.0)
        self.assertEqual(calls, 1)
        drone._blue_step = 2
        self.assertEqual(drone._planned_carry_waypoint(start, home), waypoint)
        self.assertEqual(calls, 1)

    def test_map01_known_wall_gates_allow_carry_astar(self):
        grid = OccupancyGrid((1250.0, 800.0), 20.0)
        grid.states[:] = FREE
        # Test fixture based on the public Map01 wall layout. The controller
        # itself still receives only sensor-built occupancy cells.
        for x, low, high in ((-115, -80, 399), (285, -399, 140)):
            for y in range(low, high + 1, 5):
                grid.states[grid.world_to_cell((x, y))] = OCCUPIED
        for x in range(-618, -391, 10):
            for y in range(211, 394, 10):
                grid.states[grid.world_to_cell((x, y))] = OCCUPIED
        route = GridPathPlanner(grid).plan(
            (540.0, -320.0), (-505.0, 210.0), 1,
            clearance=MyDroneBlueAdvanced._CARRY_CLEARANCE,
            max_expansions=3500,
        )
        self.assertEqual(route.status, "READY")
        self.assertGreater(max(point[1] for point in route.waypoints), 160.0)
        self.assertLess(min(point[1] for point in route.waypoints), -100.0)

    def test_carry_route_rejects_newly_blocked_segment(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(400, 400), number_drones=2),
        )
        drone._grid.states[:] = FREE
        drone._grid.updated_step = 1
        start, target = (-100.0, 0.0), (100.0, 0.0)
        drone._blue_step = 1
        self.assertEqual(drone._planned_carry_waypoint(start, target), target)
        cell = drone._grid.world_to_cell((0.0, 0.0))
        drone._grid.dynamic_until[cell] = 2
        drone._grid.revision += 1
        drone._blue_step = 2
        self.assertIsNone(drone._planned_carry_waypoint(start, target))
        self.assertEqual(drone._carry_route, ())

    def test_failed_carry_route_is_not_replanned_each_step(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(400, 400), number_drones=2),
        )
        calls = 0

        def fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(status="NO_PATH", waypoints=())

        drone._planner.plan = fail
        for step in range(1, 20):
            drone._blue_step = step
            self.assertIsNone(drone._planned_carry_waypoint(
                (-100.0, 0.0), (100.0, 0.0),
            ))
        self.assertEqual(calls, 1)

    def test_carrying_with_uncertain_pose_keeps_grasper_and_moves_cautiously(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        drone._localizer._latest = PoseEstimate(
            (0.0, 0.0), 0.0, 2000.0, 0.3,
            "DEAD_RECKONING", "world", 1, False,
        )
        drone._run_lidar_escape_if_needed = lambda command: False
        drone._apply_lidar_avoidance = lambda command: None
        result = drone._control_carry(drone._empty_command())
        self.assertEqual(result["grasper"], 1)
        self.assertGreater(result["forward"], 0.0)
        self.assertLess(result["forward"], 0.4)

    def test_route_waypoint_progress_does_not_go_backwards(self):
        drone = MyDroneBlueAdvanced(
            identifier=0,
            misc_data=MiscData(size_area=(1250, 800), number_drones=10),
        )
        drone._grid.states[:] = FREE
        drone._mode = "navigate"
        drone.grasped_bombs = lambda: []
        drone.measured_gps_position = lambda: drone._localizer.latest.position
        drone.measured_compass_angle = lambda: 0.0
        drone.lidar_values = lambda: None
        drone._planned_goal = (180.0, 0.0)
        drone._planned_route = ((10.0, 0.0), (100.0, 0.0), (180.0, 0.0))
        drone._route_step = 0
        drone._localizer._latest = PoseEstimate(
            (0.0, 0.0), 0.0, 25.0, 0.01, "GPS_FUSED", "world", 1, True,
        )
        drone._move_toward(180.0, 0.0, drone._empty_command(), use_lidar=True)
        self.assertEqual(drone._waypoint_index, 1)
        drone._localizer._latest = PoseEstimate(
            (110.0, 0.0), 0.0, 25.0, 0.01, "GPS_FUSED", "world", 2, True,
        )
        drone._move_toward(180.0, 0.0, drone._empty_command(), use_lidar=True)
        self.assertEqual(drone._waypoint_index, 2)


if __name__ == "__main__":
    unittest.main()

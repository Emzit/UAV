import unittest
from unittest.mock import patch

from swarm_rescue.simulation.gui_map.playground import Playground
from swarm_rescue.simulation.reporting.timing_profile import TimingProfile


class TimingProfileTests(unittest.TestCase):
    def test_reports_phases_drones_and_wall_rate(self):
        profile = TimingProfile()
        with patch(
            "swarm_rescue.simulation.reporting.timing_profile.time.perf_counter",
            side_effect=[10.0, 10.2, 10.3, 10.6],
        ):
            profile.begin_step(1)
            profile.add_control(3, 0.05)
            profile.add_phase("physics", 0.10)
            profile.end_step()
            profile.begin_step(2)
            profile.add_control(3, 0.04)
            profile.add_phase("physics", 0.15)
            profile.end_step()
        profile.add_draw(0.02)
        profile.add_finalize(0.5)

        report = profile.format_report(interval_size=1)
        self.assertIn("2 steps", report)
        self.assertIn("physics: total=0.25s", report)
        self.assertIn("control: total=0.09s", report)
        self.assertIn("slow_control drone 3", report)
        self.assertIn("on_draw (1 calls)", report)
        self.assertIn("round_cleanup: 0.50s", report)
        self.assertIn("steps 1-1: 5.0 steps/s", report)
        self.assertIn("steps 2-2: 2.5 steps/s", report)

    def test_playground_step_records_inner_phases_without_changing_step_count(self):
        class FakeSpace:
            def __init__(self):
                self.calls = 0

            def step(self, _delta):
                self.calls += 1

        playground = object.__new__(Playground)
        playground.timing_profile = TimingProfile()
        playground._space = FakeSpace()
        playground._agents = []
        playground._timestep = 0
        playground._pre_step = lambda: None
        playground._post_step = lambda: None
        playground._apply_commands = lambda _commands: None
        playground._compute_observations = lambda: None

        playground.timing_profile.begin_step(1)
        playground.step(pymunk_steps=3)
        playground.timing_profile.end_step()

        self.assertEqual(playground._space.calls, 3)
        self.assertEqual(playground._timestep, 1)
        phases = playground.timing_profile.steps[0].phases
        self.assertTrue(
            {"pre_step", "apply_commands", "physics", "post_step"}
            <= phases.keys()
        )
        playground.timing_profile = None
        playground.step(pymunk_steps=3)
        self.assertEqual(playground._space.calls, 6)
        self.assertEqual(playground._timestep, 2)

    def test_observation_timings_separate_ray_and_agent_work(self):
        class FakeRay:
            calls = 0

            def update_sensors(self):
                self.calls += 1

        class FakeAgent:
            removed = False
            calls = 0

            def compute_observations(self):
                self.calls += 1

        playground = object.__new__(Playground)
        playground.timing_profile = TimingProfile()
        playground._ray_compute = FakeRay()
        playground._agents = [FakeAgent()]
        playground._agents_cache_needs_update = True

        playground.timing_profile.begin_step(1)
        playground._compute_observations()
        playground.timing_profile.end_step()

        self.assertEqual(playground._ray_compute.calls, 1)
        self.assertEqual(playground._agents[0].calls, 1)
        phases = playground.timing_profile.steps[0].phases
        self.assertIn("sensor_rays", phases)
        self.assertIn("sensor_agents", phases)

    def test_control_detail_is_nested_idempotent_and_survives_grid_replacement(self):
        class Grid:
            def check_segment(self, *_args):
                return "SAFE"

        class Planner:
            def __init__(self, grid):
                self.grid = grid

            def plan(self):
                return self.grid.check_segment((0, 0), (1, 1), 30, 1)

        class Frontier:
            def __init__(self, planner):
                self.planner = planner

            def update(self):
                return self.planner.plan()

        class Drone:
            _mode = "explore"

            def __init__(self):
                self._grid = Grid()
                self._planner = Planner(self._grid)
                self._frontier = Frontier(self._planner)

            def _control_explore(self):
                return self._frontier.update()

        drone = Drone()
        profile = TimingProfile()
        profile.begin_step(1)
        profile.instrument_drone(drone, 2)
        profile.instrument_drone(drone, 2)
        self.assertEqual(drone._control_explore(), "SAFE")
        profile.end_step()

        expected = {
            "explore/controller.explore",
            "explore/frontier.update",
            "explore/planner.astar",
            "explore/map.check_segment",
        }
        self.assertTrue(expected <= profile._detail_totals.keys())
        self.assertTrue(all(profile._detail_totals[key].calls == 1
                            for key in expected))
        parent = profile._detail_totals["explore/controller.explore"]
        self.assertGreaterEqual(parent.inclusive, parent.exclusive)

        drone._grid = Grid()
        profile.begin_step(2)
        profile.instrument_drone(drone, 2)
        self.assertEqual(
            drone._grid.check_segment((0, 0), (1, 1), 30, 2), "SAFE"
        )
        profile.end_step()
        self.assertEqual(
            profile._detail_totals["explore/map.check_segment"].calls, 2
        )
        self.assertIn("control detail hotspots", profile.format_report())

    def test_empty_before_first_update(self):
        self.assertIn("no simulation steps", TimingProfile().format_report())


if __name__ == "__main__":
    unittest.main()

"""Tests for ground-truth API guards during competition control steps."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.ground_truth_guard import (
    GroundTruthInControlError,
    assert_ground_truth_allowed,
    control_active,
    run_control,
)
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData


class _GpsDrone(DroneAbstract):
    def define_message_for_all(self):
        return None

    def control(self) -> CommandsDict:
        return {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0,
        }


class _CheatingDrone(_GpsDrone):
    def control(self) -> CommandsDict:
        _ = self.true_position()
        return super().control()


class _CheatingMap(MapAbstract):
    def __init__(self, drone_type):
        super().__init__(drone_type=drone_type)
        self._size_area = (200, 200)
        self._number_drones = 1
        self._drones_pos = [((0, 0), 0)]
        self._drones = []
        self._playground = ClosedPlayground(size=self._size_area)
        misc = MiscData(
            size_area=self._size_area,
            number_drones=self._number_drones,
            max_timestep_limit=self._max_timestep_limit,
            max_walltime_limit=self._max_walltime_limit,
        )
        drone = drone_type(identifier=0, misc_data=misc)
        self._drones.append(drone)
        self._playground.add(drone, self._drones_pos[0])


def test_assert_ground_truth_blocks_in_control_during_competition():
    from swarm_rescue.simulation.drone import ground_truth_guard

    token = ground_truth_guard._control_active.set(True)
    try:
        with pytest.raises(GroundTruthInControlError, match="true_position"):
            assert_ground_truth_allowed("true_position", competition_mode=True)
    finally:
        ground_truth_guard._control_active.reset(token)


def test_assert_ground_truth_allows_outside_control():
    assert_ground_truth_allowed("true_position", competition_mode=True)


def test_assert_ground_truth_allows_in_control_when_not_competition():
    from swarm_rescue.simulation.drone import ground_truth_guard

    token = ground_truth_guard._control_active.set(True)
    try:
        assert_ground_truth_allowed("true_position", competition_mode=False)
    finally:
        ground_truth_guard._control_active.reset(token)


def test_control_active_only_inside_run_control():
    drone = _GpsDrone()
    assert control_active() is False
    run_control(drone)
    assert control_active() is False


def test_true_position_allowed_outside_control_with_competition_mode():
    the_map = _CheatingMap(drone_type=_GpsDrone)
    drone = the_map.drones[0]
    drone.set_competition_mode(True)
    pos = drone.true_position()
    assert pos is not None


def test_true_position_blocked_in_control_with_competition_mode():
    the_map = _CheatingMap(drone_type=_CheatingDrone)
    drone = the_map.drones[0]
    drone.set_competition_mode(True)
    with pytest.raises(GroundTruthInControlError, match="true_position"):
        run_control(drone)


def test_true_position_allowed_in_control_without_competition_mode():
    the_map = _CheatingMap(drone_type=_CheatingDrone)
    drone = the_map.drones[0]
    drone.set_competition_mode(False)
    run_control(drone)


def test_self_position_always_disabled():
    the_map = _CheatingMap(drone_type=_GpsDrone)
    drone = the_map.drones[0]

    class _PositionDrone(_GpsDrone):
        def control(self) -> CommandsDict:
            _ = self.position
            return super().control()

    the_map._drones[0] = _PositionDrone(identifier=0, misc_data=MiscData(
        size_area=the_map.size_area,
        number_drones=1,
        max_timestep_limit=the_map.max_timestep_limit,
        max_walltime_limit=the_map.max_walltime_limit,
    ))
    drone = the_map.drones[0]
    with pytest.raises(Exception, match="Function Disabled"):
        run_control(drone)

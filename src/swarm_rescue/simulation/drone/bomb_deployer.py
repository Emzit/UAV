from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from swarm_rescue.simulation.drone.controller import PlaceBombController
from swarm_rescue.simulation.drone.device import Device
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.utils.pose import Pose

if TYPE_CHECKING:
    from swarm_rescue.simulation.drone.drone_part import DronePart
    from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract

MIN_BOMB_SEPARATION = 24.0


class BombDeployer(Device):
    """
    Device for placing bombs on the map (red team).
    """

    def __init__(self, anchor: DronePart, **kwargs):
        super().__init__(anchor=anchor, **kwargs)
        self.place_controller = PlaceBombController("place_bomb")
        self._inventory = 0
        self._enabled = False
        self._placement_map: Optional[MapAbstract] = None
        self._previous_command = 0

    @property
    def inventory(self) -> int:
        return self._inventory

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, the_map: MapAbstract, initial_count: int) -> None:
        self._placement_map = the_map
        self._inventory = max(0, int(initial_count))
        self._enabled = True
        self._previous_command = 0

    def _can_place_now(self) -> bool:
        agent = self._anchor.agent
        if self._inventory <= 0 or not self._enabled or self._placement_map is None:
            return False
        if agent.grasper.grasped_bombs:
            return False
        if agent.grasper.can_grasp:
            return False
        return True

    def _too_close_to_existing(self, x: float, y: float) -> bool:
        for bomb in self._placement_map._bombs:
            pos = bomb.true_position()
            dist = math.hypot(pos[0] - x, pos[1] - y)
            if dist < MIN_BOMB_SEPARATION:
                return True
        return False

    def _try_place_bomb(self) -> bool:
        if not self._can_place_now():
            return False

        agent = self._anchor.agent
        the_map = self._placement_map
        disposal_center = the_map.disposal_center
        if disposal_center is None:
            return False

        # Use true physical state to place the bomb at the exact location
        # in the playground (avoids GPS/compass sensor noise affecting the
        # exported JSON for the blue team).
        pos = agent.true_position()
        theta = agent.true_angle()
        x, y = float(pos[0]), float(pos[1])

        if self._too_close_to_existing(x, y):
            return False

        bomb = Bomb(disposal_center=disposal_center)
        # Red-team feature demo: keep newly-placed bombs from being pushed by
        # drones during placement (avoids "moving bombs" artifact).
        bomb.ignore_drone_collisions = True

        the_map._bombs.append(bomb)
        the_map._number_bombs = len(the_map._bombs)
        init_coords = ((x, y), theta)
        the_map.playground.add(bomb, init_coords)
        bomb.add_pose_to_path(Pose(np.array([x, y])))

        self._inventory -= 1
        return True

    def apply_commands(self) -> None:
        command_value = int(self.place_controller.command_value)
        rising_edge = command_value == 1 and self._previous_command == 0
        self._previous_command = command_value
        if rising_edge:
            self._try_place_bomb()

    def reset(self) -> None:
        self._previous_command = 0
        super().reset()

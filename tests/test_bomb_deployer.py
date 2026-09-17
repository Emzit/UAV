from unittest.mock import MagicMock

import numpy as np
import pytest

from swarm_rescue.simulation.drone.bomb_deployer import BombDeployer, MIN_BOMB_SEPARATION
from swarm_rescue.simulation.drone.controller import PlaceBombController


def test_place_bomb_controller_valid_commands():
    ctrl = PlaceBombController("place_bomb")
    assert ctrl.default == 0
    assert 1 in ctrl.valid_commands


def test_bomb_deployer_enable_sets_inventory():
    deployer = object.__new__(BombDeployer)
    deployer.place_controller = PlaceBombController("place_bomb")
    deployer._inventory = 0
    deployer._enabled = False
    deployer._placement_map = None
    deployer._previous_command = 0

    the_map = MagicMock()
    deployer.enable(the_map, 3)
    assert deployer.inventory == 3
    assert deployer.enabled is True
    assert deployer._placement_map is the_map


def test_min_bomb_separation_constant():
    assert MIN_BOMB_SEPARATION >= 20


def test_drone_part_applies_bomb_deployer_commands():
    from swarm_rescue.simulation.drone.drone_part import DronePart
    from unittest.mock import MagicMock

    deployer = MagicMock(spec=BombDeployer)
    part = MagicMock(spec=DronePart)
    part.devices = [deployer]

    DronePart.apply_commands(part)
    deployer.apply_commands.assert_called_once()

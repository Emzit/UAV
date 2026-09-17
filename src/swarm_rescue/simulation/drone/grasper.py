from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import pymunk

from swarm_rescue.simulation.drone.controller import GrasperController
from swarm_rescue.simulation.drone.device import Device
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.utils.definitions import CollisionTypes

if TYPE_CHECKING:
    from swarm_rescue.simulation.drone.drone_part import DronePart


class Grasper(Device):
    """
    Device for grasping bombs in the simulation.
    """

    def __init__(
            self, anchor: DronePart, max_grasped: Optional[int] = None, **kwargs
    ):
        super().__init__(anchor=anchor, **kwargs)

        self.grasp_controller = GrasperController("grasper")
        self._grasped_bombs: List[Bomb] = []
        self._grasp_joints: Dict[Bomb, List[pymunk.Constraint]] = {}
        self._can_grasp = False
        self._max_grasped = max_grasped

    @property
    def can_grasp(self) -> bool:
        return self._can_grasp

    @property
    def grasped_bombs(self) -> List[Bomb]:
        return self._grasped_bombs

    def grasps(self, bomb: Bomb) -> None:
        assert self._can_grasp

        if bomb not in self._grasped_bombs:

            if self._max_grasped and self._max_grasped <= len(self._grasped_bombs):
                return

            self._grasped_bombs.append(bomb)
            self._add_joints(bomb)
            bomb.grasped_by.append(self)

            for sensor in self._anchor.agent.external_sensors:
                if sensor.invisible_grasped:
                    sensor.add_to_temporary_invisible(bomb)

    @property
    def _collision_type(self):
        return CollisionTypes.GRASPER

    def _add_joints(self, bomb: Bomb) -> None:
        assert self._anchor

        joint_position = 0.3 * self._anchor.pm_body.position + 0.7 * bomb.pm_body.position

        joint = pymunk.PivotJoint(self._anchor.pm_body, bomb.pm_body, tuple(joint_position)) # type: ignore[arg-type]
        joint.collide_bodies = False

        grasp_joints = [joint]
        self._grasp_joints[bomb] = grasp_joints
        self._anchor.playground.space.add(*grasp_joints)

    def _release_grasping(self) -> None:
        for bomb in list(self._grasped_bombs):
            self.release(bomb)

        self._can_grasp = False

        assert not self._grasped_bombs
        assert not self._grasp_joints

    def release(self, bomb: Bomb) -> None:
        bomb.grasped_by.remove(self)

        if self._anchor and hasattr(self._anchor, 'agent'):
            for sensor in self._anchor.agent.external_sensors:
                if sensor.invisible_grasped:
                    sensor.remove_from_temporary_invisible(bomb)

        joints = self._grasp_joints.pop(bomb)
        if self._anchor and hasattr(self._anchor, 'playground') and self._anchor.playground:
            self._anchor.playground.space.remove(*joints)

        self._grasped_bombs.remove(bomb)

    def reset(self) -> None:
        self._release_grasping()
        super().reset()

    def apply_commands(self) -> None:
        command_value = self.grasp_controller.command_value

        if not command_value:
            self._release_grasping()
        else:
            self._can_grasp = True

import math
from typing import Tuple

import numpy as np
import pymunk

from swarm_rescue.resources import path_resources
from swarm_rescue.simulation.drone.controller import CenteredContinuousController
from swarm_rescue.simulation.elements.physical_element import PhysicalElement
from swarm_rescue.simulation.utils.constants import LINEAR_SPEED_RATIO_BOMB
from swarm_rescue.simulation.utils.definitions import CollisionTypes
from swarm_rescue.simulation.utils.definitions import LINEAR_FORCE
from swarm_rescue.simulation.utils.path import Path
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.simulation.utils.utils import clamp
from swarm_rescue.simulation.utils.utils import normalize_angle


class Bomb(PhysicalElement):
    """
    A bomb in the simulation. Used with a DisposalCenter: when a Bomb contacts
    its associated DisposalCenter, it is disposed and the transporting drone
    receives a reward.
    """

    def __init__(
        self,
        disposal_center,
        linear_ratio: float = LINEAR_SPEED_RATIO_BOMB,
        **kwargs,
    ):
        """
        Initialize a Bomb.

        Args:
            disposal_center: The associated DisposalCenter.
            linear_ratio (float): Ratio for linear speed.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            mass=100,
            filename=path_resources + "/bomb.png",
            shape_approximation="hull",
            radius=12,
            **kwargs,
        )

        for pm_shape in self._pm_shapes:
            pm_shape.elasticity = 0.5
            pm_shape.friction = 0.9

        self.disposal_center = disposal_center
        self.graspable = True

        self.forward_controller = CenteredContinuousController(name="forward")
        self.add_device(self.forward_controller)

        self.lateral_controller = CenteredContinuousController(name="lateral")
        self.add_device(self.lateral_controller)

        self.linear_ratio = LINEAR_FORCE * linear_ratio

        self.path = Path()
        self.reverse = False
        self.goal_index = 0

        self.pose = Pose(np.asarray(self.true_position()), self.true_angle())

    @property
    def _collision_type(self) -> int:
        return CollisionTypes.BOMB

    @property
    def reward(self) -> float:
        return 1

    def set_path(self, path: Path) -> None:
        self.path = path

    def clear_path(self) -> None:
        self.path = Path()

    def add_pose_to_path(self, pose: Pose) -> None:
        self.path.append(pose)

    def pre_step(self) -> None:
        super().pre_step()
        self.pose = Pose(np.asarray(self.true_position()), self.true_angle())
        self.compute_movement()

    def compute_movement(self) -> None:
        cmd_forward, cmd_lateral = self.follow_path()

        cmd_forward = max(min(cmd_forward, 1.0), -1.0)
        cmd_lateral = max(min(cmd_lateral, 1.0), -1.0)

        sqr_norm = cmd_forward ** 2 + cmd_lateral ** 2
        if sqr_norm > 1.0:
            norm = math.sqrt(sqr_norm)
            cmd_forward = cmd_forward / norm
            cmd_lateral = cmd_lateral / norm

        self._pm_body.apply_force_at_world_point(
            force=pymunk.Vec2d(cmd_forward, cmd_lateral) * self.linear_ratio,
            point=tuple(self.true_position())
        )

        self._pm_body.angular_velocity = -0.02 * (self.true_angle())

    def follow_path(self) -> Tuple[float, float]:
        if self.path.length() == 0:
            return 0, 0

        RADIUS_ARRIVED = 10.0

        goal_pose = self.path.get(index=self.goal_index)
        vector_to_goal = goal_pose.position - self.pose.position
        dist_to_goal = math.sqrt(vector_to_goal[0] ** 2 + vector_to_goal[1] ** 2)

        if dist_to_goal < RADIUS_ARRIVED:
            if self.path.length() == 1:
                return 0, 0
            elif (not self.reverse) and self.goal_index < (self.path.length() - 1):
                self.goal_index += 1
            elif (not self.reverse) and self.goal_index == (self.path.length() - 1):
                self.goal_index -= 1
                self.reverse = True
            elif self.reverse and self.goal_index > 0:
                self.goal_index -= 1
            elif self.reverse and self.goal_index == 0:
                self.goal_index += 1
                self.reverse = False

        abs_goal_direction = np.arctan2(vector_to_goal[1], vector_to_goal[0])

        intensity = 0.5
        long_force = float(np.cos(abs_goal_direction))
        lat_force = float(np.sin(abs_goal_direction))
        if dist_to_goal < 10:
            force_intensity = intensity * math.atan(dist_to_goal / 10) * 2 / np.pi
        else:
            force_intensity = intensity / max(abs(long_force), abs(lat_force))
        long_force *= force_intensity
        lat_force *= force_intensity

        long_force = clamp(long_force, -1.0, 1.0)
        lat_force = clamp(lat_force, -1.0, 1.0)

        return long_force, lat_force

    def true_position(self) -> np.ndarray:
        return np.array(self._pm_body.position)

    def true_angle(self) -> float:
        return normalize_angle(self._pm_body.angle)

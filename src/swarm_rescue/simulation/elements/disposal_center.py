from typing import Tuple

import arcade
import pymunk

from swarm_rescue.resources import path_resources
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.elements.physical_element import PhysicalElement
from swarm_rescue.simulation.gui_map.collision_handlers import get_colliding_entities
from swarm_rescue.simulation.gui_map.playground import Playground
from swarm_rescue.simulation.utils.definitions import CollisionTypes


def bomb_disposal_center_collision(arbiter: pymunk.Arbiter, _, data):
    """
    Handles collision between a bomb and a disposal center.
    """
    playground: Playground = data["playground"]
    bomb, disposal_center = get_colliding_entities(playground, arbiter)

    assert isinstance(bomb, Bomb)
    assert isinstance(disposal_center, DisposalCenter)

    if bomb.disposal_center == disposal_center:
        disposal_center.activate(bomb)

    return True


class DisposalCenter(PhysicalElement):
    """
    Disposal center: when a Bomb reaches its paired disposal center, rewards
    the drone and removes the bomb.
    """

    def __init__(self, size: Tuple[int, int], **kwargs):
        filename = path_resources + "/disposal_center.png"
        width = size[0]
        height = size[1]
        orig_x = int((800 - width) / 2)
        orig_y = int((800 - height) / 2)
        texture: arcade.Texture = arcade.load_texture(file_name=filename,
                                                      x=orig_x,
                                                      y=orig_y,
                                                      width=width,
                                                      height=height)
        super().__init__(texture=texture, **kwargs)

        for pm_shape in self._pm_shapes:
            pm_shape.elasticity = 0.1
            pm_shape.friction = 0.0

        self._quantity_rewards = None
        self._count_rewards = 0

    @property
    def _collision_type(self):
        return CollisionTypes.DISPOSAL_CENTER

    def activate(self, entity: Bomb) -> None:
        """
        Dispose a bomb, reward the drone, and remove the bomb from the playground.
        """
        if self._playground is None:
            raise ValueError("DisposalCenter is not associated with a playground.")

        grasped_by_list = entity.grasped_by.copy()
        grasped_by_size = len(entity.grasped_by)

        if grasped_by_list:
            for part in grasped_by_list:
                agent = part.agent
                agent.reward += entity.reward / grasped_by_size
                agent.grasper.reset()
        else:
            agent = self._playground.get_closest_drone(self)
            agent.reward += entity.reward

        self._playground.remove(entity)

from __future__ import annotations

from typing import TYPE_CHECKING

import pymunk

from swarm_rescue.simulation.drone.grasper import Grasper
from swarm_rescue.simulation.drone.drone_base import DroneBase
from swarm_rescue.simulation.elements.bomb import Bomb

if TYPE_CHECKING:
    from swarm_rescue.simulation.gui_map.playground import Playground


def get_colliding_entities(playground: "Playground", arbiter: pymunk.Arbiter):
    """
    Retrieve the two entities involved in a collision from the arbiter.

    Args:
        playground (Playground): The playground instance.
        arbiter (pymunk.Arbiter): The collision arbiter.

    Returns:
        tuple: The two colliding entities.
    """
    shape_1, shape_2 = arbiter.shapes
    entity_1 = playground.get_entity_from_shape(shape_1)
    entity_2 = playground.get_entity_from_shape(shape_2)

    return entity_1, entity_2


def grasper_grasps_bomb(arbiter: pymunk.Arbiter, _, data):
    """
    Handle the event where a grasper attempts to grasp a bomb.
    """
    playground: Playground = data["playground"]
    grasper, bomb = get_colliding_entities(playground, arbiter)

    assert isinstance(grasper, Grasper)

    if grasper.can_grasp:
        grasper.grasps(bomb)

    return True


def drone_collision_bomb_ignore_place(
    arbiter: pymunk.Arbiter, _, data
) -> bool:
    """
    Ignore DRONE <-> BOMB collisions when the bomb is produced by the red
    team placer (demo/placement phase).

    This prevents the physics engine from pushing newly-placed bombs away
    from their intended coordinates during placement.
    """
    playground: Playground = data["playground"]
    ent1, ent2 = get_colliding_entities(playground, arbiter)

    # Only handle DRONE/BOMB collisions.
    if not ((isinstance(ent1, DroneBase) and isinstance(ent2, Bomb)) or
            (isinstance(ent1, Bomb) and isinstance(ent2, DroneBase))):
        return True

    bomb = ent1 if isinstance(ent1, Bomb) else ent2
    if getattr(bomb, "ignore_drone_collisions", False):
        return False
    return True

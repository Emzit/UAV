"""
This program can be launched directly.
Example of how to control several drones
"""

import math
import pathlib
import random
import sys
from typing import List, Type



# Insert the 'src' directory, located two levels up from the current script,
# into sys.path. This ensures Python can find project-specific modules
# (e.g., 'swarm_rescue') when the script is run from a subfolder like 'examples/'.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData


class MyDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def process_semantic_sensor(self):
        detection_semantic = self.semantic_values()

        print("********************************************")
        if detection_semantic is not None:
            for data in detection_semantic:
                if data.distance > 60:
                    continue

                if data.entity_type == DroneSemanticSensor.TypeEntity.WALL:
                    print("type: wall, angle: {:.2f}, d: {:.1f}"
                          .format(data.angle, data.distance))
                elif data.entity_type == DroneSemanticSensor.TypeEntity.OTHER:
                    print("type: other, angle: {:.2f}, d: {:.1f}"
                          .format(data.angle, data.distance))
                elif (data.entity_type ==
                      DroneSemanticSensor.TypeEntity.BOMB):
                    print("type: bomb, angle: {:.2f}, d: {:.1f}"
                          .format(data.angle, data.distance))
                elif data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    print("type: drone, angle: {:.2f}, d: {:.1f}"
                          .format(data.angle, data.distance))

    def control(self) -> CommandsDict:
        if self.identifier == 0:
            self.process_semantic_sensor()


class MyMap(MapAbstract):
    def __init__(self, drone_type: Type[DroneAbstract]):
        super().__init__(drone_type=drone_type)

        # PARAMETERS MAP
        self._size_area = (700, 700)

        # BOMBS
        self._number_bombs = 30
        self._bombs_pos = []
        self._bombs: List[Bomb] = []

        start_area = (0.0, 0.0)
        nb_per_side = math.ceil(math.sqrt(float(self._number_bombs)))
        dist_inter_bomb = 100.0
        sx = start_area[0] - (nb_per_side - 1) * 0.5 * dist_inter_bomb
        sy = start_area[1] - (nb_per_side - 1) * 0.5 * dist_inter_bomb

        for i in range(self._number_bombs):
            x = sx + (float(i) % nb_per_side) * dist_inter_bomb
            y = sy + math.floor(float(i) / nb_per_side) * dist_inter_bomb
            pos = ((x, y), random.uniform(-math.pi, math.pi))
            self._bombs_pos.append(pos)

        # POSITIONS OF THE DRONES
        self._number_drones = 20
        self._drones_pos = []
        for i in range(self._number_drones):
            pos = ((random.uniform(-self._size_area[0] / 2,
                                   self._size_area[0] / 2),
                    random.uniform(-self._size_area[1] / 2,
                                   self._size_area[1] / 2)),
                   random.uniform(-math.pi, math.pi))
            self._drones_pos.append(pos)

        self._drones: List[DroneAbstract] = []

        self._playground = ClosedPlayground(size=self._size_area)

        # POSITIONS OF THE BOMBS
        for i in range(self._number_bombs):
            bomb = Bomb(disposal_center=None)
            self._bombs.append(bomb)
            pos = self._bombs_pos[i]
            self._playground.add(bomb, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones,
                             max_timestep_limit=self._max_timestep_limit,
                             max_walltime_limit=self._max_walltime_limit)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            self._playground.add(drone, self._drones_pos[i])


def main():
    the_map = MyMap(drone_type=MyDrone)

    gui = GuiSR(the_map=the_map,
                draw_semantic_rays=True,
                use_keyboard=True,
                use_mouse_measure=True,
                enable_visu_noises=False,
                )
    gui.run()


if __name__ == '__main__':
    main()

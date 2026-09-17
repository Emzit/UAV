import gc
import math
import pathlib
import random
import sys
from typing import List, Type

import numpy as np

# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case simulation.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.reporting.evaluation import EvalConfig, EvalPlan, ZonesConfig
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.pose import Pose

from swarm_rescue.maps.walls_05 import add_walls, add_boxes


class Map05(MapAbstract):

    def __init__(
        self,
        drone_type: Type[DroneAbstract],
        zones_config: ZonesConfig = (),
        number_drones: int | None = None,
    ):
        super().__init__(drone_type, zones_config, number_drones=number_drones)
        self._max_timestep_limit = 7200
        self._max_walltime_limit = 240  # In seconds

        # PARAMETERS MAP
        self._size_area = (1200, 1200)

        # 无人机出生区域中心点
        self._start_area_drones = (65, -15)

        self._disposal_center = DisposalCenter(size=(139, 205))
        self._disposal_center_pos = ((-122, 54), 0)

        self._bombs_pos = []
        self._bombs_path = []
        self._number_bombs = len(self._bombs_pos)
        self._bombs: List[Bomb] = []

        # POSITIONS OF THE DRONES
        self._number_drones = self._number_drones_override if self._number_drones_override is not None else 10
        start_area_drones = self._start_area_drones
        nb_per_side = math.ceil(math.sqrt(float(self._number_drones)))
        dist_inter_drone = 40.0
        sx = start_area_drones[0] - (nb_per_side - 1) * 0.5 * dist_inter_drone
        sy = start_area_drones[1] - (nb_per_side - 1) * 0.5 * dist_inter_drone

        self._drones_pos = []
        for i in range(self._number_drones):
            x = sx + (float(i) % nb_per_side) * dist_inter_drone
            y = sy + math.floor(float(i) / nb_per_side) * dist_inter_drone
            angle = random.uniform(-math.pi, math.pi)
            self._drones_pos.append(((x, y), angle))

        self._drones: List[DroneAbstract] = []
        self._playground = ClosedPlayground(size=self._size_area)

        self._playground.add(self._disposal_center, self._disposal_center_pos)

        add_walls(self._playground)
        add_boxes(self._playground)

        self._explored_map.initialize_walls(self._playground)

        # POSITIONS OF THE BOMBS
        for i in range(self._number_bombs):
            bomb = Bomb(disposal_center=self._disposal_center)
            self._bombs.append(bomb)
            init_pos = (self._bombs_pos[i], 0)
            self._playground.add(bomb, init_pos)

            bomb.add_pose_to_path(Pose(np.array(init_pos[0])))
            if i < len(self._bombs_path):
                for pt in self._bombs_path[i]:
                    bomb.add_pose_to_path(Pose(np.array(list(pt))))

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
    eval_plan = EvalPlan()
    eval_config = EvalConfig(map_name="Map05", zones_config=())
    eval_plan.add(eval_config=eval_config)

    for one_eval in eval_plan.list_eval_config:
        gc.collect()
        map_class = globals().get(one_eval.map_name)
        the_map = map_class(drone_type=DroneMotionless, zones_config=one_eval.zones_config)

        gui = GuiSR(the_map=the_map, use_mouse_measure=True)
        gui.run()


if __name__ == '__main__':
    main()

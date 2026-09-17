"""
This program can be launched directly.
Example of how to use semantic sensor, grasping and dropping
"""

import math
import pathlib
import random
import sys
from enum import Enum
from typing import Optional, List, Type

import numpy as np

# Insert the 'src' directory, located two levels up from the current script,
# into sys.path. This ensures Python can find project-specific modules
# (e.g., 'swarm_rescue') when the script is run from a subfolder like 'examples/'.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.utils import normalize_angle, circular_mean


class MyDroneSemantic(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_BOMB = 1
        GRASPING_BOMB = 2
        SEARCHING_DISPOSAL_CENTER = 3
        DROPPING_AT_DISPOSAL_CENTER = 4

    def __init__(self,
                 identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier,
                         display_lidar_graph=False,
                         **kwargs)
        # The state is initialized to searching bomb
        self.state = self.Activity.SEARCHING_BOMB

        # Those values are used by the random control function
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self) -> CommandsDict:
        command: CommandsDict = {"forward": 0.0,
                                 "lateral": 0.0,
                                 "rotation": 0.0,
                                 "grasper": 0}

        found_bomb, found_disposal_center, command_semantic = (
            self.process_semantic_sensor())

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.SEARCHING_BOMB and found_bomb:
            self.state = self.Activity.GRASPING_BOMB

        elif (self.state is self.Activity.GRASPING_BOMB and
              self.grasper.grasped_bombs):
            self.state = self.Activity.SEARCHING_DISPOSAL_CENTER

        elif (self.state is self.Activity.GRASPING_BOMB and
              not found_bomb):
            self.state = self.Activity.SEARCHING_BOMB

        elif (self.state is self.Activity.SEARCHING_DISPOSAL_CENTER and
              found_disposal_center):
            self.state = self.Activity.DROPPING_AT_DISPOSAL_CENTER

        elif (self.state is self.Activity.DROPPING_AT_DISPOSAL_CENTER and
              not self.grasper.grasped_bombs):
            self.state = self.Activity.SEARCHING_BOMB

        elif (self.state is self.Activity.DROPPING_AT_DISPOSAL_CENTER and
              not found_disposal_center):
            self.state = self.Activity.SEARCHING_DISPOSAL_CENTER

        print("state: {}, can_grasp: {}, grasped bombs: {}"
              .format(self.state.name,
                      self.grasper.can_grasp,
                      self.grasper.grasped_bombs))

        ##########
        # COMMANDS FOR EACH STATE
        # Searching randomly, but when a disposal center or bomb is
        # detected, we use a special command
        ##########
        if self.state is self.Activity.SEARCHING_BOMB:
            command = self.control_random()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_BOMB:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_DISPOSAL_CENTER:
            command = self.control_random()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_DISPOSAL_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided

    def control_random(self):
        """
        The Drone will move forward and turn for a random angle when an
        obstacle is hit
        """
        command_straight = {"forward": 0.5,
                            "rotation": 0.0}

        command_turn = {"forward": 0.0,
                        "rotation": 1.0}

        collided = self.process_lidar_sensor()

        self.counterStraight += 1

        if collided and not self.isTurning and self.counterStraight > 100:
            self.isTurning = True
            self.angleStopTurning = random.uniform(-math.pi, math.pi)

        diff_angle = normalize_angle(
            self.angleStopTurning - self.measured_compass_angle())
        if self.isTurning and abs(diff_angle) < 0.2:
            self.isTurning = False
            self.counterStraight = 0

        if self.isTurning:
            return command_turn
        else:
            return command_straight

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move
        towards a bomb or the disposal center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_bomb = False
        if (self.state is self.Activity.SEARCHING_BOMB
            or self.state is self.Activity.GRASPING_BOMB) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the bomb detected is held by nobody
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.BOMB and
                        not data.grasped):
                    found_bomb = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among bombs detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_disposal_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.SEARCHING_DISPOSAL_CENTER
            or self.state is self.Activity.DROPPING_AT_DISPOSAL_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.DISPOSAL_CENTER):
                    found_disposal_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 30)

            if found_disposal_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_disposal_center or found_bomb:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_disposal_center and is_near:
            command["forward"] = 0.0
            command["rotation"] = -1.0

        return found_bomb, found_disposal_center, command


class MyMapSemantic(MapAbstract):
    def __init__(self, drone_type: Type[DroneAbstract]):
        super().__init__(drone_type=drone_type)

        # PARAMETERS MAP
        self._size_area = (400, 400)

        self._disposal_center = DisposalCenter(size=(100, 100))
        self._disposal_center_pos = ((0, 150), 0)

        # BOMBS
        self._number_bombs = 20
        self._bombs_pos = []
        self._bombs: List[Bomb] = []

        bomb_area = (0.0, -30.0)
        nb_per_side = math.ceil(math.sqrt(float(self._number_bombs)))
        dist_inter_bomb = 60.0
        sx = bomb_area[0] - (nb_per_side - 1) * 0.5 * dist_inter_bomb
        sy = bomb_area[1] - (nb_per_side - 1) * 0.5 * dist_inter_bomb

        for i in range(self._number_bombs):
            x = sx + (float(i) % nb_per_side) * dist_inter_bomb
            y = sy + math.floor(float(i) / nb_per_side) * dist_inter_bomb
            pos = ((x, y), random.uniform(-math.pi, math.pi))
            self._bombs_pos.append(pos)

        # POSITIONS OF THE DRONES
        self._number_drones = 1
        self._drones_pos = [((-100, 100), random.uniform(-math.pi, math.pi))]
        self._drones = []

        self._playground = ClosedPlayground(size=self._size_area)

        self._playground.add(self._disposal_center, self._disposal_center_pos)

        # POSITIONS OF THE BOMBS
        for i in range(self._number_bombs):
            bomb = Bomb(disposal_center=self._disposal_center)
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
    the_map = MyMapSemantic(drone_type=MyDroneSemantic)

    # draw_semantic_rays : enable the visualization of the semantic rays
    gui = GuiSR(the_map=the_map,
                draw_semantic_rays=True,
                use_keyboard=False,
                )
    gui.run()


if __name__ == '__main__':
    main()

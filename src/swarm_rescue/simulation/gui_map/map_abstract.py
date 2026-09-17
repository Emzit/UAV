from abc import ABC
from typing import Any, Dict, List, Type, Union, Optional

import numpy as np

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.gui_map.playground import Playground
from swarm_rescue.simulation.reporting.evaluation import ZonesConfig
from swarm_rescue.simulation.reporting.explored_map import ExploredMap
from swarm_rescue.simulation.utils.constants import DRONE_INITIAL_HEALTH
from swarm_rescue.simulation.utils.pose import Pose


class MapAbstract(ABC):
    """
    The MapAbstract class is an abstract class that serves as a blueprint for
    constructing different types of maps used in the directory "maps".
    """

    def __init__(
        self,
        drone_type: Type[DroneAbstract],
        zones_config: ZonesConfig = (),
        number_drones: Optional[int] = None,
    ):
        """
        Initialize the MapAbstract.

        Args:
            drone_type (Type[DroneAbstract]): The type of drone to use.
            zones_config (ZonesConfig): Configuration for special zones.
        """
        self._playground: Optional[Playground] = None
        self._drone_type = drone_type
        self._explored_map = ExploredMap()
        self._size_area = None
        self._zones_config = zones_config
        self._number_drones_override = number_drones
        self._drones: List[DroneAbstract] = []
        # '_number_drones' is the number of drones that will be generated in
        # the map
        self._number_drones = None
        # '_max_timestep_limit' is the number of timesteps after which the
        # session will end.
        self._max_timestep_limit = None
        # 'max_walltime_limit' is the elapsed time (in seconds) after which the
        # session will end.
        self._max_walltime_limit = None  # In seconds
        # 'number_bombs' is the number of bombs that should
        # be retrieved by the drones.
        self._number_bombs = None
        self._bombs: List[Bomb] = []
        self._disposal_center = None

        self._return_area = None

    @property
    def bombs(self) -> List[Bomb]:
        """Bombs currently on the playground."""
        return self._bombs

    @property
    def disposal_center(self):
        """Disposal center associated with bombs on this map, if any."""
        return self._disposal_center

    def clear_bombs(self) -> None:
        """Remove all bombs from the playground and reset bomb counters."""
        if self._playground is None:
            self._bombs = []
            self._number_bombs = 0
            return
        for bomb in list(self._bombs):
            if bomb in self._playground.elements:
                self._playground.remove(bomb)
        self._bombs = []
        self._number_bombs = 0

    def spawn_bombs(
        self,
        bomb_entries: List[Dict[str, Any]],
        disposal_center=None,
    ) -> None:
        """
        Spawn bombs from normalized entry dicts (x, y, theta, optional path).

        Args:
            bomb_entries: List of dicts with keys x, y, theta, optional path.
            disposal_center: DisposalCenter instance; defaults to map's center.
        """
        if self._playground is None:
            raise ValueError("Cannot spawn bombs: playground is not initialized.")

        center = disposal_center if disposal_center is not None else self._disposal_center
        if center is None:
            raise ValueError(
                "Cannot spawn bombs: map has no disposal center. "
                "Use a map that defines _disposal_center."
            )

        for entry in bomb_entries:
            x = float(entry["x"])
            y = float(entry["y"])
            theta = float(entry.get("theta", 0.0))
            bomb = Bomb(disposal_center=center)
            self._bombs.append(bomb)
            init_pos = ((x, y), theta)
            self._playground.add(bomb, init_pos)
            bomb.add_pose_to_path(Pose(np.array(init_pos[0])))
            path = entry.get("path")
            if path:
                for pt in path:
                    bomb.add_pose_to_path(Pose(np.array(list(pt))))

        self._number_bombs = len(self._bombs)

    def remove_nearest_bomb(self, x: float, y: float, max_dist: float = 20.0) -> bool:
        """
        Remove the nearest bomb around world position (x, y).

        Args:
            x: World x coordinate.
            y: World y coordinate.
            max_dist: Maximum accepted distance to remove.

        Returns:
            True if one bomb was removed, else False.
        """
        if self._playground is None or not self._bombs:
            return False

        target = np.array([x, y], dtype=float)
        nearest_bomb = None
        nearest_dist = float("inf")
        for bomb in self._bombs:
            pos = np.asarray(bomb.true_position(), dtype=float)
            dist = float(np.linalg.norm(pos - target))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_bomb = bomb

        if nearest_bomb is None or nearest_dist > max_dist:
            return False

        if nearest_bomb in self._playground.elements:
            self._playground.remove(nearest_bomb)
        self._bombs.remove(nearest_bomb)
        self._number_bombs = len(self._bombs)
        return True

    @property
    def playground(self) -> Union[Playground, Type[None]]:
        """
        Returns the playground instance.

        Returns:
            Playground or None: The playground.
        """
        return self._playground

    @property
    def drones(self) -> List[DroneAbstract]:
        """
        Returns the list of drones in the map.

        Returns:
            List[DroneAbstract]: The drones.
        """
        return self._drones

    @property
    def number_drones(self) -> int:
        """
        Returns the number of drones in the map.

        Returns:
            int: Number of drones.
        """
        return self._number_drones

    @property
    def max_timestep_limit(self) -> int:
        """
        Returns the maximum timestep limit.

        Returns:
            int: Maximum timestep limit.
        """
        return self._max_timestep_limit

    @property
    def max_walltime_limit(self) -> int:
        """
        Returns the maximum walltime limit in seconds.

        Returns:
            int: Maximum walltime limit.
        """
        return self._max_walltime_limit

    @property
    def number_bombs(self) -> int:
        """
        Returns the number of bombs to be rescued.

        Returns:
            int: Number of bombs.
        """
        return self._number_bombs

    @property
    def size_area(self):
        """
        Returns the size of the area.

        Returns:
            Any: The size of the area.
        """
        return self._size_area

    @property
    def zones_config(self) -> ZonesConfig:
        """
        Returns the configuration for special zones.

        Returns:
            ZonesConfig: The zones configuration.
        """
        return self._zones_config

    @property
    def explored_map(self) -> ExploredMap:
        """
        Returns the explored map object.

        Returns:
            ExploredMap: The explored map.
        """
        return self._explored_map

    def compute_score_health_returned(self) -> float:
        """
        The method calculates a health score for returned UAVs, usually at the
        end of a run. This score is calculated only on a zone called the
        ‘return area’ if it exists. If it does not exist, the score is
        calculated over the entire map.  The score is a number between 0 and 1.
        A value of 1 means that all the drones have been returned and are in
        perfect health.

        Returns:
            float: The health score (0 to 1).
        """
        total_health = 0
        if self._return_area:
            total_health = self._return_area.compute_total_health_returned()
        else:
            for drone in self.drones:
                total_health += drone.drone_health

        mean_total_health = total_health / self.number_drones
        score_total_health = mean_total_health / DRONE_INITIAL_HEALTH
        return score_total_health

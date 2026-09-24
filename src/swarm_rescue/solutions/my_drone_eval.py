"""
Submission entrypoint helpers used by the launcher.
"""

from swarm_rescue.simulation.reporting.team_mode import TeamMode
from swarm_rescue.solutions.my_drone_blue_basic import MyDroneBlueBasic
from swarm_rescue.solutions.my_drone_place_example import MyDronePlaceExample


def drone_class_for_mode(mode: str):
    """
    Return the concrete drone class for the given team mode.
    """
    team_mode = TeamMode.from_string(mode)
    if team_mode == TeamMode.PLACE:
        return MyDronePlaceExample
    return MyDroneBlueBasic

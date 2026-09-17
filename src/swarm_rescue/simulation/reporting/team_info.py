from __future__ import annotations

import os
from typing import Optional

import yaml


class TeamInfo:
    """
    Parses and stores team information from a YAML file.

    Attributes:
        team_number (int): Team number.
        team_number_str_padded (str): Team number as zero-padded string.
        team_name (str): Team name.
        team_members (str): Team members.
    """

    def __init__(self, yaml_path: Optional[str] = None):
        """
        Initialize TeamInfo by loading from YAML file.

        Args:
            yaml_path: Optional path to team_info.yml. Defaults to solutions/team_info.yml.
        """
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__), '../..', 'solutions', 'team_info.yml'
            )

        with open(yaml_path, 'r') as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        self.team_number = int(config.get("team_number"))
        self.team_number_str = str(self.team_number)
        self.team_number_str_padded = str(self.team_number).zfill(3)
        self.team_name = str(config.get("team_name"))
        self.team_members = str(config.get("team_members"))
        print("The team '{}' n°{}, with {}".format(self.team_name,
                                                   self.team_number,
                                                   self.team_members))
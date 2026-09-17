import os
from datetime import datetime

from pathlib2 import Path


class ResultPathCreator:
    """
    Responsible for creating a path for saving results for a team.
    """

    def __init__(self, team_info):
        """
        Initializes the ResultPathCreator object. It creates the directory for
        storing the results.

        Args:
            team_info: Team information object.
        """

        self._team_info = team_info
        exact_path = os.getenv("SWARM_RESCUE_RESULT_PATH")
        if exact_path:
            self._directory = str(Path(exact_path).parent)
            self._result_path = str(Path(exact_path))
            self._make_dir()
            return

        # Write results inside the repository by default (sandbox-safe).
        # Users may override with SWARM_RESCUE_RESULTS_DIR if desired.
        env_dir = os.getenv("SWARM_RESCUE_RESULTS_DIR")
        if env_dir:
            self._directory = str(Path(env_dir))
        else:
            repo_root = Path(__file__).resolve().parents[4]
            self._directory = str(repo_root / "results_swarm_rescue")
        self._result_path = None

        self._create_path_name()
        self._make_dir()

    def _create_path_name(self) -> None:
        """
        Create the result path name based on the team and current date/time.
        """
        date = datetime.now()

        # Create the folder to keep results of the mission
        self._result_path = (self._directory
                             + f'/team{self._team_info.team_number_str_padded}'
                               f'_{date.strftime("%y%m%d_%Hh%Mmin%Ss")}')

    def _make_dir(self) -> None:
        """
        Create the necessary directories for saving results.
        """
        try:
            os.makedirs(self._directory, exist_ok=True)
        except FileExistsError as error:
            print(error)

        try:
            os.makedirs(self._result_path, exist_ok=True)
        except FileExistsError as error:
            print(error)

    @property
    def path(self) -> str:
        """
        Returns the result path.

        Returns:
            str: The result path.
        """
        return self._result_path
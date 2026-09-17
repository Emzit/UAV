import argparse
import gc
import os
import sys
import traceback
from typing import Tuple, Optional, Any

import numpy as np
import yaml

from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.reporting.data_saver import DataSaver
from swarm_rescue.simulation.reporting.bombs_io import (
    bombs_from_data,
    load_bombs_file,
    exploration_comparison_image_output_path,
    exploration_map_output_path,
    placed_bombs_output_path,
    save_placed_bombs,
    save_exploration_comparison_image,
    save_exploration_map,
    save_wall_destruction_image,
    validate_bombs_file_for_eval,
    wall_destruction_image_output_path,
)
from swarm_rescue.simulation.reporting.evaluation import EvalConfig, EvalPlan, ZonesConfig
from swarm_rescue.simulation.reporting.result_path_creator import ResultPathCreator
from swarm_rescue.simulation.reporting.place_score_manager import (
    PlaceScoreManager,
    load_place_score_config,
    walls_and_boundary_grid,
)
from swarm_rescue.simulation.reporting.score_manager import ScoreManager
from swarm_rescue.simulation.reporting.team_info import TeamInfo
from swarm_rescue.simulation.reporting.team_mode import TeamMode
from swarm_rescue.simulation.utils.constants import DRONE_INITIAL_HEALTH
from swarm_rescue.tools.map_exploration_scoring import scorer_for_walltime

from swarm_rescue.maps.map_01 import Map01
from swarm_rescue.maps.map_02 import Map02
from swarm_rescue.maps.map_03 import Map03
from swarm_rescue.maps.map_04 import Map04
from swarm_rescue.maps.map_05 import Map05

from swarm_rescue.solutions.my_drone_eval import drone_class_for_mode


class Launcher:
    """
    The Launcher class is responsible for running a simulation of drone rescue
    sessions. It creates an instance of the map with a specified zone type,
    which automatically constructs its playground during initialization, and
    initializes a GUI with the map. It then runs the GUI, allowing the user to
    interact with it. After the GUI finishes, it calculates the score for the
    exploration of the map and saves the images and data related to the round.

    Attributes:
        team_info (TeamInfo): Stores team information.
        eval_plan (EvalPlan): The evaluation plan.
        eval_plan_ok (bool): Whether the evaluation plan is valid.
        number_drones (Optional[int]): Number of drones in the simulation.
        max_timestep_limit (Optional[int]): Maximum number of time steps.
        max_walltime_limit (Optional[int]): Maximum wall time.
        number_bombs (Optional[int]): Number of bombs.
        size_area (Optional[Any]): Size of the simulation area.
        score_manager (Optional[ScoreManager]): Score manager instance.
        video_capture_enabled (bool): Whether video capture is enabled.
        result_path (Optional[str]): Path for results.
        data_saver (DataSaver): Data saver instance.
    """

    team_info: TeamInfo
    eval_plan: EvalPlan
    eval_plan_ok: bool
    number_drones: Optional[int]
    max_timestep_limit: Optional[int]
    max_walltime_limit: Optional[int]
    number_bombs: Optional[int]
    size_area: Optional[Any]
    score_manager: Optional[ScoreManager]
    video_capture_enabled: bool
    result_path: Optional[str]
    data_saver: DataSaver
    team_mode: TeamMode
    default_bombs_file: Optional[str]

    def __init__(
        self,
        config_path: Optional[str] = None,
        team_mode: Optional[str] = None,
        default_bombs_file: Optional[str] = None,
        manual_bomb_edit: bool = False,
        manual_bombs_in: Optional[str] = None,
        manual_bombs_out: Optional[str] = None,
    ) -> None:
        """
        Initializes the Launcher.

        Args:
            config_path (Optional[str]): Path to YAML configuration file for evaluation plan.
        """
        self.team_info = TeamInfo()

        # Create an EvalPlan from YAML configuration if provided
        self.eval_plan = EvalPlan()
        self.eval_plan_ok = True
        if config_path:
            self.eval_plan.from_yaml(config_path)
            if not self.eval_plan.list_eval_config:
                print(f"\nError: Could not load evaluation plan {config_path} !")
                self.eval_plan_ok = False
        else:
            print("\nError: No evaluation plan provided.")
            print("Please pass a valid evaluation plan YAML file with -c/--config, e.g.:")
            print("  -c config/competition_rescue_eval_plan.yml   # blue/rescue example")
            print("  -c config/competition_place_eval_plan.yml    # red/place example")
            self.eval_plan_ok = False
            exit(1)

        self.team_mode = self.eval_plan.team_mode
        if team_mode is not None:
            self.team_mode = TeamMode.from_string(team_mode)
        self.default_bombs_file = default_bombs_file
        self.manual_bomb_edit = manual_bomb_edit
        self.manual_bombs_in = manual_bombs_in
        self.manual_bombs_out = manual_bombs_out
        self._last_placed_bombs_file: Optional[str] = None

        if self.team_mode is not None:
            print(f"Default team mode: {self.team_mode.value}")
        else:
            print("Default team mode: <not set>")
        self.eval_plan.pretty_print()
        # Check if all eval configurations are valid
        for eval_config in self.eval_plan.list_eval_config:
            map_class = globals().get(eval_config.map_name)

            # Check if the class was found in the global namespace
            if not map_class:
                # If the class is not found, print a warning and skip this configuration
                print(f"Error: Unknown map type '{eval_config.map_name}' in evaluation plan!")
                print(f"If the '{eval_config.map_name}' class exists, please check that it is imported in launcher.py.")
                self.eval_plan_ok = False
                exit(1)

        self.number_drones = None
        self.max_timestep_limit = None
        self.max_walltime_limit = None
        self.number_bombs = None
        self.size_area = None

        self.score_manager = None

        # Red-team (place) scoring config comes from the optional
        # ``place_scoring`` section of the evaluation-plan YAML. It carries the
        # score weights and wall-destruction parameters (blast radius, boundary
        # wall weight), so the wall score stored in score.json is computed with
        # the same parameters the offline aggregation will use.
        _place_scoring_data = self._load_place_scoring_data(config_path)
        self.place_score_manager = PlaceScoreManager(
            load_place_score_config(_place_scoring_data)
        )

        # Set this value to True to generate stat data and pdf report
        stat_saving_enabled = self.eval_plan.stat_saving_enabled
        # Set this value to True to generate a video of the mission
        self.video_capture_enabled = self.eval_plan.video_capture_enabled
        # Set this value to True to record entity states (default recording method)
        self.state_recording_enabled = self.eval_plan.state_recording_enabled

        self.result_path = None
        needs_bombs_json_output = any(
            self._resolve_team_mode(ec) == TeamMode.PLACE
            for ec in self.eval_plan.list_eval_config
        )
        if (stat_saving_enabled or self.video_capture_enabled
                or self.state_recording_enabled
                or self.team_mode == TeamMode.PLACE or needs_bombs_json_output):
            rpc = ResultPathCreator(self.team_info)
            self.result_path = rpc.path
        self.data_saver = DataSaver(team_info=self.team_info,
                                    result_path=self.result_path,
                                    enabled=stat_saving_enabled)

    @staticmethod
    def _load_place_scoring_data(config_path: Optional[str]) -> Optional[dict]:
        """Extract the ``place_scoring`` section from the plan YAML, if any."""
        if not config_path or not os.path.exists(config_path):
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return None
        return data.get("place_scoring")

    def _resolve_team_mode(self, eval_config: EvalConfig) -> TeamMode:
        if eval_config.team_mode is not None:
            return eval_config.team_mode
        return self.team_mode

    def _resolve_bombs_file(self, eval_config: EvalConfig) -> Optional[str]:
        bombs_file = eval_config.bombs_file or self.default_bombs_file
        if bombs_file in ("auto", "last") and self._last_placed_bombs_file:
            return self._last_placed_bombs_file
        return bombs_file

    @staticmethod
    def _sync_rescue_bomb_total(my_gui, score_manager: ScoreManager) -> int:
        """Refresh the rescue bomb total from the GUI after a round.

        Manual bomb edit mode (``--manual-bomb-edit``) adds or imports bombs
        inside the GUI, after the round's ScoreManager was built from the
        map's initial count (0 on the demo maps). The GUI's count is
        authoritative: it is the value the round's own termination test used.
        For rounds loaded from a bombs file the refresh is a no-op.

        Args:
            my_gui: The finished GuiSR instance.
            score_manager: The round's ScoreManager to update.

        Returns:
            int: The authoritative total number of bombs.
        """
        total = my_gui.total_number_bombs
        if score_manager is not None:
            score_manager.total_number_bombs = total
        return total

    def _configure_map_for_team(
        self,
        the_map,
        eval_config: EvalConfig,
        round_team_mode: TeamMode,
    ) -> None:
        bombs_file = self._resolve_bombs_file(eval_config)

        if round_team_mode == TeamMode.PLACE:
            the_map.clear_bombs()
            for drone in the_map.drones:
                if drone._misc_data is not None:
                    drone._misc_data.team_mode = TeamMode.PLACE
                drone.enable_bomb_deployer(
                    the_map,
                    eval_config.initial_bombs_per_drone,
                )
            return

        if bombs_file:
            data = load_bombs_file(bombs_file)
            validate_bombs_file_for_eval(data, eval_config)
            the_map.clear_bombs()
            the_map.spawn_bombs(bombs_from_data(data))

    def one_round(
        self,
        eval_config: EvalConfig,
        num_round: int,
        hide_solution_output: bool = False,
        headless: bool = False,
        competition_mode: bool = False,
    ) -> Optional[Tuple[float, float, int, int, float, int, float, float, bool, bool]]:
        """
        Runs a single round of the session.

        It creates an instance of the map class with the specified
        eval_config, which automatically constructs its playground during
        initialization, and initializes a GUI with the map. It then runs the
        GUI, which allows the user to interact with it. After the GUI finishes,
        it calculates the score for the exploration of the map and saves the
        images and data related to the round.

        Args:
            eval_config (EvalConfig): The evaluation configuration.
            num_round (int): The round number.
            hide_solution_output (bool): Whether to hide solution output.
            headless (bool): Whether to run in headless mode.
            competition_mode (bool): If True, disable ground-truth APIs inside control().

        Returns:
            Optional[Tuple]: Various statistics and results from the round, or None if map class not found.
        """

        # Retrieve the class object from the global namespace using its name
        map_class = globals().get(eval_config.map_name)

        # Check if the class was found in the global namespace
        if not map_class:
            # If the class is not found, print a warning and skip this configuration
            print(f"Warning: Unknown map type '{eval_config.map_name}', skipping configuration")
            return None

        round_team_mode = self._resolve_team_mode(eval_config)
        drone_type = drone_class_for_mode(round_team_mode.value)
        map_kwargs = {
            "drone_type": drone_type,
            "zones_config": eval_config.zones_config,
        }
        if eval_config.number_drones is not None:
            map_kwargs["number_drones"] = eval_config.number_drones
        the_map = map_class(**map_kwargs)
        self._configure_map_for_team(the_map, eval_config, round_team_mode)

        self.number_drones = the_map.number_drones
        self.max_timestep_limit = the_map.max_timestep_limit
        self.max_walltime_limit = the_map.max_walltime_limit
        self.number_bombs = the_map.number_bombs
        self.size_area = the_map.size_area

        self.score_manager = ScoreManager(number_drones=self.number_drones,
                                          max_timestep_limit=self.max_timestep_limit,
                                          max_walltime_limit=self.max_walltime_limit,
                                          total_number_bombs=self.number_bombs,
                                          exploration_threshold_for_time_bonus=(
                                              self.eval_plan.exploration_threshold_for_time_bonus
                                          ))

        num_round_str = str(num_round)
        if self.video_capture_enabled:
            try:
                os.makedirs(self.result_path + "/videos/", exist_ok=True)
            except FileExistsError as error:
                print(error)
            filename_video_capture = (f"{self.result_path}/videos/"
                                      f"team{self.team_info.team_number_str_padded}_"
                                      f"{eval_config.map_name}_"
                                      f"{eval_config.zones_name_for_filename}_"
                                      f"rd{num_round_str}"
                                      f".avi")
        else:
            filename_video_capture = None

        if self.state_recording_enabled:
            try:
                os.makedirs(self.result_path + "/recordings/", exist_ok=True)
            except FileExistsError as error:
                print(error)
            filename_state_recording = (f"{self.result_path}/recordings/"
                                        f"team{self.team_info.team_number_str_padded}_"
                                        f"{eval_config.map_name}_"
                                        f"{eval_config.zones_name_for_filename}_"
                                        f"rd{num_round_str}"
                                        f".npz")
        else:
            filename_state_recording = None

        my_gui = GuiSR(the_map=the_map,
                       draw_interactive=False,
                       filename_video_capture=filename_video_capture,
                       filename_state_recording=filename_state_recording,
                       manual_bomb_edit_enabled=(
                           self.manual_bomb_edit and round_team_mode == TeamMode.RESCUE
                       ),
                       manual_bombs_in_path=self.manual_bombs_in,
                       manual_bombs_out_path=self.manual_bombs_out,
                       headless=headless,
                       competition_mode=competition_mode)

        window_title = (f"Team: {self.team_info.team_number_str}   -   "
                        f"Mode: {round_team_mode.value}   -   "
                        f"Map: {type(the_map).__name__}   -   "
                        f"Round: {num_round_str}")
        my_gui.set_caption(window_title)

        my_gui.set_state_recording_metadata({
            "team_mode": round_team_mode.value,
            "number_drones": self.number_drones,
        })

        the_map.explored_map.reset()

        # Clear possible stale red-team submissions in case the simulator reuses drone
        # instances across rounds.
        for d in the_map.drones:
            if hasattr(d, "_exploration_map_submission"):
                d._exploration_map_submission = None

        has_crashed = False
        error_msg = ""

        original_stdout = sys.stdout
        if hide_solution_output:
            sys.stdout = open(os.devnull, 'w')

        try:
            # this function below is a blocking function until the round is finished
            my_gui.run()
        except Exception:
            error_msg = traceback.format_exc()
            my_gui.close()

            # Clean up resources even in case of crash
            if hasattr(the_map, 'playground') and the_map.playground:
                the_map.playground.cleanup()
                # ensure window is closed as well (safe no-op if already closed)
                try:
                    the_map.playground.close_window()
                except Exception:
                    pass

            has_crashed = True
        finally:
            if hide_solution_output:
                sys.stdout.close()
                sys.stdout = original_stdout

        if has_crashed:
            print(error_msg)

        # Refresh the rescue score denominator: manual bomb edit mode may
        # have changed the bomb set after the ScoreManager was built, so the
        # count captured before the run can be stale.
        if round_team_mode == TeamMode.RESCUE:
            self.number_bombs = self._sync_rescue_bomb_total(
                my_gui, self.score_manager
            )

        score_exploration = the_map.explored_map.score() * 100.0
        score_health_returned = the_map.compute_score_health_returned() * 100

        last_image_explo_lines = the_map.explored_map.get_pretty_map_explo_lines()
        last_image_explo_zones = the_map.explored_map.get_pretty_map_explo_zones()
        self.data_saver.save_images(my_gui.last_image,
                                    last_image_explo_lines,
                                    last_image_explo_zones,
                                    eval_config.map_name,
                                    eval_config.zones_name_for_filename,
                                    num_round)

        # Capture the walls-only grid from the playground's real wall entities
        # BEFORE cleanup() wipes its elements. The truth grid, bombs and the
        # submitted exploration map live on the map / explored-map and survive
        # cleanup. (Place mode only; rescue rounds never use it.)
        walls_grid = None
        boundary_grid = None
        if round_team_mode == TeamMode.PLACE:
            walls_grid, boundary_grid = walls_and_boundary_grid(
                the_map.playground,
                truth_grid=the_map.explored_map.truth_obstacle_grid(),
            )

        # Exploration-map scoring and its diff image compare the submission
        # against the same walls-only truth as the wall score: only real wall
        # entities count as "walls". Non-wall fills -- the disposal center,
        # return area and disabler zones -- carry other collision types, so
        # drawing them earns no credit and they never appear on the truth
        # panel. Falls back to the full obstacle render when the walls grid
        # is unavailable (no wall entities found). Place mode only.
        exploration_truth = (
            walls_grid
            if walls_grid is not None
            else the_map.explored_map.truth_obstacle_grid()
        )

        # Clean up resources after the round to prevent memory leaks
        if hasattr(the_map, 'playground') and the_map.playground:
            the_map.playground.cleanup()
            # explicitly close the GUI window associated with the playground
            try:
                the_map.playground.close_window()
            except Exception:
                pass

        placed_bombs_path = None
        remaining_inventory = 0
        place_phase1 = None
        if round_team_mode == TeamMode.PLACE:
            remaining_inventory = sum(
                d.carried_bombs_count() for d in the_map.drones
            )
            placed_bombs_path = placed_bombs_output_path(
                self.result_path,
                self.team_info,
                eval_config,
                num_round,
            )
            if placed_bombs_path:
                save_placed_bombs(placed_bombs_path, the_map, eval_config)
                self._last_placed_bombs_file = placed_bombs_path
                print(f"\t\tbombs file written: {placed_bombs_path}")

            submitted_grid = None
            for d in the_map.drones:
                submitted_grid = getattr(d, "_exploration_map_submission", None)
                if submitted_grid is not None:
                    break

            if submitted_grid is not None:
                exploration_path = exploration_map_output_path(
                    self.result_path,
                    self.team_info,
                    eval_config,
                    num_round,
                )
                if exploration_path:
                    save_exploration_map(
                        exploration_path,
                        submitted_grid,
                        elapsed_walltime=my_gui.elapsed_walltime,
                    )
                    print(f"\t\texploration map file written: {exploration_path}")
            else:
                print("\t\tno exploration map submitted "
                      "(submit_exploration_map() was never called); "
                      "the comparison image is rendered with an empty submission")

            # The exploration-map comparison image is always produced: without a
            # submission it degrades to an empty prediction, so the round's
            # output directory still shows exactly which truth pixels the team
            # failed to map (and what that cost).
            diff_image_path = exploration_comparison_image_output_path(
                self.result_path,
                self.team_info,
                eval_config,
                num_round,
            )
            if diff_image_path:
                try:
                    save_exploration_comparison_image(
                        diff_image_path,
                        exploration_truth,
                        submitted_grid
                        if submitted_grid is not None
                        else np.zeros_like(exploration_truth),
                        scorer=scorer_for_walltime(self.max_walltime_limit),
                        elapsed_walltime=my_gui.elapsed_walltime,
                    )
                    print("\t\texploration diff image written: "
                          f"{diff_image_path}")
                except Exception as e:
                    print("\t\texploration diff image NOT generated: "
                          f"{e}")

            expected_bombs = (
                the_map.number_drones * eval_config.initial_bombs_per_drone
            )
            bomb_positions = [
                tuple(float(c) for c in bomb.true_position())
                for bomb in (getattr(the_map, "_bombs", None) or [])
            ]
            # Wall scoring/visualization use a walls-only grid built from the
            # map's real wall entities (captured before the playground cleanup
            # above), so the disposal center / return area / disabler zones are
            # not counted as destructible "walls". The boundary grid marks
            # only the map's boundary-frame walls. Exploration-map scoring
            # compares against the same walls-only truth (``exploration_truth``).
            place_phase1 = self.place_score_manager.compute_phase1_score(
                exploration_truth,
                submitted_grid,
                elapsed_walltime=my_gui.elapsed_walltime,
                max_walltime_limit=self.max_walltime_limit,
                score_exploration_trajectory=score_exploration,
                placed_count=the_map.number_bombs,
                expected_count=expected_bombs,
                has_crashed=has_crashed,
                bomb_positions=bomb_positions,
                elapsed_timestep=my_gui.elapsed_timestep,
                max_timestep_limit=self.max_timestep_limit,
                wall_grid=walls_grid,
                boundary_grid=boundary_grid,
            )

            # Wall-destruction visualization: one PNG per round showing how
            # much of the map's walls the bombs destroy (same geometry as the
            # wall score). Always produced -- a crashed round simply has no
            # bombs placed yet, so every wall renders as intact and the score
            # legend reads 0.
            wall_image_path = wall_destruction_image_output_path(
                self.result_path,
                self.team_info,
                eval_config,
                num_round,
            )
            if wall_image_path:
                try:
                    save_wall_destruction_image(
                        wall_image_path,
                        walls_grid
                        if walls_grid is not None
                        else the_map.explored_map.truth_obstacle_grid(),
                        bomb_positions,
                        boundary_mask=boundary_grid,
                        blast_radius=(
                            self.place_score_manager.blast_radius_px(
                                (walls_grid if walls_grid is not None else
                                 the_map.explored_map.truth_obstacle_grid()).shape
                            )
                        ),
                        boundary_wall_weight=(
                            self.place_score_manager.config.boundary_wall_weight
                        ),
                        wall_breakdown=place_phase1.wall_breakdown,
                        score_wall_percent=place_phase1.score_wall,
                    )
                    print("\t\twall destruction image written: "
                          f"{wall_image_path}")
                except Exception as e:
                    print("\t\twall destruction image NOT generated: "
                          f"{e}")

        return (my_gui.percent_drones_destroyed,
                my_gui.mean_drones_health,
                my_gui.elapsed_timestep,
                my_gui.full_disposal_timestep,
                score_exploration,
                my_gui.disposed_number,
                score_health_returned,
                my_gui.elapsed_walltime,
                my_gui.is_max_walltime_limit_reached,
                has_crashed,
                round_team_mode,
                the_map.number_bombs,
                placed_bombs_path,
                remaining_inventory,
                place_phase1)

    def go(
        self,
        stop_at_first_crash: bool = False,
        hide_solution_output: bool = False,
        headless: bool = False,
        competition_mode: bool = False,
    ) -> bool:
        """
        Runs the simulation for all evaluation configurations and calculates scores.

        Args:
            stop_at_first_crash (bool): Stop at the first crash if True.
            hide_solution_output (bool): Hide solution output if True.
            headless (bool): Run in headless mode if True.
            competition_mode (bool): If True, disable ground-truth APIs inside control().

        Returns:
            bool: True if all rounds completed successfully, False if any crashed.
        """
        ok = True
        # Red-team rounds, for the competition's best-round aggregation (the
        # bomb-difficulty term stays pending locally, see the summary below).
        place_round_records = []

        print(f"--------------------------------------------------------------------------------------------")

        for eval_config in self.eval_plan.list_eval_config:
            gc.collect()
            print("")

            if (eval_config.zones_config
                    and not isinstance(eval_config.zones_config[0], ZoneType)):
                raise ValueError(
                    "Invalid eval_config.zones_config. "
                    "It should be a tuple of ZoneType values."
                )

            print(f"--------------------------------------------------------------------------------------------")
            for num_round in range(eval_config.nb_rounds):
                print(f"--------------------------------------------------------------------------------------------")
                round_team_mode = self._resolve_team_mode(eval_config)
                print(
                    f"* Map: {eval_config.map_name}, mode: {round_team_mode.value}, "
                    f"special zones: {eval_config.zones_name_casual}, "
                    f"round: {num_round + 1}/{eval_config.nb_rounds}"
                )
                gc.collect()
                result = self.one_round(
                    eval_config,
                    num_round + 1,
                    hide_solution_output,
                    headless,
                    competition_mode,
                )
                if result is None:
                    return False
                (percent_drones_destroyed, mean_drones_health, elapsed_timestep,
                 full_disposal_timestep, score_exploration, disposed_number,
                 score_health_returned, elapsed_walltime,
                 is_max_walltime_limit_reached, has_crashed,
                 result_team_mode, placed_bomb_count, placed_bombs_path,
                 remaining_inventory, place_phase1) = result

                mean_drones_health_percent = mean_drones_health / DRONE_INITIAL_HEALTH * 100.

                if result_team_mode == TeamMode.PLACE:
                    round_score = (
                        place_phase1.round_score if place_phase1 is not None else 0.0
                    )
                    percent_disposed = 0.0
                    score_timestep = 0.0
                    score_map = (
                        place_phase1.score_map if place_phase1 is not None else 0.0
                    )
                    score_wall = (
                        place_phase1.score_wall if place_phase1 is not None else 0.0
                    )
                    score_time = (
                        place_phase1.score_time if place_phase1 is not None else 0.0
                    )
                    place_cfg = self.place_score_manager.config
                    expected_bombs = (
                        self.number_drones * eval_config.initial_bombs_per_drone
                    )
                    # The bomb-difficulty term is pending locally, so the highest
                    # reachable partial score is the sum of the other weights.
                    max_partial = 100.0 * (
                        place_cfg.w_exploration + place_cfg.w_wall + place_cfg.w_time
                    )
                    print(
                        f"\t* Round n°{num_round + 1}/{eval_config.nb_rounds} (place mode): "
                        f"\n\t\tbombs placed: {placed_bomb_count}/{expected_bombs}, "
                        f"inventory remaining: {remaining_inventory}, "
                        f"map score: {score_map:.1f}%, "
                        f"wall score: {score_wall:.1f}%, "
                        f"time score: {score_time:.1f}%, "
                        f"bomb score: pending, "
                        f"partial round score: {round_score:.1f}% "
                        f"(max {max_partial:.1f}%), "
                        f"trajectory explor.: {score_exploration:.1f}%, "
                        f"walltime elapsed: {elapsed_walltime:.0f}s/{self.max_walltime_limit}s, "
                        f"elapse timestep: {elapsed_timestep}/{self.max_timestep_limit} steps."
                        f"\n\t\tscore breakdown (competition weights): "
                        f"map {score_map:.1f} x {place_cfg.w_exploration:.0%} = "
                        f"{place_cfg.w_exploration * score_map:.1f}, "
                        f"bomb pending x {place_cfg.w_bomb:.0%} = n/a, "
                        f"wall {score_wall:.1f} x {place_cfg.w_wall:.0%} = "
                        f"{place_cfg.w_wall * score_wall:.1f}, "
                        f"time {score_time:.1f} x {place_cfg.w_time:.0%} = "
                        f"{place_cfg.w_time * score_time:.1f}, "
                        f"total {round_score:.1f}/{max_partial:.1f}."
                        f"\n\t\tpercentage of drones destroyed: {percent_drones_destroyed:.1f} %, "
                        f"mean percentage of drones health : {mean_drones_health_percent:.1f} %."
                    )
                    if placed_bombs_path:
                        print(f"\t\tbombs file: {placed_bombs_path}")
                    place_round_records.append({
                        "map": eval_config.map_name,
                        "round": num_round + 1,
                        "score": round_score,
                        "max_partial": max_partial,
                        "bomb_weight": place_cfg.w_bomb,
                    })
                else:
                    result_score = self.score_manager.compute_score(
                        disposed_number,
                        score_exploration,
                        score_health_returned,
                        elapsed_timestep,
                        has_crashed,
                    )
                    (round_score, percent_disposed, score_timestep) = result_score
                    print(
                        f"\t* Round n°{num_round + 1}/{eval_config.nb_rounds}: "
                        f"\n\t\trescued nb: {int(disposed_number)}/{self.number_bombs} "
                        f"(disposal score: {percent_disposed:.1f}%), "
                        f"health return score: {score_health_returned:.1f}%, "
                        f"time score: {score_timestep:.1f}%, "
                        f"explor. score: {score_exploration:.1f}% "
                        f"(weight {self.score_manager.w_exploration:.0%}), "
                        f"walltime elapsed: {elapsed_walltime:.0f}s/{self.max_walltime_limit}s, "
                        f"elapse timestep: {elapsed_timestep}/{self.max_timestep_limit} steps, "
                        f"time to dispose all: {full_disposal_timestep} steps."
                        f"\n\t\tscore breakdown (competition weights): "
                        f"disposal {percent_disposed:.1f} x {self.score_manager.w_disposal:.0%} = "
                        f"{self.score_manager.w_disposal * percent_disposed:.1f}, "
                        f"health {score_health_returned:.1f} x "
                        f"{self.score_manager.w_score_health_returned:.0%} = "
                        f"{self.score_manager.w_score_health_returned * score_health_returned:.1f}, "
                        f"time {score_timestep:.1f} x {self.score_manager.w_time:.0%} = "
                        f"{self.score_manager.w_time * score_timestep:.1f}, "
                        f"total {round_score:.1f}/100."
                        f"\n\t\tpercentage of drones destroyed: {percent_drones_destroyed:.1f} %, "
                        f"mean percentage of drones health : {mean_drones_health_percent:.1f} %."
                        f"\n\t\tround score: {round_score:.1f}%, "
                        f"frequency: {elapsed_timestep / elapsed_walltime:.2f} steps/s."
                    )

                if is_max_walltime_limit_reached:
                    print(f"\t\tThe max walltime limit of {self.max_walltime_limit}s is reached first.")

                if result_team_mode == TeamMode.RESCUE:
                    self.data_saver.save_one_round(
                        eval_config,
                        num_round + 1,
                        percent_drones_destroyed,
                        mean_drones_health_percent,
                        percent_disposed,
                        score_exploration,
                        score_health_returned,
                        elapsed_timestep,
                        elapsed_walltime,
                        full_disposal_timestep,
                        score_timestep,
                        has_crashed,
                        round_score,
                    )

                if has_crashed:
                    print(f"\t* WARNING, this program have crashed !")
                    ok = False
                    if stop_at_first_crash:
                        self.data_saver.generate_pdf_report()
                        return ok

        print(f"--------------------------------------------------------------------------------------------")
        if place_round_records:
            best = max(place_round_records, key=lambda rec: rec["score"])
            mean_place_score = (
                sum(rec["score"] for rec in place_round_records)
                / len(place_round_records)
            )
            print(
                "* Red team (place) aggregation — the competition takes the "
                "best round:\n"
                f"\trounds: {len(place_round_records)}, "
                f"best partial round score: {best['score']:.1f}% "
                f"({best['map']}, round {best['round']}, max possible "
                f"{best['max_partial']:.1f}%), "
                f"mean: {mean_place_score:.1f}%.\n"
                f"\tthe bomb-difficulty term ({best['bomb_weight']:.0%} of the "
                "final score) needs reference blue-team runs against this bomb "
                "layout and cannot be produced by a local run, so the "
                "competition score for these rounds lies in "
                f"[{best['score']:.1f}, "
                f"{best['score'] + 100.0 * best['bomb_weight']:.1f}] "
                "under the same best-round rule."
            )
            print(f"--------------------------------------------------------------------------------------------")
        self.data_saver.generate_pdf_report()

        return ok


if __name__ == "__main__":
    gc.disable()
    parser = argparse.ArgumentParser(description="Launcher of a swarm-rescue simulator for the competition")
    parser.add_argument("--stop_at_first_crash", "-s", action="store_true", help="Stop the code at first crash")
    parser.add_argument("--hide_solution_output", "-o", action="store_true", help="Hide print output of the solution")
    parser.add_argument("--headless", "-H", action="store_true", help="Run evaluations without opening a display window (suitable for servers)")
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Disable ground-truth APIs (true_position, etc.) inside control()",
    )
    parser.add_argument("--config", "-c", type=str, help="Path to evaluation plan YAML configuration file")
    parser.add_argument(
        "--team",
        type=str,
        choices=["rescue", "place"],
        help="Team mode: rescue (blue, default) or place (red)",
    )
    parser.add_argument(
        "--bombs-file",
        type=str,
        help="JSON file with bomb positions for rescue mode (overrides map defaults)",
    )
    parser.add_argument(
        "--manual-bomb-edit",
        action="store_true",
        help="Enable manual bomb editing in rescue mode before simulation starts",
    )
    parser.add_argument(
        "--manual-bombs-in",
        type=str,
        help="Import bombs JSON when manual bomb edit mode starts",
    )
    parser.add_argument(
        "--manual-bombs-out",
        type=str,
        help="Default output path used by manual bomb export",
    )
    parser.add_argument(
        "--replay",
        type=str,
        metavar="NPZ_PATH",
        help="Replay a recorded state file (.npz) instead of running a simulation",
    )
    args = parser.parse_args()

    if args.replay:
        from swarm_rescue.simulation.gui_map.replay_view import run_replay

        npz_path = args.replay
        if not os.path.exists(npz_path):
            print(f"Error: recording file not found: {npz_path}")
            exit(1)

        import numpy as np
        data = np.load(npz_path, allow_pickle=True)
        metadata = data["metadata"].item()
        map_name = metadata.get("map_name")
        if not map_name:
            print("Error: recording file has no map_name in metadata")
            exit(1)
        map_class = globals().get(map_name)
        if not map_class:
            print(f"Error: unknown map type '{map_name}'. Ensure it is imported in launcher.py.")
            exit(1)

        if args.headless:
            print("Replay in headless mode is not supported. Use --headless only for simulation.")
            exit(1)

        run_replay(npz_path, map_class)
        exit(0)

    launcher = Launcher(
        config_path=args.config,
        team_mode=args.team,
        default_bombs_file=args.bombs_file,
        manual_bomb_edit=args.manual_bomb_edit,
        manual_bombs_in=args.manual_bombs_in,
        manual_bombs_out=args.manual_bombs_out,
    )
    success = launcher.go(
        stop_at_first_crash=args.stop_at_first_crash,
        hide_solution_output=args.hide_solution_output,
        headless=args.headless,
        competition_mode=args.competition,
    )
    if not success:
        exit(1)

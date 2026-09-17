import time
from typing import Optional, Tuple, List, Dict
import sys
import time

import arcade
import cv2

from swarm_rescue.simulation.drone.controller import CommandName, Command
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.ground_truth_guard import run_control
from swarm_rescue.simulation.gui_map.keyboard_controller import KeyboardController
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.gui_map.playground import AllSentMessagesDict
from swarm_rescue.simulation.gui_map.top_down_view import TopDownView
from swarm_rescue.simulation.reporting.bombs_io import (
    bombs_from_data,
    load_bombs_file,
    save_bombs_document,
)
from swarm_rescue.simulation.reporting.screen_recorder import ScreenRecorder
from swarm_rescue.simulation.reporting.state_recorder import StateRecorder
from swarm_rescue.simulation.utils.constants import (
    DRONE_INITIAL_HEALTH,
    ENABLE_WINDOW_AUTO_RESIZE,
    FRAME_RATE,
    VIDEO_RECORDING_SCALE,
)
from swarm_rescue.simulation.utils.fps_display import FpsDisplay
from swarm_rescue.simulation.utils.mouse_measure import MouseMeasure
from swarm_rescue.simulation.utils.visu_noises import VisuNoises
from swarm_rescue.simulation.utils.window_utils import auto_resize_window


class GuiSR(TopDownView):
    _HUD_HEIGHT = 96

    """
    The GuiSR class is a subclass of TopDownView and provides a graphical user
    interface for the simulation. It handles the rendering of the playground,
    drones, and other visual elements, as well as user input and interaction.
    """


    def _handle_window_auto_resize(self, the_map: MapAbstract, size: Optional[Tuple[int, int]],
                                  zoom: float, headless: bool) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Handle automatic window resizing for small screens.

        Args:
            the_map: The map object containing the playground
            size: Initial window size
            zoom: Initial zoom factor
            headless: Whether running in headless mode

        Returns:
            Tuple of (adjusted_size, adjusted_zoom)
        """
        # Auto-resize window if needed for small screens (before window creation)
        if not headless and ENABLE_WINDOW_AUTO_RESIZE:
            # If size is None, use the playground size or a default
            if size is None:
                size = the_map.playground.size if the_map.playground.size else (1200, 800)

            # Check if this is a problematic map size that causes rendering issues
            problematic_sizes = [(1660, 1122)]  # Map04 (1660x1122) and similar
            is_problematic = size in problematic_sizes

            if not is_problematic:
                # Apply auto-resize with zoom adjustment only for non-problematic maps
                adjusted_size, calculated_zoom = auto_resize_window(size)
                if adjusted_size != size:
                    size = adjusted_size
                    zoom = zoom * calculated_zoom # Apply the calculated zoom factor
            else:
                # For problematic maps, disable auto-resize completely
                # Let the user handle window size manually or use system defaults
                print(f"Auto-resize désactivé pour cette carte ({size[0]}x{size[1]})")
        elif not headless and not ENABLE_WINDOW_AUTO_RESIZE:
            # Auto-resize is globally disabled
            if size is None:
                size = the_map.playground.size if the_map.playground.size else (1200, 800)
            print("Auto-resize désactivé globalement via ENABLE_WINDOW_AUTO_RESIZE")

        return size, zoom

    def _install_headless_flip_skip(self) -> None:
        """Skip the window buffer swap in headless mode; keep OpenGL GC."""
        window = self._playground.window

        def _headless_flip() -> None:
            window.ctx.gc()

        window.flip = _headless_flip

    def __init__(
            self,
            the_map: MapAbstract,
            size: Optional[Tuple[int, int]] = None,
            center: Tuple[float, float] = (0, 0),
            zoom: float = 1,
            use_keyboard: bool = False,
            use_color_uid: bool = False,
            draw_transparent: bool = False,
            draw_interactive: bool = False,
            draw_zone: bool = True,
            draw_lidar_rays: bool = False,
            draw_semantic_rays: bool = False,
            draw_gps: bool = False,
            draw_com: bool = False,
            print_rewards: bool = False,
            print_messages: bool = False,
            use_mouse_measure: bool = False,
            enable_visu_noises: bool = False,
            filename_video_capture: str = None,
            filename_state_recording: str = None,
            manual_bomb_edit_enabled: bool = False,
            manual_bombs_in_path: Optional[str] = None,
            manual_bombs_out_path: Optional[str] = None,
            headless: bool = False,
            competition_mode: bool = False,
    ) -> None:
        """
        Initialize the GuiSR graphical user interface.

        Args:
            the_map (MapAbstract): The map object containing the playground and drones.
            size (Optional[Tuple[int, int]]): Size of the window.
            center (Tuple[float, float]): Center of the view.
            zoom (float): Zoom factor.
            use_keyboard (bool): Enable keyboard control for the first drone.
            use_color_uid (bool): Use color UID for sprites.
            draw_transparent (bool): Draw transparent sprites.
            draw_interactive (bool): Draw interactive sprites.
            draw_zone (bool): Draw zone sprites.
            draw_lidar_rays (bool): Draw lidar sensor rays.
            draw_semantic_rays (bool): Draw semantic sensor rays.
            draw_gps (bool): Draw GPS sensor visualization.
            draw_com (bool): Draw communication visualization.
            print_rewards (bool): Print rewards to console.
            print_messages (bool): Print messages to console.
            use_mouse_measure (bool): Enable mouse measurement tool.
            enable_visu_noises (bool): Enable visualization of sensor noises.
            filename_video_capture (str): Output filename for video capture.
            filename_state_recording (str): Output filename for state recording (.npz).
        """
        # Handle automatic window resizing
        size, zoom = self._handle_window_auto_resize(the_map, size, zoom, headless)
        self._manual_bomb_edit_enabled = manual_bomb_edit_enabled and not headless
        self._hud_height = self._HUD_HEIGHT if self._manual_bomb_edit_enabled else 0
        if size is None:
            size = the_map.playground.size if the_map.playground.size else (1200, 800)
        window_size = size
        if self._hud_height > 0:
            window_size = (size[0], size[1] + self._hud_height)

        super().__init__(
            the_map.playground,
            size,
            center,
            zoom,
            use_color_uid,
            draw_transparent,
            draw_interactive,
            draw_zone,
        )


        self._headless = headless
        self._window_size = window_size
        self._playground.window.set_size(*self._window_size)


        # image_icon = pyglet.resource.image("resources/drone_v2.png")
        # self._playground.window.set_icon(image_icon)
        # Ok for the first round, crash for the second round ! I dont know
        # why...

        self._playground.window.set_visible(not self._headless)
        self._playground.window.headless = self._headless
        if self._headless:
            self._install_headless_flip_skip()

        self._playground.window.on_update = self.on_update
        self._playground.window.on_key_press = self.on_key_press
        self._playground.window.on_key_release = self.on_key_release
        self._playground.window.on_mouse_motion = self.on_mouse_motion
        self._playground.window.on_mouse_press = self.on_mouse_press
        self._playground.window.on_mouse_release = self.on_mouse_release
        self._playground.window.set_update_rate(FRAME_RATE)
        # self._playground.window.set_location(4500, 0)

        self._the_map = the_map
        self._drones = self._the_map.drones
        self._competition_mode = competition_mode
        for drone in self._drones:
            drone.set_competition_mode(competition_mode)
        self._number_drones = self._the_map.number_drones

        self._max_walltime_limit = self._the_map.max_walltime_limit
        if self._max_walltime_limit is None:
            self._max_walltime_limit = 100000000

        self._max_timestep_limit = self._the_map.max_timestep_limit
        if self._max_timestep_limit is None:
            self._max_timestep_limit = 100000000

        self._drones_commands: Optional[Dict[DroneAbstract, Dict[CommandName, Command]]] = None
        if self._drones:
            self._drones_commands = {}

        self._messages = None
        self._print_rewards = print_rewards
        self._print_messages = print_messages

        self._use_keyboard = use_keyboard

        self._draw_lidar_rays = draw_lidar_rays
        self._draw_semantic_rays = draw_semantic_rays
        self._draw_gps = draw_gps
        self._draw_com = draw_com
        self._use_mouse_measure = use_mouse_measure
        self._enable_visu_noises = enable_visu_noises

        # 'number_bombs' is the number of bombs that should
        # be retrieved by the drones.
        self._percent_drones_destroyed = 0.0
        self._mean_drones_health = 0.0

        self._total_number_bombs = (
            self._the_map.number_bombs)
        self._disposed_number = 0
        self._full_disposal_timestep = 0
        self._elapsed_timestep = 0
        self._start_timestamp = time.time()
        self._is_max_walltime_limit_reached = False
        self._elapsed_walltime = 0.001

        self._last_image = None
        self._terminate = False
        self._manual_bombs_in_path = manual_bombs_in_path
        self._manual_bombs_out_path = manual_bombs_out_path
        self._editing_locked = self._manual_bomb_edit_enabled
        self._manual_remove_distance = 24.0

        if self._manual_bomb_edit_enabled:
            print("Manual bomb edit mode enabled (rescue only).")
            print("Left click: add bomb | Right click: remove nearest bomb")
            print("I: import bombs | J: export bombs | K: clear bombs | Enter: start simulation")
            if self._manual_bombs_in_path:
                self._import_manual_bombs(self._manual_bombs_in_path)

        self.fps_display = FpsDisplay(period_display=2)
        self._keyboardController = KeyboardController()
        self._mouse_measure = MouseMeasure(playground_size=the_map.playground.size)
        self._visu_noises = VisuNoises(playground_size=the_map.playground.size,
                                       drones=self._drones)

        self._recording_view = None
        if filename_video_capture is not None:
            scale = VIDEO_RECORDING_SCALE
            rec_w = max(1, int(self._size[0] * scale))
            rec_h = max(1, int(self._size[1] * scale))
            # Match the main view's world extent: smaller framebuffer + proportionally
            # smaller zoom keeps the full map in frame (not a cropped corner).
            self._recording_view = TopDownView(
                the_map.playground,
                size=(rec_w, rec_h),
                center=self._center,
                zoom=self._zoom * scale,
                draw_transparent=self._draw_transparent,
                draw_interactive=self._draw_interactive,
                draw_zone=self._draw_zone,
            )

        rec_w = self._recording_view.width if self._recording_view else self._size[0]
        rec_h = self._recording_view.height if self._recording_view else self._size[1]
        self.recorder = ScreenRecorder(rec_w, rec_h, fps=30,
                                       out_file=filename_video_capture)

        self._state_recorder = (
            StateRecorder() if filename_state_recording is not None else None
        )
        self._filename_state_recording = filename_state_recording
        self._state_recording_metadata = {}

        if self._headless and filename_video_capture is not None:
            self._playground.window.on_draw = self._on_draw_headless_skip
        else:
            self._playground.window.on_draw = self.on_draw

    def _on_draw_headless_skip(self) -> None:
        """No-op draw hook: headless recording uses the recording FBO only."""
        return

    def close(self) -> None:
        """
        Close the simulation window.
        """
        self._playground.close_window()

    def set_caption(self, window_title: str) -> None:
        """
        Set the window caption/title.

        Args:
            window_title (str): The title to set.
        """
        self._playground.window.set_caption(window_title)

    def run(self) -> None:
        """
        Start the simulation event loop.
        """
        self._playground.window.run()


    def on_draw(self) -> None:
        """
        Render the current frame to the window.
        """
        # Clear the window
        self._playground.window.clear()
        # Binding the framebuffer object to the window
        # Is it necessary ? It seems to work without it.
        self._fbo.use()

        # Draw the playground and all the entities in it
        # Copier le contenu de draw() ici ?
        self.draw()

    def on_update(self, delta_time: float) -> None:
        """
        Update the simulation state and draw the playground and entities.

        Args:
            delta_time (float): Time since last update.
        """
        if self._editing_locked:
            return

        self._elapsed_timestep += 1

        if self._elapsed_timestep < 2:
            self._playground.step(all_commands=self._drones_commands,
                                  all_messages=self._messages)
            # self._the_map.explored_map.update(self._drones)
            # self._the_map.explored_map._process_positions()
            # self._the_map.explored_map.display()
            return

        self._the_map.explored_map.update_drones(self._drones)
        # self._the_map.explored_map._process_positions()
        # self._the_map.explored_map.display()

        # COMPUTE ALL THE MESSAGES
        self._messages = self.collect_all_messages(self._drones)

        # COMPUTE COMMANDS
        for i in range(self._number_drones):
            self._drones[i].elapsed_walltime = self._elapsed_walltime
            self._drones[i].elapsed_timestep = self._elapsed_timestep
            command = run_control(self._drones[i])
            if self._use_keyboard and i == 0:
                command = self._keyboardController.control()

            self._drones_commands[self._drones[i]] = command

        # Early termination for red-team map submission.
        # We end the round before the next physics step is applied.
        exploration_submitted = any(
            getattr(drone, "_exploration_map_submission", None) is not None
            for drone in self._drones
        )
        if exploration_submitted:
            self._terminate = True
        else:
            if self._drones:
                self._drones[0].display()

            self._playground.step(all_commands=self._drones_commands,
                                  all_messages=self._messages)

            # self._playground.debug_draw()

            self._visu_noises.update(enable=self._enable_visu_noises)
            # self._the_map.explored_map.display()

            # REWARDS
            new_reward = 0
            for i in range(self._number_drones):
                new_reward += self._drones[i].reward

            self._update_disposal_progress(new_reward)

        last_timestamp = time.time()
        # last_elapsed_walltime = self._elapsed_walltime
        self._elapsed_walltime = (last_timestamp - self._start_timestamp)
        # delta = self._elapsed_walltime - last_elapsed_walltime
        # if delta > 0.5:
        #     print("self._elapsed_walltime = {:.1f}, delta={:.1f},
        #     freq={:.1f}, freq moy={:.1f}".format(
        #         self._elapsed_walltime,
        #         delta,
        #         1 / (delta + 0.0001),
        #         self._elapsed_timestep / (self._elapsed_walltime + 0.00001)))
        if self._elapsed_walltime > self._max_walltime_limit:
            self._elapsed_walltime = self._max_walltime_limit
            self._is_max_walltime_limit_reached = True
            self._terminate = True

        if self._elapsed_timestep > self._max_timestep_limit:
            self._elapsed_timestep = self._max_timestep_limit
            self._terminate = True

        if self._print_rewards:
            for agent in self._playground.agents:
                if agent.reward != 0:
                    print(agent.reward)

        if self._print_messages:
            for drone in self._playground.agents:
                for comm in drone.communicators:
                    for _, msg in comm.received_messages:
                        print(f"Drone {drone.name} received message {msg}")

        self._messages = {}

        # Capture state recording (before video capture)
        if self._state_recorder is not None:
            self._state_recorder.capture_state(
                self._playground, self._elapsed_walltime
            )

        # Capture the frame
        # Au bon endroit ? Il faudrait le mettre avant le draw() ?
        capture_view = self._recording_view if self._recording_view is not None else self
        self.recorder.capture_frame(capture_view)

        self.fps_display.update(display=False)

        # print("can_grasp: {}, entities: {}".format(self._drone.grasper.can_grasp,
        #                                            self._drone.grasper.grasped_bombs))

        if self._terminate:
            self.compute_health_stats()
            self.recorder.end_recording()
            if self._state_recorder is not None:
                metadata = {
                    "map_name": type(self._the_map).__name__,
                    "map_size": self._the_map.playground.size,
                    "frame_rate": 30.0,
                    **self._state_recording_metadata,
                }
                self._state_recorder.save(
                    self._filename_state_recording, metadata
                )
            self._last_image = self.get_playground_image()
            arcade.close_window()

    def _update_disposal_progress(self, new_reward: int) -> None:
        """Update disposal counters and stop once all bombs are disposed."""
        if new_reward != 0:
            self._disposed_number += new_reward

        if (
            self._total_number_bombs > 0
            and self._disposed_number == self._total_number_bombs
        ):
            if self._full_disposal_timestep == 0:
                self._full_disposal_timestep = self._elapsed_timestep
            self._terminate = True

    def get_playground_image(self) -> cv2.typing.MatLike:
        """
        Get the image of the playground in the framebuffer.

        Returns:
            Any: The image as a numpy array.
        """
        self.update_and_draw_in_framebuffer()
        # The image should be flip and the color channel permuted
        image = cv2.flip(self.get_np_img(), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def draw(self, force: bool = False) -> None:
        """
        Draw the playground and all the entities in it in the window.

        Args:
            force (bool): If True, force update of all sprites.
        """
        arcade.start_render()
        self.update_sprites_position(force)

        self._playground.window.use()
        self._playground.window.clear(self._background)
        self._ctx.projection_2d = 0, self.width, -self._hud_height, self.height

        for drone in self._playground.agents:
            drone.draw_bottom_layer()

        if self._draw_lidar_rays:
            for drone in self._playground.agents:
                drone.lidar().draw()

        if self._draw_semantic_rays:
            for drone in self._playground.agents:
                drone.semantic().draw()

        if self._draw_gps:
            for drone in self._playground.agents:
                drone.draw_gps()

        if self._draw_com:
            for drone in self._playground.agents:
                drone.draw_com()

        self._mouse_measure.draw(enable=self._use_mouse_measure)
        self._visu_noises.draw(enable=self._enable_visu_noises)

        if self._draw_transparent:
            self._transparent_sprites.draw(pixelated=True)

        if self._draw_interactive:
            self._interactive_sprites.draw(pixelated=True)

        if self._draw_zone:
            self._zone_sprites.draw(pixelated=True)

        self._visible_sprites.draw(pixelated=True)

        for drone in self._playground.agents:
            drone.draw_top_layer()

        self._draw_hud()

        # display a circle representing semantic detection radius
        # width, height = self._size
        # drone = self._playground.agents[0]
        # x, y = drone.true_position()
        # x = x + width / 2
        # y = y + height / 2
        #
        # r = drone.true_angle()
        # for i in range(34):
        #     arcade.draw_line(x, y, x + 200 * cos(r + i * (2 * pi / 34)),
        #     y + 200 * sin(r + i * (2 * pi / 34)), (60, 120, 80))
        # # endregion

    def on_key_press(self, key: int, modifiers: int) -> None:
        """
        Called whenever a key is pressed.

        Args:
            key (int): The key code pressed.
            modifiers (int): Modifier keys pressed.
        """
        self._keyboardController.on_key_press(key, modifiers)

        if self._manual_bomb_edit_enabled:
            if key == arcade.key.ENTER:
                self._editing_locked = False
                self._total_number_bombs = self._the_map.number_bombs
                print(f"Manual edit confirmed. Simulation starts with {self._total_number_bombs} bombs.")
                return
            if key == arcade.key.I:
                in_path = self._manual_bombs_in_path or self._prompt_manual_bombs_import_path()
                if in_path:
                    self._import_manual_bombs(in_path)
                return
            if key == arcade.key.J:
                out_path = self._manual_bombs_out_path
                if not out_path:
                    out_path = f"manual_bombs_{type(self._the_map).__name__}.json"
                self._export_manual_bombs(out_path)
                return
            if key == arcade.key.K:
                self._the_map.clear_bombs()
                self._total_number_bombs = self._the_map.number_bombs
                print("Manual bombs cleared.")
                return

        if key == arcade.key.C:
            self._draw_com = not self._draw_com

        if key == arcade.key.P:
            self._draw_gps = not self._draw_gps

        if key == arcade.key.L:
            self._draw_lidar_rays = not self._draw_lidar_rays

        if self._drones:

            if key == arcade.key.M:
                self._messages = {
                    self._drones[0]: {
                        self._drones[0].communicator: (
                            None,
                            f"Currently at timestep {self._playground.timestep}",
                        )
                    }
                }
                print(f"Drone {self._drones[0].name} sends message")

        if key == arcade.key.Q:
            self._terminate = True

        if key == arcade.key.E:
            print("Touche E pressée - Arrêt complet du programme...")
            arcade.close_window()
            sys.exit(0)

        if key == arcade.key.R:
            self._playground.reset()
            self._visu_noises.reset()

        if key == arcade.key.S:
            self._draw_semantic_rays = not self._draw_semantic_rays

    def on_key_release(self, key: int, modifiers: int) -> None:
        """
        Called whenever a key is released.

        Args:
            key (int): The key code released.
            modifiers (int): Modifier keys pressed.
        """
        self._keyboardController.on_key_release(key, modifiers)

    # Creating function to check the position of the mouse
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        """
        Called whenever the mouse is moved.

        Args:
            x (int): X position.
            y (int): Y position.
            dx (int): Change in X.
            dy (int): Change in Y.
        """
        self._mouse_measure.on_mouse_motion(x, y, dx, dy)

    # Creating function to check the mouse clicks
    def on_mouse_press(self, x: int, y: int, button: int, _: int) -> None:
        """
        Called whenever a mouse button is pressed.

        Args:
            x (int): X position.
            y (int): Y position.
            button (int): Mouse button.
            _ (int): Modifier keys pressed.
        """
        if self._editing_locked:
            if y < self._hud_height:
                return
            world_x, world_y = self._screen_to_world(x, y)
            if button == arcade.MOUSE_BUTTON_LEFT:
                self._the_map.spawn_bombs([{"x": world_x, "y": world_y, "theta": 0.0}])
                self._total_number_bombs = self._the_map.number_bombs
                print(f"Bomb added at ({world_x:.1f}, {world_y:.1f}). Total={self._total_number_bombs}")
                return
            if button == arcade.MOUSE_BUTTON_RIGHT:
                removed = self._the_map.remove_nearest_bomb(
                    world_x,
                    world_y,
                    max_dist=self._manual_remove_distance,
                )
                self._total_number_bombs = self._the_map.number_bombs
                if removed:
                    print(f"Bomb removed near ({world_x:.1f}, {world_y:.1f}). Total={self._total_number_bombs}")
                else:
                    print("No bomb found nearby to remove.")
                return

        self._mouse_measure.on_mouse_press(x, y, button, enable=self._use_mouse_measure)

    def on_mouse_release(self, x: int, y: int, button: int, _: int) -> None:
        """
        Called whenever a mouse button is released.

        Args:
            x (int): X position.
            y (int): Y position.
            button (int): Mouse button.
            _ (int): Modifier keys pressed.
        """
        self._mouse_measure.on_mouse_release(x, y, button,
                                             enable=self._use_mouse_measure)

    def collect_all_messages(self, drones: List[DroneAbstract]) -> AllSentMessagesDict:
        """
        Collect messages from all drones.

        Args:
            drones (List[DroneAbstract]): List of drones.

        Returns:
            AllSentMessagesDict: Dictionary of messages for each drone.
        """
        messages: AllSentMessagesDict = {}
        for i in range(self._number_drones):
            msg_data = drones[i].define_message_for_all()
            messages[drones[i]] = {drones[i].communicator: (None, msg_data)}
        return messages

    def _screen_to_world(self, x: int, y: int) -> Tuple[float, float]:
        world_x = (x - self.width / 2) / self.zoom + self.center[0]
        world_y = ((y - self._hud_height) - self.height / 2) / self.zoom + self.center[1]
        return world_x, world_y

    def _draw_hud(self) -> None:
        if self._hud_height <= 0:
            return

        self._ctx.projection_2d = 0, self._window_size[0], 0, self._window_size[1]
        hud_bg = (20, 20, 20, 240)
        hud_border = (90, 90, 90, 255)
        hud_text = (230, 230, 230, 255)
        arcade.draw_lrtb_rectangle_filled(0, self.width, self._hud_height, 0, hud_bg)
        arcade.draw_lrtb_rectangle_outline(0, self.width, self._hud_height, 0, hud_border, border_width=2)

        state = "EDITING" if self._editing_locked else "RUNNING"
        line1 = (
            f"Manual Bomb Mode | State: {state} | Bombs: {self._the_map.number_bombs} | "
            f"Round step: {self._elapsed_timestep}"
        )
        line2 = (
            "Keys: LeftClick add, RightClick remove, I import, J export, "
            "K clear, Enter start"
        )
        arcade.draw_text(line1, 12, self._hud_height - 30, hud_text, 14, bold=True)
        arcade.draw_text(line2, 12, 14, hud_text, 12)

    def _expected_zones_names(self) -> List[str]:
        zones = self._the_map.zones_config
        if not zones:
            return []
        return [zone.name for zone in zones]

    def _import_manual_bombs(self, path: str) -> None:
        try:
            data = load_bombs_file(path)
            file_map = data.get("map_name")
            expected_map = type(self._the_map).__name__
            if file_map and file_map != expected_map:
                print(f"Warning: bombs file map_name {file_map!r} != {expected_map!r}")
            file_zones = data.get("zones_config")
            expected_zones = self._expected_zones_names()
            if file_zones is not None and list(file_zones) != expected_zones:
                print(f"Warning: bombs file zones_config {file_zones!r} != {expected_zones!r}")
            self._the_map.clear_bombs()
            self._the_map.spawn_bombs(bombs_from_data(data))
            self._total_number_bombs = self._the_map.number_bombs
            print(f"Imported {self._total_number_bombs} bombs from {path}")
        except Exception as exc:
            print(f"Manual bombs import failed: {exc}")

    @staticmethod
    def _prompt_manual_bombs_import_path() -> Optional[str]:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Import bombs JSON",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            root.destroy()
            if not path:
                return None
            return path
        except Exception as exc:
            print(f"Manual bombs import dialog unavailable: {exc}")
            return None

    def _export_manual_bombs(self, path: str) -> None:
        try:
            document = {
                "map_name": type(self._the_map).__name__,
                "zones_config": self._expected_zones_names(),
                "bombs": [
                    {
                        "x": float(b.true_position()[0]),
                        "y": float(b.true_position()[1]),
                        "theta": float(b.true_angle()),
                    }
                    for b in self._the_map.bombs
                ],
            }
            save_bombs_document(path, document)
            print(f"Exported {len(document['bombs'])} bombs to {path}")
        except Exception as exc:
            print(f"Manual bombs export failed: {exc}")

    def compute_health_stats(self) -> None:
        """
        Compute statistics about drone health and destruction.
        """
        sum_health = 0
        for drone in self._playground.agents:
            sum_health += drone.drone_health

        nb_destroyed = self._number_drones - len(self._playground.agents)

        if self._number_drones > 0:
            self._mean_drones_health = sum_health / self._number_drones
            self._percent_drones_destroyed = nb_destroyed / self._number_drones * 100
        else:
            self._mean_drones_health = DRONE_INITIAL_HEALTH
            self._percent_drones_destroyed = 0.0

    @property
    def last_image(self):
        """
        Returns the last captured image of the playground.

        Returns:
            Any: The last image.
        """
        return self._last_image

    @property
    def percent_drones_destroyed(self) -> float:
        """
        Returns the percentage of drones destroyed.

        Returns:
            float: Percentage of destroyed drones.
        """
        return self._percent_drones_destroyed

    @property
    def mean_drones_health(self) -> float:
        """
        Returns the mean health of all drones.

        Returns:
            float: Mean drone health.
        """
        return self._mean_drones_health

    @property
    def elapsed_timestep(self) -> int:
        """
        Returns the number of elapsed timesteps.

        Returns:
            int: Elapsed timesteps.
        """
        return self._elapsed_timestep

    @property
    def elapsed_walltime(self) -> float:
        """
        Returns the elapsed wall time in seconds.

        Returns:
            float: Elapsed wall time.
        """
        return self._elapsed_walltime

    @property
    def total_number_bombs(self) -> int:
        """
        Returns the total number of bombs this round counts for disposal.

        This is the authoritative count: it drives the round's termination
        test and is refreshed when manual bomb editing confirms (Enter) or
        edits the bomb set, so it stays correct even when bombs are added or
        imported after the GUI was created.

        Returns:
            int: Total number of bombs.
        """
        return self._total_number_bombs

    @property
    def disposed_number(self) -> int:
        """
        Returns the number of rescued bombs.

        Returns:
            int: Number rescued.
        """
        return self._disposed_number

    @property
    def full_disposal_timestep(self) -> int:
        """
        Returns the timestep at which all bombs were rescued.

        Returns:
            int: Full rescue timestep.
        """
        return self._full_disposal_timestep

    @property
    def is_max_walltime_limit_reached(self) -> bool:
        """
        Returns whether the maximum walltime limit has been reached.

        Returns:
            bool: True if reached, False otherwise.
        """
        return self._is_max_walltime_limit_reached

    def set_state_recording_metadata(self, metadata: dict) -> None:
        self._state_recording_metadata = metadata

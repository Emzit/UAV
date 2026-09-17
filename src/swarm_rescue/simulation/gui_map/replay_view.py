import arcade

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.gui_map.top_down_view import TopDownView
from swarm_rescue.simulation.reporting.state_replayer import StateReplayer
from swarm_rescue.simulation.utils.constants import FRAME_RATE


class _ReplayDrone(DroneAbstract):
    def control(self) -> dict:
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}

    def define_message_for_all(self):
        return ""


def run_replay(npz_path: str, map_class, map_kwargs_extra: dict = None):
    def _log(msg):
        print(msg, flush=True)

    _log(f"Loading replay: {npz_path}")
    replayer = StateReplayer(npz_path)
    metadata = replayer.metadata

    map_kwargs = {"drone_type": _ReplayDrone}
    if "number_drones" in metadata:
        map_kwargs["number_drones"] = int(metadata["number_drones"])
    if map_kwargs_extra:
        map_kwargs.update(map_kwargs_extra)

    _log("Creating replay environment...")
    the_map = map_class(**map_kwargs)
    playground = the_map.playground
    window = playground.window

    max_bombs = metadata.get("max_bombs", 0)
    if max_bombs > 0:
        the_map.spawn_bombs([
            {"x": 0.0, "y": 0.0, "theta": 0.0}
            for _ in range(max_bombs)
        ])

    view = TopDownView(
        playground,
        size=playground.size,
        center=(0, 0),
        zoom=1,
        draw_interactive=False,
    )

    window.set_size(*playground.size)
    window.set_visible(True)

    frame_idx = [0]
    paused = [True]
    speed = [1]

    # Apply initial frame to show correct positions when paused
    replayer.apply_to_playground(playground, 0)

    def on_update(delta_time):
        if paused[0]:
            return
        for _ in range(speed[0]):
            if frame_idx[0] >= replayer.n_frames:
                arcade.close_window()
                return
            replayer.apply_to_playground(playground, frame_idx[0])
            frame_idx[0] += 1

    def on_draw():
        arcade.start_render()
        view.update_sprites_position(force=True)
        window.use()
        window.clear(view._background)
        ctx = playground.ctx
        ctx.projection_2d = 0, view.width, 0, view.height

        if view._draw_transparent:
            view._transparent_sprites.draw(pixelated=True)
        if view._draw_interactive:
            view._interactive_sprites.draw(pixelated=True)
        if view._draw_zone:
            view._zone_sprites.draw(pixelated=True)
        view._visible_sprites.draw(pixelated=True)

    def on_key_press(key, modifiers):
        if key == arcade.key.SPACE:
            paused[0] = not paused[0]
            state = "paused" if paused[0] else "playing"
            print(f"Replay {state} at frame {frame_idx[0]}")
        elif key == arcade.key.LEFT:
            frame_idx[0] = max(0, frame_idx[0] - 30)
            replayer.seek(frame_idx[0])
            print(f"Frame: {frame_idx[0]}/{replayer.n_frames}")
        elif key == arcade.key.RIGHT:
            frame_idx[0] = min(replayer.n_frames - 1, frame_idx[0] + 30)
            replayer.seek(frame_idx[0])
            print(f"Frame: {frame_idx[0]}/{replayer.n_frames}")
        elif key == arcade.key.UP:
            speed[0] = min(speed[0] + 1, 10)
            print(f"Speed: {speed[0]}x")
        elif key == arcade.key.DOWN:
            speed[0] = max(speed[0] - 1, 1)
            print(f"Speed: {speed[0]}x")
        elif key == arcade.key.HOME:
            frame_idx[0] = 0
            replayer.seek(0)
            print("Restart")
        elif key == arcade.key.END:
            frame_idx[0] = replayer.n_frames - 1
            replayer.seek(frame_idx[0])
            print(f"End: {frame_idx[0]}/{replayer.n_frames}")
        elif key == arcade.key.Q or key == arcade.key.ESCAPE:
            arcade.close_window()

    window.on_update = on_update
    window.on_draw = on_draw
    window.on_key_press = on_key_press
    window.set_update_rate(FRAME_RATE)

    name = metadata["map_name"]
    frames = replayer.n_frames
    window.set_caption(f"Replay: {name} ({frames} frames) - SPACE:play,"
                       f" LEFT/RIGHT:seek, UP/DOWN:speed, Q:quit")
    _log(f"Replay ready: {name}, {frames} frames, "
         f"{metadata.get('max_drones', 0)} drones, "
         f"{metadata.get('max_bombs', 0)} bombs")
    _log("SPACE: play/pause | LEFT/RIGHT: seek | UP/DOWN: speed | Q: quit")

    window.run()

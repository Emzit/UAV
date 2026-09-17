import numpy as np

from swarm_rescue.simulation.elements.bomb import Bomb

DRONE_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("angle", "f4"),
    ("active", "?"), ("health", "i4"),
])

BOMB_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("angle", "f4"),
    ("active", "?"),
])


class StateRecorder:
    """
    Records per-timestep entity states (drones, bombs) to a numpy .npz file.

    Usage:
        recorder = StateRecorder()
        # in simulation loop, after physics step:
        recorder.capture_state(playground, elapsed_walltime)
        # at round end:
        recorder.save(filepath, metadata)
    """

    def __init__(self):
        self._drone_frames: list[list[dict]] = []
        self._bomb_frames: list[list[dict]] = []
        self._timestamps: list[float] = []

    def capture_state(self, playground, walltime: float) -> None:
        drone_frame = []
        for agent in playground.agents:
            base = agent.base
            drone_frame.append({
                "x": float(base.position.x),
                "y": float(base.position.y),
                "angle": float(base.angle),
                "active": not agent.removed,
                "health": agent.drone_health,
            })
        self._drone_frames.append(drone_frame)

        bomb_frame = []
        for element in playground.elements:
            if isinstance(element, Bomb):
                bomb_frame.append({
                    "x": float(element.position.x),
                    "y": float(element.position.y),
                    "angle": float(element.angle),
                    "active": not element.removed,
                })
        self._bomb_frames.append(bomb_frame)
        self._timestamps.append(walltime)

    def save(self, filepath: str, metadata: dict) -> None:
        n_frames = len(self._drone_frames)
        max_drones = max((len(f) for f in self._drone_frames), default=0)
        max_bombs = max((len(f) for f in self._bomb_frames), default=0)

        drone_states = np.zeros((n_frames, max_drones), dtype=DRONE_DTYPE)
        for t, frame in enumerate(self._drone_frames):
            for i, d in enumerate(frame):
                drone_states[t, i] = (
                    d["x"], d["y"], d["angle"],
                    d["active"], d["health"],
                )

        bomb_states = np.zeros((n_frames, max_bombs), dtype=BOMB_DTYPE)
        for t, frame in enumerate(self._bomb_frames):
            for i, b in enumerate(frame):
                bomb_states[t, i] = (
                    b["x"], b["y"], b["angle"], b["active"],
                )

        timestamps = np.array(self._timestamps, dtype="f8")

        metadata["n_timesteps"] = n_frames
        metadata["max_drones"] = max_drones
        metadata["max_bombs"] = max_bombs

        print(f"\nState recording: {n_frames} frames, "
              f"{max_drones} drones, {max_bombs} bombs "
              f"-> {filepath}")
        np.savez_compressed(
            filepath,
            drone_states=drone_states,
            bomb_states=bomb_states,
            timestamps=timestamps,
            metadata=metadata,
        )

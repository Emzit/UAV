import numpy as np

from swarm_rescue.simulation.elements.bomb import Bomb

# Sentinel position for entities that should be invisible (far off-screen).
_OFF_SCREEN = (1e6, 1e6)


class StateReplayer:
    """
    Loads a recorded .npz file and provides frame-by-frame access
    to entity states for visual replay.

    Usage:
        replayer = StateReplayer("recording.npz")
        # create map/playground normally...
        # for each frame:
        replayer.apply_to_playground(playground, frame_idx)
        view.update_and_draw_in_framebuffer(force=True)
        replayer.advance()
    """

    def __init__(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.metadata = data["metadata"].item()
        self.drone_states = data["drone_states"]
        self.bomb_states = data["bomb_states"]
        self.timestamps = data["timestamps"]
        self.n_frames = len(self.drone_states)
        self.current_frame = 0

    def seek(self, frame_idx: int) -> None:
        self.current_frame = min(max(frame_idx, 0), self.n_frames - 1)

    def advance(self) -> bool:
        if self.current_frame < self.n_frames - 1:
            self.current_frame += 1
            return True
        return False

    def get_frame_state(self, frame_idx=None):
        idx = self.current_frame if frame_idx is None else frame_idx
        return {
            "drone_states": self.drone_states[idx],
            "bomb_states": self.bomb_states[idx],
            "timestamp": self.timestamps[idx],
        }

    def apply_to_playground(self, playground, frame_idx=None) -> None:
        idx = self.current_frame if frame_idx is None else frame_idx

        drone_states = self.drone_states[idx]
        agents = playground.agents
        for i, agent in enumerate(agents):
            if i < len(drone_states):
                ds = drone_states[i]
                if ds["active"]:
                    agent.base._pm_body.position = (
                        float(ds["x"]), float(ds["y"])
                    )
                    agent.base._pm_body.angle = float(ds["angle"])
                else:
                    agent.base._pm_body.position = _OFF_SCREEN

        bomb_states = self.bomb_states[idx]
        bomb_idx = 0
        for element in playground.elements:
            if isinstance(element, Bomb):
                if bomb_idx < len(bomb_states):
                    bs = bomb_states[bomb_idx]
                    if bs["active"]:
                        element._pm_body.position = (
                            float(bs["x"]), float(bs["y"])
                        )
                        element._pm_body.angle = float(bs["angle"])
                    else:
                        element._pm_body.position = _OFF_SCREEN
                    bomb_idx += 1

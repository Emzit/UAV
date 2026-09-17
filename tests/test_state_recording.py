import tempfile
import numpy as np

from swarm_rescue.simulation.reporting.state_recorder import (
    StateRecorder, DRONE_DTYPE, BOMB_DTYPE,
)
from swarm_rescue.simulation.reporting.state_replayer import StateReplayer


class MockBase:
    def __init__(self, x, y, angle):
        self.position = MockVec2d(x, y)
        self.angle = angle
        self.velocity = MockVec2d(0, 0)
        self.angular_velocity = 0.0


class MockVec2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MockAgent:
    def __init__(self, x, y, angle, health, removed=False):
        self.base = MockBase(x, y, angle)
        self.drone_health = health
        self.removed = removed


class MockBomb:
    def __init__(self, x, y, angle, removed=False):
        self.position = MockVec2d(x, y)
        self.angle = angle
        self.removed = removed


class MockPlayground:
    def __init__(self, agents, elements):
        self.agents = agents
        self.elements = elements


def test_state_recorder_empty():
    recorder = StateRecorder()
    pg = MockPlayground([], [])
    recorder.capture_state(pg, 0.0)
    recorder.capture_state(pg, 0.1)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        filepath = f.name

    try:
        recorder.save(filepath, {"test": True, "map_name": "TestMap"})
        data = np.load(filepath, allow_pickle=True)
        assert data["drone_states"].shape == (2, 0)
        assert data["bomb_states"].shape == (2, 0)
        assert data["metadata"].item()["n_timesteps"] == 2
    finally:
        import os
        os.unlink(filepath)


def test_state_recorder_basic():
    recorder = StateRecorder()
    agents = [
        MockAgent(1.0, 2.0, 0.5, 50),
        MockAgent(10.0, 20.0, 1.0, 30),
    ]
    pg = MockPlayground(agents, [])

    for t in range(3):
        recorder.capture_state(pg, t * 0.1)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        filepath = f.name

    try:
        recorder.save(filepath, {"map_name": "TestMap"})
        data = np.load(filepath, allow_pickle=True)
        ds = data["drone_states"]
        assert ds.shape == (3, 2)
        assert ds[0, 0]["x"] == 1.0
        assert ds[0, 0]["y"] == 2.0
        assert ds[0, 0]["angle"] == 0.5
        assert ds[0, 0]["active"] == True
        assert ds[0, 0]["health"] == 50
        assert ds[0, 1]["health"] == 30
    finally:
        import os
        os.unlink(filepath)


def test_state_recorder_variable_counts():
    recorder = StateRecorder()
    pg1 = MockPlayground(
        [MockAgent(0, 0, 0, 50)],
        [],
    )
    pg2 = MockPlayground(
        [MockAgent(0, 0, 0, 50), MockAgent(1, 1, 1, 40)],
        [],
    )
    recorder.capture_state(pg1, 0.0)
    recorder.capture_state(pg2, 0.1)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        filepath = f.name

    try:
        recorder.save(filepath, {"map_name": "TestMap"})
        data = np.load(filepath, allow_pickle=True)
        assert data["drone_states"].shape == (2, 2)
        assert data["metadata"].item()["max_drones"] == 2
        assert data["metadata"].item()["max_bombs"] == 0
    finally:
        import os
        os.unlink(filepath)


def test_state_replayer_roundtrip():
    recorder = StateRecorder()
    agents = [
        MockAgent(1.0, 2.0, 0.5, 50),
        MockAgent(10.0, 20.0, 1.0, 30),
    ]
    pg = MockPlayground(agents, [])

    for t in range(5):
        agents[0].base.position.x = 1.0 + t
        agents[1].base.position.y = 20.0 + t
        recorder.capture_state(pg, t * 0.1)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        filepath = f.name

    try:
        recorder.save(filepath, {"map_name": "TestMap"})
        replayer = StateReplayer(filepath)
        assert replayer.n_frames == 5
        assert replayer.metadata["map_name"] == "TestMap"

        state = replayer.get_frame_state(3)
        assert state["drone_states"][0]["x"] == 4.0
        assert state["drone_states"][1]["y"] == 23.0

        replayer.seek(2)
        assert replayer.current_frame == 2
        replayer.advance()
        assert replayer.current_frame == 3
    finally:
        import os
        os.unlink(filepath)


def test_dtype_definitions():
    assert len(DRONE_DTYPE) == 5
    assert DRONE_DTYPE.names == ("x", "y", "angle", "active", "health")
    assert len(BOMB_DTYPE) == 4
    assert BOMB_DTYPE.names == ("x", "y", "angle", "active")

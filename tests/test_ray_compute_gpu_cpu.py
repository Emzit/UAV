"""
Regression tests: GPU shader vs CPU ray casting must agree on sensor output.

Only one OpenGL window may be active at a time; each scenario runs CPU first,
closes the window, then runs GPU and compares snapshots.

Guards against semantic/lidar drift after changes to ray_compute, id_compute.glsl,
or ClosedPlayground shader defaults.
"""

from __future__ import annotations

import math
import platform
from typing import Callable, List, Tuple, TypeVar

import numpy as np
import pytest

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.playground import Playground
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData

MAP_SIZE = (400, 400)
DISPOSAL_CENTER_POS = ((300.0, 300.0), 0.0)
ORBIT_RADIUS = 30.0
ZERO_COMMAND: CommandsDict = {
    "forward": 0.0,
    "lateral": 0.0,
    "rotation": 0.0,
    "grasper": 0,
}

DIST_ATOL = 4.0
DIST_RTOL = 0.08
ANGLE_ATOL = 0.05
MAX_BOMB_COUNT_DELTA = 1

requires_gpu_raycast = pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="GPU ray casting is disabled on macOS (ClosedPlayground fallback).",
)

T = TypeVar("T")


class RayTestPlayground(ClosedPlayground):
    """Playground with explicit GPU/CPU ray casting (bypasses platform default)."""

    def __init__(self, use_shaders: bool, size: Tuple[int, int] = MAP_SIZE):
        Playground.__init__(
            self,
            size=size,
            seed=None,
            background=(220, 220, 220),
            use_shaders=use_shaders,
        )
        self._walls_creation(6)


class RayTestDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.semantic()._noise = False
        self.lidar()._noise = False

    def define_message_for_all(self):
        return None

    def control(self) -> CommandsDict:
        return dict(ZERO_COMMAND)


BombRows = List[Tuple[int, float, float, bool]]


def _build_scene(
    use_shaders: bool,
    drone_pos: Tuple[float, float],
    drone_angle: float,
    bomb_pos: Tuple[float, float],
) -> Tuple[RayTestPlayground, RayTestDrone, Bomb]:
    playground = RayTestPlayground(use_shaders=use_shaders)
    misc = MiscData(size_area=MAP_SIZE, number_drones=1)
    drone = RayTestDrone(identifier=0, misc_data=misc)
    disposal = DisposalCenter(size=(80, 80))
    playground.add(disposal, DISPOSAL_CENTER_POS)
    bomb = Bomb(disposal_center=disposal)
    playground.add(bomb, (bomb_pos, 0.0))
    playground.add(drone, (drone_pos, drone_angle))
    return playground, drone, bomb


def _step(playground: RayTestPlayground, drone: RayTestDrone, n: int = 1) -> None:
    for _ in range(n):
        playground.step(all_commands={drone: ZERO_COMMAND})


def _semantic_bomb_rows(drone: RayTestDrone) -> BombRows:
    values = drone.semantic_values()
    if not values:
        return []
    rows: BombRows = []
    for data in values:
        if data.entity_type != DroneSemanticSensor.TypeEntity.BOMB:
            continue
        rows.append(
            (
                int(data.entity_type.value),
                float(data.angle),
                float(data.distance),
                bool(data.grasped),
            )
        )
    return sorted(rows, key=lambda row: row[1])


def _lidar_distances(drone: RayTestDrone) -> np.ndarray:
    values = drone.lidar_values()
    assert values is not None
    return np.asarray(values, dtype=np.float64)


def _raw_hitpoint_ids(drone: RayTestDrone, sensor_name: str) -> np.ndarray:
    sensor = getattr(drone, sensor_name)()
    hitpoints = sensor._hitpoints
    assert hitpoints is not None
    return hitpoints[:, 8].astype(np.int64)


def _with_scene(
    use_shaders: bool,
    drone_pos: Tuple[float, float],
    drone_angle: float,
    bomb_pos: Tuple[float, float],
    body: Callable[[RayTestPlayground, RayTestDrone, Bomb], T],
) -> T:
    playground, drone, bomb = _build_scene(
        use_shaders, drone_pos, drone_angle, bomb_pos
    )
    try:
        return body(playground, drone, bomb)
    finally:
        # cleanup() only — arcade cannot reliably open a new Window after close()
        # in the same pytest process (see test_sensor_ouput.py pattern).
        playground.cleanup()


def _assert_bomb_rows_compatible(gpu_rows: BombRows, cpu_rows: BombRows) -> None:
    """GPU and CPU should agree on nearby bomb detections (counts may differ by 1)."""
    if not cpu_rows:
        assert gpu_rows == []
        return

    assert len(gpu_rows) >= 1, (
        f"GPU must detect bomb when CPU does: gpu={gpu_rows} cpu={cpu_rows}"
    )
    assert abs(len(gpu_rows) - len(cpu_rows)) <= MAX_BOMB_COUNT_DELTA, (
        f"bomb count delta too large: gpu={len(gpu_rows)} cpu={len(cpu_rows)}\n"
        f"  gpu={gpu_rows}\n  cpu={cpu_rows}"
    )

    matched = 0
    used_gpu: set[int] = set()
    for cpu_row in cpu_rows:
        best_idx = None
        best_delta = float("inf")
        for idx, gpu_row in enumerate(gpu_rows):
            if idx in used_gpu:
                continue
            delta = abs(gpu_row[1] - cpu_row[1])
            if delta < best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is None or best_delta > ANGLE_ATOL:
            continue
        used_gpu.add(best_idx)
        gpu_row = gpu_rows[best_idx]
        assert gpu_row[0] == cpu_row[0]
        assert gpu_row[3] == cpu_row[3]
        assert gpu_row[2] == pytest.approx(
            cpu_row[2], abs=DIST_ATOL, rel=DIST_RTOL
        )
        matched += 1

    assert matched >= min(len(gpu_rows), len(cpu_rows)), (
        f"too few matched bomb rays: matched={matched} "
        f"gpu={len(gpu_rows)} cpu={len(cpu_rows)}"
    )


def _collect_orbit_signatures(
    use_shaders: bool,
    orbit_steps: int,
    warmup_steps: int = 3,
) -> List[BombRows]:
    def body(playground: RayTestPlayground, drone: RayTestDrone, _bomb: Bomb):
        _step(playground, drone, warmup_steps)
        signatures: List[BombRows] = []
        for step_i in range(orbit_steps):
            phi = 2.0 * math.pi * step_i / orbit_steps
            dx = ORBIT_RADIUS * math.cos(phi)
            dy = ORBIT_RADIUS * math.sin(phi)
            heading = math.atan2(-dy, -dx)
            drone.move_to(((dx, dy), heading), allow_overlapping=True)
            _step(playground, drone, 1)
            signatures.append(_semantic_bomb_rows(drone))
        return signatures

    return _with_scene(
        use_shaders,
        (ORBIT_RADIUS, 0.0),
        0.0,
        (0.0, 0.0),
        body,
    )


@requires_gpu_raycast
def test_semantic_bomb_static_position_matches_cpu():
    """Semantic bomb detections: GPU vs CPU (sequential windows)."""

    def cpu_body(pg, drone, _bomb):
        _step(pg, drone, 3)
        return _semantic_bomb_rows(drone)

    def gpu_body(pg, drone, _bomb):
        _step(pg, drone, 3)
        return _semantic_bomb_rows(drone)

    cpu_rows = _with_scene(False, (30.0, 0.0), math.pi, (0.0, 0.0), cpu_body)
    gpu_rows = _with_scene(True, (30.0, 0.0), math.pi, (0.0, 0.0), gpu_body)
    _assert_bomb_rows_compatible(gpu_rows, cpu_rows)


@requires_gpu_raycast
def test_lidar_static_position_matches_cpu():
    """Lidar distance vector: GPU vs CPU (sequential windows)."""

    def cpu_body(pg, drone, _bomb):
        _step(pg, drone, 3)
        return _lidar_distances(drone)

    def gpu_body(pg, drone, _bomb):
        _step(pg, drone, 3)
        return _lidar_distances(drone)

    cpu_dist = _with_scene(False, (30.0, 0.0), math.pi, (0.0, 0.0), cpu_body)
    gpu_dist = _with_scene(True, (30.0, 0.0), math.pi, (0.0, 0.0), gpu_body)
    np.testing.assert_allclose(gpu_dist, cpu_dist, atol=DIST_ATOL, rtol=DIST_RTOL)


@requires_gpu_raycast
def test_gpu_raw_hitpoint_uids_are_known_entities():
    """GPU path must not return invalid UIDs (semantic empty-read regression)."""

    def body(playground, drone, bomb):
        _step(playground, drone, 3)
        sem_ids = _raw_hitpoint_ids(drone, "semantic")
        lidar_ids = _raw_hitpoint_ids(drone, "lidar")
        unknown = [
            int(uid)
            for uid in np.concatenate([sem_ids, lidar_ids])
            if uid and uid not in playground._uids_to_entities
        ]
        assert unknown == [], f"unknown GPU UIDs: {unknown[:10]}"
        assert bomb.uid in set(int(u) for u in sem_ids)
        assert len(_semantic_bomb_rows(drone)) > 0
        return None

    _with_scene(True, (30.0, 0.0), math.pi, (0.0, 0.0), body)


@requires_gpu_raycast
@pytest.mark.parametrize("orbit_steps", [12, 24])
def test_orbit_semantic_bomb_parity_cpu_gpu(orbit_steps: int):
    """Orbiting drone: GPU semantic bomb list tracks CPU across steps."""
    cpu_sigs = _collect_orbit_signatures(use_shaders=False, orbit_steps=orbit_steps)
    gpu_sigs = _collect_orbit_signatures(use_shaders=True, orbit_steps=orbit_steps)
    assert len(gpu_sigs) == len(cpu_sigs)
    for step_i, (gpu_rows, cpu_rows) in enumerate(zip(gpu_sigs, cpu_sigs)):
        _assert_bomb_rows_compatible(gpu_rows, cpu_rows)


@requires_gpu_raycast
def test_orbit_gpu_never_empty_when_bomb_nearby():
    """GPU must not return empty semantic when bomb is within sensor range."""

    def body(playground, drone, _bomb):
        _step(playground, drone, 3)
        for step_i in range(24):
            phi = 2.0 * math.pi * step_i / 24
            dx = ORBIT_RADIUS * math.cos(phi)
            dy = ORBIT_RADIUS * math.sin(phi)
            heading = math.atan2(-dy, -dx)
            drone.move_to(((dx, dy), heading), allow_overlapping=True)
            _step(playground, drone, 1)
            assert drone.semantic_values() is not None
            assert len(_semantic_bomb_rows(drone)) > 0, (
                f"step {step_i}: GPU semantic empty near bomb"
            )
        return None

    _with_scene(True, (ORBIT_RADIUS, 0.0), 0.0, (0.0, 0.0), body)


@requires_gpu_raycast
@pytest.mark.parametrize(
    "distance,angle_deg",
    [
        (20, 0),
        (30, 45),
        (40, 120),
        (24, 200),
    ],
)
def test_grid_semantic_bomb_parity_cpu_gpu(distance: int, angle_deg: int):
    """Several bomb positions: GPU vs CPU semantic parity."""
    theta = math.radians(angle_deg)
    bx = distance * math.cos(theta)
    by = distance * math.sin(theta)
    drone_pos = (0.0, 0.0)
    bomb_pos = (bx, by)

    def cpu_body(pg, drone, _bomb):
        _step(pg, drone, 1)
        return _semantic_bomb_rows(drone)

    def gpu_body(pg, drone, _bomb):
        _step(pg, drone, 1)
        return _semantic_bomb_rows(drone)

    cpu_rows = _with_scene(False, drone_pos, 0.0, bomb_pos, cpu_body)
    gpu_rows = _with_scene(True, drone_pos, 0.0, bomb_pos, gpu_body)
    _assert_bomb_rows_compatible(gpu_rows, cpu_rows)

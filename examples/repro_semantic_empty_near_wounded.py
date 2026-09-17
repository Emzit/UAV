"""
Reproduce semantic sensor empty reads when a bomb is nearby.

Scenarios:
  A - Static grid: drone at origin, bomb at various distances/angles
  B - Grasp control: grasped bomb is invisible to semantic (expected)
  C - Before step(): initial null sensor vs empty list
  D - Orbit: drone circles bomb at map center, print semantic each step

Usage (use the project venv):
  .venv/bin/python examples/repro_semantic_empty_near_wounded.py
  .venv/bin/python examples/repro_semantic_empty_near_wounded.py --orbit
  .venv/bin/python examples/repro_semantic_empty_near_wounded.py --use-cpu
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.bomb import Bomb
from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.playground import Playground
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData

MAP_SIZE = (400, 400)
NEAR_THRESHOLD = 50.0
DISTANCES = (12, 18, 24, 30, 40, 50)
ANGLE_STEP_DEG = 5
DISPOSAL_CENTER_POS = ((300.0, 300.0), 0.0)
ZERO_COMMAND: CommandsDict = {
    "forward": 0.0,
    "lateral": 0.0,
    "rotation": 0.0,
    "grasper": 0,
}


class ReproClosedPlayground(ClosedPlayground):
    """Closed playground with explicit GPU/CPU ray casting mode."""

    def __init__(
        self,
        size: Tuple[int, int] = MAP_SIZE,
        use_cpu: bool = False,
        border_thickness: int = 6,
    ):
        background = (220, 220, 220)
        Playground.__init__(
            self,
            size=size,
            seed=None,
            background=background,
            use_shaders=not use_cpu,
        )
        self._walls_creation(border_thickness)


class ReproDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.semantic()._noise = False

    def define_message_for_all(self):
        pass

    def control(self) -> CommandsDict:
        return dict(ZERO_COMMAND)


@dataclass
class SemanticSnapshot:
    values: Optional[list]
    empty_list: bool
    no_bomb: bool
    no_bomb_ungrasped: bool
    bomb_count: int
    semantic_len: int
    raw_nonzero_uid_count: int

    @property
    def is_fail_near(self) -> bool:
        return self.empty_list or self.no_bomb or self.no_bomb_ungrasped


def euclidean_dist(
    a: Tuple[float, float], b: Tuple[float, float]
) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def analyze_semantic(drone: ReproDrone) -> SemanticSnapshot:
    values = drone.semantic_values()
    empty_list = values is not None and len(values) == 0
    bomb_count = 0
    no_bomb = False
    no_bomb_ungrasped = False

    if values is None:
        no_bomb = True
        no_bomb_ungrasped = True
    else:
        bombs = [
            d
            for d in values
            if d.entity_type == DroneSemanticSensor.TypeEntity.BOMB
        ]
        bomb_count = len(bombs)
        no_bomb = bomb_count == 0
        no_bomb_ungrasped = not any(not d.grasped for d in bombs)

    hitpoints = drone.semantic()._hitpoints
    if hitpoints is not None:
        raw_nonzero_uid_count = int(np.count_nonzero(hitpoints[:, 8].astype(int)))
    else:
        raw_nonzero_uid_count = 0

    return SemanticSnapshot(
        values=values,
        empty_list=empty_list,
        no_bomb=no_bomb,
        no_bomb_ungrasped=no_bomb_ungrasped,
        bomb_count=bomb_count,
        semantic_len=0 if values is None else len(values),
        raw_nonzero_uid_count=raw_nonzero_uid_count,
    )


def format_bomb_entries(drone: ReproDrone) -> str:
    values = drone.semantic_values()
    if not values:
        return "[none]"
    parts = []
    for data in values:
        if data.entity_type != DroneSemanticSensor.TypeEntity.BOMB:
            continue
        g = 1 if data.grasped else 0
        parts.append(f"B d={data.distance:.1f} ang={data.angle:.2f} g={g}")
    return " ".join(parts) if parts else "[none]"


def setup_playground(use_cpu: bool) -> ReproClosedPlayground:
    return ReproClosedPlayground(size=MAP_SIZE, use_cpu=use_cpu)


def add_drone_and_bomb(
    playground: ReproClosedPlayground,
    drone_pos: Tuple[float, float],
    drone_angle: float,
    bomb_pos: Tuple[float, float],
    bomb_angle: float = 0.0,
) -> Tuple[ReproDrone, Bomb]:
    misc = MiscData(size_area=MAP_SIZE, number_drones=1)
    drone = ReproDrone(identifier=0, misc_data=misc)
    disposal_center = DisposalCenter(size=(80, 80))
    playground.add(disposal_center, DISPOSAL_CENTER_POS)
    bomb = Bomb(disposal_center=disposal_center)
    playground.add(bomb, ((bomb_pos[0], bomb_pos[1]), bomb_angle))
    playground.add(drone, ((drone_pos[0], drone_pos[1]), drone_angle))
    return drone, bomb


def step_playground(
    playground: ReproClosedPlayground, drone: ReproDrone, command: CommandsDict
) -> None:
    playground.step(all_commands={drone: command})


def reset_ray_compute(playground: ReproClosedPlayground) -> None:
    """
    Drop cached RayCompute after removing drones.

    Removed sensors stay in RayCompute._sensors otherwise, which corrupts GPU
    buffers and causes intermittent empty semantic reads.
    """
    playground._ray_compute = None


def clear_drones_and_bombs(playground: ReproClosedPlayground) -> None:
    for agent in list(playground.agents):
        playground.remove(agent, definitive=True)
    for element in list(playground.elements):
        if isinstance(element, (Bomb, DisposalCenter)):
            playground.remove(element, definitive=True)
    reset_ray_compute(playground)


def flush_gpu_before_step(playground: ReproClosedPlayground) -> None:
    """Redraw UID view and sync GPU before ray cast (shader path only)."""
    if not playground._use_shaders:
        return
    ray_compute = playground._ray_compute
    if ray_compute is None:
        return
    ray_compute._id_view.update_and_draw_in_framebuffer(force=True)
    playground.window.ctx.finish()


def step_orbit_frame(playground: ReproClosedPlayground, drone: ReproDrone) -> None:
    flush_gpu_before_step(playground)
    step_playground(playground, drone, ZERO_COMMAND)


def scenario_a(playground: ReproClosedPlayground) -> int:
    print("\n=== Scenario A: static grid (drone at origin) ===")
    n_angles = len(range(0, 360, ANGLE_STEP_DEG))
    total_expected = len(DISTANCES) * n_angles
    print(
        f"  {total_expected} samples — reposition only (not reset per cell); "
        "use --skip-a to skip",
        flush=True,
    )

    fail_count = 0
    total = 0
    failures: List[str] = []

    clear_drones_and_bombs(playground)

    drone, bomb = add_drone_and_bomb(
        playground, (0.0, 0.0), 0.0, (float(DISTANCES[0]), 0.0)
    )

    for dist in DISTANCES:
        print(f"  distance={dist}px ...", flush=True)
        for angle_deg in range(0, 360, ANGLE_STEP_DEG):
            total += 1
            theta = math.radians(angle_deg)
            bx = dist * math.cos(theta)
            by = dist * math.sin(theta)

            bomb.move_to(((bx, by), 0.0), allow_overlapping=True)
            drone.move_to(((0.0, 0.0), 0.0), allow_overlapping=True)
            step_playground(playground, drone, ZERO_COMMAND)

            true_dist = euclidean_dist(
                tuple(drone.true_position()), tuple(bomb.true_position())
            )
            snap = analyze_semantic(drone)

            if true_dist <= NEAR_THRESHOLD and snap.is_fail_near:
                fail_count += 1
                msg = (
                    f"  FAIL d={dist} theta={angle_deg}deg true_dist={true_dist:.1f} "
                    f"len={snap.semantic_len} bombs={snap.bomb_count} "
                    f"raw_uids={snap.raw_nonzero_uid_count} "
                    f"empty={snap.empty_list} no_b={snap.no_bomb} "
                    f"no_ungrasped={snap.no_bomb_ungrasped}"
                )
                failures.append(msg)
                if len(failures) <= 20:
                    print(msg)

    print(f"Scenario A: {fail_count}/{total} FAIL (near threshold {NEAR_THRESHOLD}px)")
    if fail_count > 20:
        print(f"  (... {fail_count - 20} more failures omitted)")
    return fail_count


def scenario_b(playground: ReproClosedPlayground) -> int:
    print("\n=== Scenario B: grasped bomb (expected invisible) ===")
    clear_drones_and_bombs(playground)

    drone, bomb = add_drone_and_bomb(
        playground, (15.0, 0.0), 0.0, (25.0, 0.0)
    )
    grasped = False
    for _ in range(60):
        step_playground(
            playground,
            drone,
            {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 1},
        )
        if bomb in drone.grasper.grasped_bombs:
            grasped = True
            break

    true_dist = euclidean_dist(
        tuple(drone.true_position()), tuple(bomb.true_position())
    )
    snap = analyze_semantic(drone)
    print(f"  grasped={grasped} true_dist={true_dist:.1f}")
    print(
        f"  semantic: len={snap.semantic_len} bombs={snap.bomb_count} "
        f"no_ungrasped={snap.no_bomb_ungrasped} empty={snap.empty_list}"
    )
    if grasped and snap.no_bomb_ungrasped:
        print("  OK (expected): grasped bomb not reported as ungrasped")
    elif not grasped:
        print("  NOTE: grasp did not occur; scenario B inconclusive")
    return 0


def scenario_c(playground: ReproClosedPlayground) -> int:
    print("\n=== Scenario C: before playground.step() ===")
    clear_drones_and_bombs(playground)

    drone, _bomb = add_drone_and_bomb(
        playground, (0.0, 0.0), 0.0, (30.0, 0.0)
    )
    snap = analyze_semantic(drone)
    values = drone.semantic_values()
    has_nan = False
    if values:
        has_nan = any(
            getattr(d, "distance", None) is not None
            and math.isnan(d.distance)
            for d in values
        )
    print(f"  semantic_len={snap.semantic_len} empty_list={snap.empty_list} has_nan={has_nan}")
    print("  Expected: non-empty null sensor (nan entries), not [] — differs from 'empty' after step")
    return 0


def scenario_d(
    playground: ReproClosedPlayground,
    orbit_radius: float,
    orbit_steps: int,
    debug_uids: bool = False,
) -> int:
    print(
        f"\n=== Scenario D: orbit bomb at (0,0), R={orbit_radius}, "
        f"steps={orbit_steps} ==="
    )
    if playground._use_shaders:
        print(
            "  GPU shaders can be flaky; use --use-cpu for stable reproduction",
            flush=True,
        )

    clear_drones_and_bombs(playground)

    drone, bomb = add_drone_and_bomb(
        playground, (orbit_radius, 0.0), 0.0, (0.0, 0.0)
    )
    n_ray_sensors = len(playground.ray_compute._sensors)
    print(f"  ray sensors registered: {n_ray_sensors} (expected 2)", flush=True)

    for _ in range(3):
        step_orbit_frame(playground, drone)

    fail_count = 0
    fail_phis: List[float] = []

    for step_i in range(orbit_steps):
        phi = 2.0 * math.pi * step_i / orbit_steps
        dx = orbit_radius * math.cos(phi)
        dy = orbit_radius * math.sin(phi)
        heading = math.atan2(-dy, -dx)
        drone.move_to(((dx, dy), heading), allow_overlapping=True)

        step_orbit_frame(playground, drone)

        true_dist = euclidean_dist(
            tuple(drone.true_position()), tuple(bomb.true_position())
        )
        snap = analyze_semantic(drone)
        phi_deg = math.degrees(phi)
        fail_mark = ""
        if true_dist <= NEAR_THRESHOLD and snap.is_fail_near:
            fail_count += 1
            fail_phis.append(phi_deg)
            fail_mark = " ** FAIL **"
            if debug_uids and fail_count == 1:
                hp_dbg = drone.semantic()._hitpoints
                if hp_dbg is not None:
                    unique_ids = sorted({int(u) for u in hp_dbg[:, 8] if u})
                    known = [u for u in unique_ids if u in playground._uids_to_entities]
                    unknown = [u for u in unique_ids if u not in playground._uids_to_entities]
                    print(
                        f"  DEBUG bomb.uid={bomb.uid} known_uids={known[:8]} "
                        f"unknown_sample={unknown[:8]} n_unknown={len(unknown)}",
                        flush=True,
                    )

        print(
            f"step={step_i:03d} phi={phi_deg:5.1f}deg dist={true_dist:.1f} "
            f"empty={int(snap.empty_list)} bombs={snap.bomb_count} "
            f"raw_uids={snap.raw_nonzero_uid_count} "
            f"{format_bomb_entries(drone)}{fail_mark}"
        )

    pct = 100.0 * fail_count / orbit_steps if orbit_steps else 0.0
    print(
        f"Scenario D: {fail_count}/{orbit_steps} FAIL steps ({pct:.1f}%)"
    )
    if fail_phis:
        print(f"  FAIL phi (deg): {[round(p, 1) for p in fail_phis]}")
    return fail_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce semantic sensor empty reads near bombs"
    )
    parser.add_argument(
        "--orbit",
        action="store_true",
        help="Run only scenario D (orbit)",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Disable GPU shaders for ray casting",
    )
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=30.0,
        help="Orbit radius in pixels (default: 30)",
    )
    parser.add_argument(
        "--orbit-steps",
        type=int,
        default=72,
        help="Steps per full orbit (default: 72)",
    )
    parser.add_argument(
        "--skip-a",
        action="store_true",
        help="Skip scenario A grid scan",
    )
    parser.add_argument(
        "--debug-uids",
        action="store_true",
        help="Print sample invalid UIDs on first FAIL step",
    )
    args = parser.parse_args()

    playground = setup_playground(use_cpu=args.use_cpu)
    if args.use_cpu:
        print("Ray casting: CPU (use_shaders=False)")
    else:
        print("Ray casting: GPU shaders (use_shaders=True)")

    exit_code = 0

    if args.orbit:
        fails = scenario_d(
            playground,
            args.orbit_radius,
            args.orbit_steps,
            debug_uids=args.debug_uids,
        )
        return 1 if fails else 0

    if not args.skip_a:
        fails_a = scenario_a(playground)
        if fails_a:
            exit_code = 1

    fails_b = scenario_b(playground)
    if fails_b:
        exit_code = 1

    scenario_c(playground)

    fails_d = scenario_d(
        playground,
        args.orbit_radius,
        args.orbit_steps,
        debug_uids=args.debug_uids,
    )
    if fails_d:
        exit_code = 1

    print(f"\nDone. exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

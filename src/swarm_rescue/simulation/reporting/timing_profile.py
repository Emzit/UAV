"""Opt-in wall-clock timings for one simulation round.

The profile is deliberately separate from the score's simulation timestamps.
It measures where real execution time is spent, without changing the physics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class StepTiming:
    step: int
    started_at: float
    ended_at: float
    phases: Dict[str, float]
    details: Dict[str, float]

    @property
    def total(self) -> float:
        return self.ended_at - self.started_at


@dataclass
class DetailTotals:
    calls: int = 0
    inclusive: float = 0.0
    exclusive: float = 0.0
    maximum: float = 0.0


class TimingProfile:
    """Collect non-overlapping phases and render callbacks for a round."""

    def __init__(self) -> None:
        self.steps: List[StepTiming] = []
        self._current_step = 0
        self._current_start = 0.0
        self._current_phases: Dict[str, float] = {}
        self._current_details: Dict[str, float] = {}
        self._draw_times: List[float] = []
        self._control_by_drone: Dict[int, List[float]] = {}
        self._control_by_mode: Dict[str, List[float]] = {}
        self._detail_totals: Dict[str, DetailTotals] = {}
        self._detail_stacks: Dict[int, List[List[float]]] = {}
        self._finalize_time = 0.0

    def begin_step(self, step: int) -> None:
        if self._current_start:
            raise RuntimeError("Previous timing step was not finished")
        self._current_step = step
        self._current_phases = {}
        self._current_details = {}
        self._current_start = time.perf_counter()

    def add_phase(self, name: str, seconds: float) -> None:
        if not self._current_start:
            return
        self._current_phases[name] = (
            self._current_phases.get(name, 0.0) + max(0.0, seconds)
        )

    def add_control(self, drone_id: int, seconds: float,
                    mode: str = "unknown") -> None:
        self.add_phase("control", seconds)
        self._control_by_drone.setdefault(drone_id, []).append(max(0.0, seconds))
        self._control_by_mode.setdefault(mode, []).append(max(0.0, seconds))

    def instrument_drone(self, drone: Any, drone_id: int) -> None:
        """Wrap blue-controller methods on this drone only, idempotently.

        Reinstalling each step handles grids/planners replaced by a GPS
        re-anchor, while ordinary non-profiled runs never install wrappers.
        """
        methods = (
            (drone, "_control_explore", "controller.explore"),
            (drone, "_control_carry", "controller.carry"),
            (drone, "_control_chase_bomb", "controller.chase"),
            (drone, "_control_to_remembered_bomb", "controller.navigate"),
            (drone, "_move_toward", "controller.move_toward"),
            (drone, "_apply_traffic", "controller.traffic"),
            (getattr(drone, "_localizer", None), "update", "localization.update"),
            (getattr(drone, "_grid", None), "update_from_scan", "map.scan"),
            (getattr(drone, "_grid", None), "check_segment", "map.check_segment"),
            (getattr(drone, "_frontier", None), "update", "frontier.update"),
            (getattr(drone, "_planner", None), "plan", "planner.astar"),
            (getattr(drone, "_allocator", None), "bid", "allocation.bid"),
            (getattr(drone, "_return_nav", None), "next_waypoint",
             "return.next_waypoint"),
        )
        for owner, method_name, label in methods:
            if owner is None:
                continue
            original = getattr(owner, method_name, None)
            if not callable(original):
                continue
            if getattr(original, "_timing_profile_owner", None) is self:
                continue

            @wraps(original)
            def timed(*args: Any, _original: Callable = original,
                      _label: str = label, **kwargs: Any) -> Any:
                return self._time_detail(drone, drone_id, _label,
                                         _original, args, kwargs)

            timed._timing_profile_owner = self  # type: ignore[attr-defined]
            setattr(owner, method_name, timed)

    def _time_detail(self, drone: Any, drone_id: int, label: str,
                     method: Callable, args: tuple, kwargs: dict) -> Any:
        mode = str(getattr(drone, "_mode", "unknown"))
        key = f"{mode}/{label}"
        stack = self._detail_stacks.setdefault(drone_id, [])
        frame = [0.0]
        stack.append(frame)
        started = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            stack.pop()
            if stack:
                stack[-1][0] += elapsed
            exclusive = max(0.0, elapsed - frame[0])
            totals = self._detail_totals.setdefault(key, DetailTotals())
            totals.calls += 1
            totals.inclusive += elapsed
            totals.exclusive += exclusive
            totals.maximum = max(totals.maximum, elapsed)
            if self._current_start:
                self._current_details[key] = (
                    self._current_details.get(key, 0.0) + exclusive
                )

    def end_step(self) -> None:
        if not self._current_start:
            raise RuntimeError("No timing step is active")
        ended_at = time.perf_counter()
        self.steps.append(StepTiming(
            self._current_step, self._current_start, ended_at,
            self._current_phases.copy(), self._current_details.copy(),
        ))
        self._current_start = 0.0
        self._current_phases = {}
        self._current_details = {}

    def add_draw(self, seconds: float) -> None:
        self._draw_times.append(max(0.0, seconds))

    def add_finalize(self, seconds: float) -> None:
        self._finalize_time += max(0.0, seconds)

    @staticmethod
    def _stats(values: List[float]) -> str:
        ordered = sorted(values)
        count = len(ordered)
        total = sum(ordered)
        mean_ms = 1000.0 * total / count
        p95_ms = 1000.0 * ordered[max(0, math.ceil(0.95 * count) - 1)]
        return (f"total={total:.2f}s avg={mean_ms:.1f}ms "
                f"p95={p95_ms:.1f}ms max={1000.0 * ordered[-1]:.1f}ms")

    def format_report(self, interval_size: int = 100) -> str:
        if not self.steps:
            return "Timing profile: no simulation steps were recorded."
        if interval_size < 1:
            raise ValueError("interval_size must be positive")

        totals = [sample.total for sample in self.steps]
        phase_names = sorted({name for sample in self.steps
                              for name in sample.phases})
        phases = {name: [sample.phases.get(name, 0.0) for sample in self.steps]
                  for name in phase_names}
        other = [max(0.0, sample.total - sum(sample.phases.values()))
                 for sample in self.steps]
        wall_interval = self.steps[-1].ended_at - self.steps[0].started_at
        lines = [
            "Timing profile (wall-clock; enabled by --profile-timing):",
            (f"  {len(self.steps)} steps, first-to-last update interval "
             f"{wall_interval:.2f}s"),
            f"  on_update: {self._stats(totals)}",
        ]
        for name in sorted(phase_names, key=lambda key: sum(phases[key]),
                           reverse=True):
            lines.append(f"  {name}: {self._stats(phases[name])}")
        lines.append(f"  update_other: {self._stats(other)}")
        if self._draw_times:
            lines.append(
                f"  on_draw ({len(self._draw_times)} calls): "
                f"{self._stats(self._draw_times)}"
            )
        else:
            lines.append("  on_draw: no callbacks observed")
        if self._finalize_time:
            lines.append(f"  round_cleanup: {self._finalize_time:.2f}s "
                         "(excluded from on_update timings)")
        for drone_id, values in sorted(
                self._control_by_drone.items(),
                key=lambda item: sum(item[1]), reverse=True)[:3]:
            lines.append(f"  slow_control drone {drone_id}: "
                         f"{self._stats(values)}")
        if self._detail_totals:
            lines.append("  control mode totals:")
            for mode, values in sorted(
                    self._control_by_mode.items(),
                    key=lambda item: sum(item[1]), reverse=True):
                lines.append(f"    {mode} ({len(values)} calls): "
                             f"{self._stats(values)}")
            lines.append(
                "  control detail hotspots "
                "(self excludes wrapped children; inclusive overlaps):"
            )
            for key, stat in sorted(
                    self._detail_totals.items(),
                    key=lambda item: item[1].exclusive, reverse=True)[:15]:
                lines.append(
                    f"    {key}: calls={stat.calls} "
                    f"self={stat.exclusive:.2f}s "
                    f"inclusive={stat.inclusive:.2f}s "
                    f"avg_self={1000.0 * stat.exclusive / stat.calls:.2f}ms "
                    f"max_inclusive={1000.0 * stat.maximum:.1f}ms"
                )
            lines.append("  control detail parents (inclusive, overlaps children):")
            for key, stat in sorted(
                    self._detail_totals.items(),
                    key=lambda item: item[1].inclusive, reverse=True)[:5]:
                lines.append(
                    f"    {key}: calls={stat.calls} "
                    f"inclusive={stat.inclusive:.2f}s"
                )
            control_total = sum(sum(values)
                                for values in self._control_by_drone.values())
            measured_self = sum(stat.exclusive
                                for stat in self._detail_totals.values())
            lines.append(
                "  control_unprofiled_approx: "
                f"{max(0.0, control_total - measured_self):.2f}s "
                "(includes wrapper overhead and unwrapped code)"
            )
        lines.append(
            f"  {interval_size}-step wall rates "
            "(include event-loop/render delays):"
        )
        for index in range(0, len(self.steps), interval_size):
            group = self.steps[index:index + interval_size]
            interval_start = (self.steps[index - 1].ended_at if index
                              else group[0].started_at)
            duration = group[-1].ended_at - interval_start
            rate = len(group) / duration if duration > 0 else 0.0
            phase_totals: Dict[str, float] = {}
            for sample in group:
                for name, elapsed in sample.phases.items():
                    phase_totals[name] = phase_totals.get(name, 0.0) + elapsed
            top_phases = sorted(phase_totals.items(),
                                key=lambda item: item[1], reverse=True)[:3]
            top_text = ", ".join(
                f"{name}={1000.0 * elapsed / len(group):.1f}ms"
                for name, elapsed in top_phases
            )
            update_ms = 1000.0 * sum(sample.total for sample in group) / len(group)
            lines.append(f"    steps {group[0].step}-{group[-1].step}: "
                         f"{rate:.1f} steps/s, update={update_ms:.1f}ms"
                         + (f", top: {top_text}" if top_text else ""))
            detail_totals: Dict[str, float] = {}
            for sample in group:
                for name, elapsed in sample.details.items():
                    detail_totals[name] = detail_totals.get(name, 0.0) + elapsed
            if detail_totals:
                top_details = sorted(detail_totals.items(),
                                     key=lambda item: item[1], reverse=True)[:3]
                lines.append("      control self: " + ", ".join(
                    f"{name}={1000.0 * elapsed / len(group):.1f}ms"
                    for name, elapsed in top_details
                ))
        return "\n".join(lines)

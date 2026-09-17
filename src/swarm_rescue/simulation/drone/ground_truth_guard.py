"""Guard ground-truth drone APIs during competition control steps."""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract

_control_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "swarm_rescue_control_active",
    default=False,
)


class GroundTruthInControlError(RuntimeError):
    """Raised when a ground-truth API is used inside control() during competition."""


def control_active() -> bool:
    return _control_active.get()


def assert_ground_truth_allowed(api_name: str, *, competition_mode: bool) -> None:
    if competition_mode and control_active():
        raise GroundTruthInControlError(
            f"{api_name}() is disabled inside control() during competition/evaluation. "
            "Use measured_gps_position(), measured_compass_angle(), measured_velocity(), "
            "or measured_angular_velocity() instead."
        )


def run_control(drone: DroneAbstract):
    """Call drone.control() with the control-step guard active."""
    token = _control_active.set(True)
    try:
        return drone.control()
    finally:
        _control_active.reset(token)

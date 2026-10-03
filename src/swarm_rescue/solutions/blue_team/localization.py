"""Small, sensor-only pose filter for the blue controller.

The filter never reads simulator ground truth.  GPS/compass measurements are
temporally correlated, so they are deliberately not fused on every frame.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .contracts import PoseEstimate, WorldPoint


def _angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _point(value: object) -> Optional[WorldPoint]:
    try:
        x, y = float(value[0]), float(value[1])  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _finite(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class LocalizationManager:
    """Three-state EKF with conservative GPS recovery and odometry fallback."""

    def __init__(self, drone_id: int) -> None:
        self.drone_id = drone_id
        self._x: Optional[np.ndarray] = None
        self._p = np.diag((100.0, 100.0, math.radians(4.0) ** 2))
        self._step = -1
        self._last_gps_step = -100
        self._last_compass_step = -100
        self._recovery: list[Tuple[int, WorldPoint]] = []
        self.reanchored = False
        self._latest = PoseEstimate(None, None, math.inf, math.inf,
                                    "UNINITIALIZED", "world", -1, False)

    @property
    def latest(self) -> PoseEstimate:
        return self._latest

    def update(
        self,
        step: int,
        gps: object,
        compass: object,
        odometry: Optional[Sequence[float]],
    ) -> PoseEstimate:
        if step == self._step:
            return self._latest
        if step < self._step:
            raise ValueError("localization steps must increase")
        self.reanchored = False
        gps_point = _point(gps)
        heading = _finite(compass)
        if heading is not None:
            heading = _angle(heading)
        odom = self._checked_odometry(odometry)

        if self._x is None:
            self._step = step
            if gps_point is None or heading is None:
                self._latest = PoseEstimate(
                    gps_point, heading, math.inf if gps_point is None else 100.0,
                    math.inf if heading is None else math.radians(4.0) ** 2,
                    "HEADING_ONLY" if gps_point is None and heading is not None
                    else "UNINITIALIZED", "world", step, False,
                )
                return self._latest
            self._x = np.array((gps_point[0], gps_point[1], heading), dtype=float)
            self._last_gps_step = step
            self._last_compass_step = step
            return self._publish(step, "GPS_FUSED")

        gap = max(1, step - self._step)
        self._step = step
        if gap > 1:
            self._p += np.diag((1.0, 1.0, math.radians(1.0) ** 2)) * (gap - 1)
        if odom is not None:
            self._predict(*odom)
        else:
            self._p += np.diag((2.0, 2.0, math.radians(1.0) ** 2))

        source = "DEAD_RECKONING"
        if heading is not None and step - self._last_compass_step >= 8:
            residual = _angle(heading - float(self._x[2]))
            s = float(self._p[2, 2]) + 0.05
            if residual * residual <= 9.0 * s and abs(residual) <= math.pi / 2:
                self._correct(np.array((2,)), np.array((residual,)),
                              np.array(((0.05,),)))
                self._x[2] = _angle(float(self._x[2]))
                self._last_compass_step = step
                source = "ODOM_COMPASS_FUSED"

        if gps_point is not None and step - self._last_gps_step >= 8:
            residual = np.array(gps_point) - self._x[:2]
            innovation = self._p[:2, :2] + np.eye(2) * 250.0
            nis = float(residual @ np.linalg.solve(innovation, residual))
            if nis <= 11.83:
                self._correct(np.array((0, 1)), residual, np.eye(2) * 250.0)
                self._last_gps_step = step
                self._recovery.clear()
                source = "GPS_FUSED"
            else:
                # One noisy GPS sample must not teleport the map or route.
                if not self._recovery or step - self._recovery[-1][0] >= 5:
                    self._recovery.append((step, gps_point))
                    self._recovery = self._recovery[-3:]
                if len(self._recovery) == 3 and all(
                    math.dist(self._recovery[i][1], self._recovery[i - 1][1]) < 35.0
                    for i in (1, 2)
                ):
                    self._x[:2] = np.median(
                        np.array([item[1] for item in self._recovery]), axis=0
                    )
                    self._p[:2, :2] = np.eye(2) * 100.0
                    self._p[:2, 2] = self._p[2, :2] = 0.0
                    self._last_gps_step = step
                    self._recovery.clear()
                    self.reanchored = True
                    source = "GPS_FUSED"
                else:
                    source = "RELOCALIZING"

        self._p = (self._p + self._p.T) * 0.5
        if not np.all(np.isfinite(self._x)) or not np.all(np.isfinite(self._p)):
            self._x = None
            self._latest = PoseEstimate(None, None, math.inf, math.inf,
                                        "LOST", "world", step, False)
            return self._latest
        return self._publish(step, source)

    @staticmethod
    def _checked_odometry(value: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
        if value is None:
            return None
        try:
            d, alpha, theta = (float(value[i]) for i in range(3))
        except (IndexError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (d, alpha, theta)) or not -0.6 <= d <= 40.0:
            return None
        return max(0.0, d), _angle(alpha), _angle(theta)

    def _predict(self, distance: float, alpha: float, theta: float) -> None:
        assert self._x is not None
        phi = float(self._x[2]) + alpha
        c, s = math.cos(phi), math.sin(phi)
        f = np.eye(3)
        f[0, 2], f[1, 2] = -distance * s, distance * c
        g = np.array(((c, -distance * s, 0.0),
                      (s, distance * c, 0.0), (0.0, 0.0, 1.0)))
        q = np.diag((0.2 ** 2, math.radians(8.0) ** 2,
                     math.radians(1.0) ** 2))
        floor = np.diag((0.25, 0.25, math.radians(0.5) ** 2))
        self._x += np.array((distance * c, distance * s, theta))
        self._x[2] = _angle(float(self._x[2]))
        self._p = f @ self._p @ f.T + g @ q @ g.T + floor

    def _correct(self, indices: np.ndarray, residual: np.ndarray, r: np.ndarray) -> None:
        assert self._x is not None
        h = np.eye(3)[indices]
        s = h @ self._p @ h.T + r
        k = self._p @ h.T @ np.linalg.inv(s)
        self._x += k @ residual
        a = np.eye(3) - k @ h
        self._p = a @ self._p @ a.T + k @ r @ k.T

    def _publish(self, step: int, source: str) -> PoseEstimate:
        assert self._x is not None
        variance = float(np.linalg.eigvalsh(self._p[:2, :2])[-1])
        heading_variance = float(self._p[2, 2])
        valid = variance <= 1600.0 and heading_variance <= 0.25
        self._latest = PoseEstimate(
            (float(self._x[0]), float(self._x[1])), float(self._x[2]),
            variance, heading_variance, source, "world", step, valid,
        )
        return self._latest

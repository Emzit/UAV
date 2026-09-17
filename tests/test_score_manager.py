"""Tests for the blue-team ScoreManager, focusing on the time score.

The blue time score uses the same convention as the red team: at or before
half the timestep limit it is full (100), it reaches 0 at the limit, and it
is linear in between. There is no completion gate.
"""

import pytest

from swarm_rescue.simulation.reporting.score_manager import ScoreManager


def _manager(max_timestep_limit: int = 2000,
             total_number_bombs: int = 5,
             time_best_ratio: float = 0.5,
             **kwargs) -> ScoreManager:
    return ScoreManager(
        number_drones=10,
        max_timestep_limit=max_timestep_limit,
        max_walltime_limit=90,
        total_number_bombs=total_number_bombs,
        time_best_ratio=time_best_ratio,
        **kwargs,
    )


@pytest.mark.parametrize("elapsed,expected", [
    (0, 100.0),          # instant all-disposal: full
    (500, 100.0),        # before half the limit: full
    (1000, 100.0),       # exactly half the limit: full
    (1250, 75.0),        # midpoint between best and limit
    (1500, 50.0),        # three quarters of the limit
    (2000, 0.0),         # at the limit: zero
    (3000, 0.0),         # past the limit: clamped to zero
])
def test_time_score_curve(elapsed, expected):
    """score_timestep out of 100 mirrors the red-team formula."""
    _, _, score_timestep = _manager().compute_score(
        5, 0.0, 0.0, elapsed
    )
    assert score_timestep == pytest.approx(expected)


def test_time_score_needs_no_completion_gate():
    """Partial disposal does not zero the time score; incompleteness is
    reflected by the round running to the limit (elapsed = limit -> 0)."""
    # Round hits the limit without full disposal: time score 0.
    _, _, incomplete = _manager().compute_score(3, 0.0, 0.0, 2000)
    assert incomplete == 0.0
    # Half-way elapsed keeps full time score even with partial disposal
    # (unreachable in practice because the round only ends early on full
    # disposal, but the formula itself has no gate).
    _, _, early = _manager().compute_score(3, 0.0, 0.0, 800)
    assert early == 100.0


def test_time_score_contributes_to_total():
    manager = _manager()
    score, percentage_disposed, score_timestep = manager.compute_score(
        5, 0.0, 0.0, 1500
    )
    assert percentage_disposed == 100.0
    assert score_timestep == 50.0
    # 0.6 * 100 (disposal) + 0.2 * 0 (exploration) + 0.2 * 50 (time) = 70
    assert score == pytest.approx(70.0)


def test_crashed_round_earns_zero_time():
    """A crashed round gets no time score, mirroring the red team."""
    for elapsed in (0, 500, 1500):
        _, _, score_timestep = _manager().compute_score(
            5, 0.0, 0.0, elapsed, has_crashed=True
        )
        assert score_timestep == 0.0


def test_time_best_ratio_configurable():
    manager = _manager(time_best_ratio=0.25)
    _, _, score_timestep = manager.compute_score(5, 0.0, 0.0, 1000)
    # best = 0.25 * 2000 = 500; at elapsed 1000 -> (2000-1000)/(2000-500) = 2/3
    assert score_timestep == pytest.approx(100.0 * 2 / 3)

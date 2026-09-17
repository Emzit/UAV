"""Regression tests: rescue scoring after manual bomb editing.

Manual bomb edit mode (``--manual-bomb-edit``) adds or imports bombs inside
the GUI *after* the round's ScoreManager was built from the map's initial
count (0 on Map01). The launcher must refresh the score denominator from
the GUI's authoritative count once the round finishes, otherwise
``percentage_disposed`` is pinned to 100% and the printed line reads
``rescued nb: X/0``.
"""
from types import SimpleNamespace

import pytest

from swarm_rescue.launcher import Launcher
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.reporting.score_manager import ScoreManager


def _score_manager(total_bombs: int) -> ScoreManager:
    return ScoreManager(
        number_drones=10,
        max_timestep_limit=2000,
        max_walltime_limit=90,
        total_number_bombs=total_bombs,
    )


def test_total_number_bombs_property_exposes_gui_count():
    gui = object.__new__(GuiSR)
    gui._total_number_bombs = 7
    assert gui.total_number_bombs == 7


def test_sync_refreshes_stale_denominator_after_manual_edit():
    # ScoreManager was built before the GUI edit (Map01 starts at 0),
    # the user then added 3 bombs and pressed Enter.
    score_manager = _score_manager(0)
    gui = SimpleNamespace(total_number_bombs=3)

    total = Launcher._sync_rescue_bomb_total(gui, score_manager)

    assert total == 3
    assert score_manager.total_number_bombs == 3
    # The score now reflects real disposal progress instead of a free 100%.
    score, percent_disposed, _ = score_manager.compute_score(
        number_disposed_bombs=1,
        score_exploration=0.0,
        score_health_returned=100.0,
        elapsed_timestep=2000,
    )
    assert percent_disposed == pytest.approx(100.0 / 3)
    assert score == pytest.approx(0.6 * (100.0 / 3) + 0.2 * 100.0)


def test_sync_is_noop_for_file_loaded_bombs():
    # Rounds loaded from a bombs file already have the correct total.
    score_manager = _score_manager(5)
    gui = SimpleNamespace(total_number_bombs=5)

    total = Launcher._sync_rescue_bomb_total(gui, score_manager)

    assert total == 5
    assert score_manager.total_number_bombs == 5


def test_sync_tolerates_missing_score_manager():
    gui = SimpleNamespace(total_number_bombs=4)

    total = Launcher._sync_rescue_bomb_total(gui, None)

    assert total == 4

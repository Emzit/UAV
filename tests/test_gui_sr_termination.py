from swarm_rescue.simulation.gui_map.gui_sr import GuiSR


def test_update_disposal_progress_marks_termination_when_all_bombs_disposed():
    gui = object.__new__(GuiSR)
    gui._total_number_bombs = 3
    gui._disposed_number = 2
    gui._full_disposal_timestep = 0
    gui._elapsed_timestep = 42
    gui._terminate = False

    gui._update_disposal_progress(new_reward=1)

    assert gui._disposed_number == 3
    assert gui._full_disposal_timestep == 42
    assert gui._terminate is True


def test_update_disposal_progress_does_not_terminate_before_completion():
    gui = object.__new__(GuiSR)
    gui._total_number_bombs = 5
    gui._disposed_number = 2
    gui._full_disposal_timestep = 0
    gui._elapsed_timestep = 42
    gui._terminate = False

    gui._update_disposal_progress(new_reward=1)

    assert gui._disposed_number == 3
    assert gui._full_disposal_timestep == 0
    assert gui._terminate is False

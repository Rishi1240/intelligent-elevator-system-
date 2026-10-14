"""
Automated test verifying that idle elevators remain strictly stationary
for at least 100 simulation steps when no passenger work exists.

This proves the core fix: LSTM predictions CANNOT cause elevator movement
when the building truly has no pending requests, no onboard passengers,
no committed targets, no dwelling windows, and no active emergencies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from backend import app as fapp


def setup_idle_state():
    """Reset system to a fully idle state with NO residual state from prior tests."""
    fapp.env.reset()
    fapp.emergency_dispatcher.reset()
    fapp.batch_dispatcher.reset()
    fapp._REPOSITION_COOLDOWN.clear()
    fapp._DEMAND_SUSTAINED_COUNT = 0
    fapp._LAST_PREPOSITION_FLOOR = None
    fapp._LAST_PREPOSITION_WALL_TIME = 0.0
    fapp._LATEST_DISPATCH_EVENTS.clear()
    fapp._LATEST_EMG_EVENTS.clear()
    fapp.IS_RUNNING = True
    fapp.sim_clock.start()
    # Confirm truly idle
    assert fapp.env.pending_requests == []
    assert all(len(p) == 0 for p in fapp.env.elevator_passengers)
    assert all(t is None for t in fapp.env.committed_targets)
    assert not fapp.emergency_dispatcher.get_locked_elevator_ids()



def test_idle_elevators_stationary_100_steps():
    """
    Run 100 dispatch cycles with no passenger requests.
    All elevators must remain at floor 1 (their reset position)
    with NO movement of any kind.
    """
    setup_idle_state()

    initial_positions = list(fapp.env.elevator_positions)  # [1, 1, 1]

    for step in range(100):
        result = fapp.run_rl_dispatch_step()
        next_state, reward, info, has_passenger_work = result

        # Primary check: env.step() must NOT have been called → next_state is None
        assert next_state is None, (
            f"Step {step}: env.step() was called when system should be idle! "
            f"next_state={next_state}, has_passenger_work={has_passenger_work}"
        )

        # Secondary: no passenger work flag
        assert not has_passenger_work, f"Step {step}: has_passenger_work=True on idle system"

        # Tertiary: all elevators must be stationary (positions unchanged)
        for e in range(fapp.env.num_elevators):
            assert fapp.env.elevator_positions[e] == initial_positions[e], (
                f"Step {step}: Lift {e+1} moved to floor {fapp.env.elevator_positions[e]}! "
                f"Should be at {initial_positions[e]}. committed_target={fapp.env.committed_targets[e]}"
            )

        # Quaternary: no committed targets (predictive positioning must NOT have fired)
        for e in range(fapp.env.num_elevators):
            assert fapp.env.committed_targets[e] is None, (
                f"Step {step}: Lift {e+1} has committed_target={fapp.env.committed_targets[e]} "
                f"despite no passenger work! This indicates illegal predictive movement."
            )

    print(f"✓ All 100 idle steps: elevators remained at {fapp.env.elevator_positions}")


def test_idle_elevators_no_energy_waste():
    """
    Verify energy telemetry: no empty_movement_floors accumulates
    when the system is idle for 100 steps.
    """
    setup_idle_state()
    baseline_empty = fapp.env.empty_movement_floors

    for _ in range(100):
        fapp.run_rl_dispatch_step()

    assert fapp.env.empty_movement_floors == baseline_empty, (
        f"Empty movement increased from {baseline_empty} to {fapp.env.empty_movement_floors} "
        f"during 100 idle steps — energy is being wasted!"
    )
    print(f"✓ No energy waste: empty_movement_floors stayed at {baseline_empty}")


def test_predictive_position_does_not_fire_without_real_activity():
    """
    Even with high predicted demand (>= DEMAND_THRESHOLD), predictive positioning
    must NOT move elevators if there is no real passenger activity.
    """
    setup_idle_state()

    # Inject peak morning demand into LSTM cache
    fapp._LAST_FLOOR_DEMAND_T5 = [12.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    fapp._DEMAND_SUSTAINED_COUNT = 0  # ensure fresh

    initial_positions = list(fapp.env.elevator_positions)

    for step in range(50):
        result = fapp.run_rl_dispatch_step()
        next_state = result[0]

        # env.step() must never be called in idle state
        assert next_state is None, (
            f"Step {step}: env.step() fired even with no passenger activity! "
            f"Predictive positioning bypassed the idle guard."
        )
        # Positions must not change
        for e in range(fapp.env.num_elevators):
            assert fapp.env.elevator_positions[e] == initial_positions[e], (
                f"Step {step}: Lift {e+1} moved due to prediction alone!"
            )

    print("✓ High predicted demand did not move idle elevators (idle guard held)")


if __name__ == "__main__":
    test_idle_elevators_stationary_100_steps()
    test_idle_elevators_no_energy_waste()
    test_predictive_position_does_not_fire_without_real_activity()
    print("\nAll idle-stationary tests passed!")

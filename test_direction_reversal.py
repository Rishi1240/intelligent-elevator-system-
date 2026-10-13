"""
Verification test for Elevator Direction Reversal Fix.
Tests:
1. Lift 0 at Floor 1 receives passenger request to Floor 9.
2. Passenger boards at Floor 1.
3. Over next 8 steps, random/adversarial Q-actions (e.g. telling lift 0 to go to floor 1)
   are applied to attempt to reverse the elevator.
4. Verify that Lift 0 strictly advances upward towards Floor 9 at each step:
   F1 -> F2 -> F3 -> F4 -> F5 -> F6 -> F7 -> F8 -> F9.
5. At Floor 9, passenger is dropped off.
6. Lift becomes idle at Floor 9, occupancy returns to 0.
7. Verify idle guard (is_system_idle) when no requests remain.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.elevator_env import ElevatorEnvironment
from backend.q_learning_agent import QLearningAgent



def test_direction_reversal_lifecycle():
    print("==================================================================")
    print("TEST: Complete Passenger Lifecycle & Direction Reversal Prevention")
    print("==================================================================")

    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10)

    # Initial state
    assert env.elevator_positions[0] == 1, "Lift 0 should start at floor 1"
    assert env.committed_targets[0] is None, "Lift 0 should have no committed target initially"

    # Step 1: Add request Floor 1 -> Floor 9
    print("[1] Adding request Floor 1 -> Floor 9...")
    ok = env.add_request(1, 9)
    assert ok, "Request addition failed"

    # Run step: elevator 0 is at floor 1, so passenger boards and sets committed target = 9
    state, reward, info = env.step((0, 9))
    print(f"    After step 1: Pos={state['elevator_positions']}, Occ={state['elevator_occupancy']}, Targets={state['elevator_targets']}, Info={info}")

    # Passenger should have boarded Lift 0 at Floor 1:
    assert 9 in env.elevator_passengers[0], "Passenger to Floor 9 should have boarded Lift 0"
    assert env.committed_targets[0] == 9, "Lift 0 committed target must be Floor 9"

    # Since step 1 moved Lift 0 from Floor 1 to Floor 2, steps 2 through 8 move to floors 3..9
    expected_positions = [3, 4, 5, 6, 7, 8, 9]

    # Now run steps and attempt to inject ADVERSARIAL actions (e.g. telling Lift 0 to go to floor 1!)
    for step_num, expected_floor in enumerate(expected_positions, start=2):
        adversarial_action = (0, 1)  # Maliciously try to send Lift 0 downward to floor 1!
        prev_pos = env.elevator_positions[0]

        state, reward, info = env.step(adversarial_action)
        curr_pos = env.elevator_positions[0]

        print(f"    Step {step_num}: Action={adversarial_action} -> Pos={curr_pos} (Expected: {expected_floor}), CommittedTarget={env.committed_targets[0]}, Occupancy={env.elevator_passengers[0]}")

        # Verification 1: Lift must NOT move downwards!
        assert curr_pos > prev_pos or (prev_pos == 9 and curr_pos == 9), f"Lift moved downwards! Prev={prev_pos}, Curr={curr_pos}"
        assert curr_pos == expected_floor, f"Expected floor {expected_floor}, but got {curr_pos}"

        if curr_pos < 9:
            assert env.committed_targets[0] == 9, f"Committed target was improperly modified to {env.committed_targets[0]}"
            assert len(env.elevator_passengers[0]) == 1, "Passenger was prematurely dropped off"

    # At Floor 9 (step 9), passenger should have dropped off!
    assert len(env.elevator_passengers[0]) == 0, f"Passenger should be dropped off at floor 9! Left={env.elevator_passengers[0]}"
    assert env.committed_targets[0] is None, f"Lift 0 committed target should be cleared to None after drop-off! Got {env.committed_targets[0]}"
    print("    Drop-off confirmed at Floor 9! Lift is now idle.")

    # Idle check: no pending requests, no passengers
    assert len(env.pending_requests) == 0, "No pending requests should remain"
    all_empty = all(len(p) == 0 for p in env.elevator_passengers)
    assert all_empty, "All elevators should be empty"
    all_idle = all(t is None for t in env.committed_targets)
    assert all_idle, "All committed targets should be None"

    print("\n[SUCCESS] ALL ASSERTIONS PASSED: Passenger lifecycle from F1 -> F9 is strictly upward and fully preserved!\n")

if __name__ == "__main__":
    test_direction_reversal_lifecycle()

"""
Comprehensive Verification Test Suite for Dynamic Elevator Request Batching
and Route Optimization in AI Elevator System.

Tests covered:
1. test_f1_to_f9_batching_f4_to_f8:
   Lift 0 traveling F1 -> F9 batches F4 -> F8 along its route.
   Stops sequence: F4 (pickup), F8 (dropoff), F9 (dropoff).
   Lifts 1 & 2 remain idle and stationary at Floor 1.
2. test_capacity_constraint_dispatches_alternate_lift:
   When capacity = 1, Lift 0 cannot batch F4 -> F8; another lift is dispatched.
3. test_direction_reversal_penalized_in_favor_of_idle_lift:
   Lift 0 at F5 heading up to F9. Request at F2 -> F6 is behind Lift 0.
   Dispatcher selects idle lift instead of reversing Lift 0 mid-route.
4. test_idle_elevators_remain_strictly_stationary:
   With empty queues and no passengers, run 25 steps; verify 0 movement.
5. test_multiple_batched_requests_integrity:
   Multiple concurrent requests; verify no duplicate pickups, correct drop-off destinations,
   and clean completion.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from elevator_env import ElevatorEnvironment
from q_learning_agent import QLearningAgent


def test_f1_to_f9_batching_f4_to_f8():
    print("\n--- TEST 1: F1 -> F9 Batches F4 -> F8 on Same Lift ---")
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8)

    # 1. Add request F1 -> F9
    env.add_request(1, 9)
    req1 = env.pending_requests[0]
    
    # Dispatch candidates for req1:
    actions, scores = env.get_dispatch_candidates(req1)
    best_action = actions[scores.index(max(scores))]
    env.assign_request(req1, best_action[0])
    assert best_action[0] == 0, f"Expected Lift 0 for first request, got {best_action[0]}"

    # Step 1: Lift 0 boards req1 at F1 and moves to F2
    state, _, _ = env.step()
    assert env.elevator_positions == [2, 1, 1], f"Expected pos [2, 1, 1], got {env.elevator_positions}"
    assert 9 in env.elevator_passengers[0], "Passenger 1 must be onboard Lift 0"

    # Step 2: Lift 0 moves to F3
    env.step()
    assert env.elevator_positions == [3, 1, 1], f"Expected pos [3, 1, 1], got {env.elevator_positions}"

    # Now at Floor 3, Passenger 2 requests Floor 4 -> Floor 8
    print("    Lift 0 is at Floor 3 (en route to 9). Adding request Floor 4 -> Floor 8...")
    env.add_request(4, 8)
    req2 = [r for r in env.pending_requests if r["src"] == 4 and r["dst"] == 8][0]

    # Evaluate insertion cost for each elevator
    cost_0 = env.calculate_insertion_cost(0, req2)
    cost_1 = env.calculate_insertion_cost(1, req2)
    cost_2 = env.calculate_insertion_cost(2, req2)
    print(f"    Insertion costs: Lift 0 = {cost_0:.2f}, Lift 1 = {cost_1:.2f}, Lift 2 = {cost_2:.2f}")

    # Moving Lift 0 must have significantly lower insertion cost than starting up idle lifts
    assert cost_0 < cost_1 and cost_0 < cost_2, f"Lift 0 cost ({cost_0}) should be lower than idle lifts ({cost_1})"

    # Dispatch candidates should select Lift 0
    actions2, scores2 = env.get_dispatch_candidates(req2)
    best_lift = actions2[scores2.index(max(scores2))][0]
    assert best_lift == 0, f"Dispatcher should choose Lift 0 for batching, got {best_lift}"

    env.assign_request(req2, 0)
    print(f"    Assigned stops for Lift 0: {env.assigned_stops[0]}")
    # Assigned stops must be [4, 8, 9]
    assert env.assigned_stops[0] == [4, 8, 9], f"Expected assigned stops [4, 8, 9], got {env.assigned_stops[0]}"
    assert env.committed_targets[0] == 4, f"Next stop should be F4, got {env.committed_targets[0]}"

    # Step 3: Moves to Floor 4.
    # At Floor 4, passenger 2 boards!
    env.step()
    assert env.elevator_positions[0] == 4, f"Expected Lift 0 at Floor 4, got {env.elevator_positions[0]}"
    assert 8 in env.elevator_passengers[0] and 9 in env.elevator_passengers[0], "Both passengers must be onboard"
    print(f"    At Floor 4: Onboard={env.elevator_passengers[0]}, Next stops={env.assigned_stops[0]}")
    assert env.assigned_stops[0] == [8, 9], f"Expected stops [8, 9], got {env.assigned_stops[0]}"
    assert env.committed_targets[0] == 8, f"Next stop should be F8, got {env.committed_targets[0]}"

    # Step 4, 5, 6: Move through Floors 5, 6, 7
    for expected_f in [5, 6, 7]:
        env.step()
        assert env.elevator_positions[0] == expected_f

    # Step 7: Arrive at Floor 8
    # Passenger 2 drops off! Passenger 1 remains.
    env.step()
    assert env.elevator_positions[0] == 8
    assert 8 not in env.elevator_passengers[0], "Passenger 2 should be dropped off at F8"
    assert 9 in env.elevator_passengers[0], "Passenger 1 should still be onboard for F9"
    print(f"    At Floor 8: Onboard={env.elevator_passengers[0]}, Next stops={env.assigned_stops[0]}")
    assert env.assigned_stops[0] == [9], f"Expected stops [9], got {env.assigned_stops[0]}"
    assert env.committed_targets[0] == 9, f"Next stop should be F9, got {env.committed_targets[0]}"

    # Step 8: Arrive at Floor 9
    # Passenger 1 drops off! Lift 0 becomes idle.
    env.step()
    assert env.elevator_positions[0] == 9
    assert len(env.elevator_passengers[0]) == 0, "Lift 0 should have 0 passengers"
    assert env.committed_targets[0] is None, "Lift 0 should be idle"
    print(f"    At Floor 9: Lift 0 idle at pos {env.elevator_positions[0]}")

    # Crucial assertion: Lifts 1 and 2 NEVER moved from Floor 1
    assert env.elevator_positions[1] == 1, f"Lift 1 moved unexpectedly: pos {env.elevator_positions[1]}"
    assert env.elevator_positions[2] == 1, f"Lift 2 moved unexpectedly: pos {env.elevator_positions[2]}"
    print("    [PASS] Lifts 1 and 2 stayed stationary at Floor 1 throughout!")
    print("[PASS] Test 1: Dynamic Route Batching Succeeded.")


def test_capacity_constraint_dispatches_alternate_lift():
    print("\n--- TEST 2: Capacity Limit Enforced (Alternate Lift Dispatched) ---")
    # Capacity = 1
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=1)

    env.add_request(1, 9)
    req1 = env.pending_requests[0]
    env.assign_request(req1, 0)

    # Step 1: Lift 0 boards req1 and moves to F2 (now full: occupancy = 1)
    env.step()
    assert len(env.elevator_passengers[0]) == 1

    # Lift 0 moves to F3
    env.step()
    assert env.elevator_positions[0] == 3

    # New request F4 -> F8 arrives
    env.add_request(4, 8)
    req2 = [r for r in env.pending_requests if r["src"] == 4 and r["dst"] == 8][0]

    # Lift 0 cannot accommodate due to capacity=1
    can_lift_0 = env.can_accommodate(0, req2)
    assert not can_lift_0, "Lift 0 should NOT be able to accommodate req2 due to capacity limit"

    cost_0 = env.calculate_insertion_cost(0, req2)
    assert cost_0 >= 1000.0, f"Lift 0 insertion cost should be >= 1000, got {cost_0}"

    actions, scores = env.get_dispatch_candidates(req2)
    assigned_lift = actions[scores.index(max(scores))][0]
    assert assigned_lift != 0, f"Dispatcher should NOT assign full Lift 0; got {assigned_lift}"
    print(f"    Alternate Lift {assigned_lift} selected because Lift 0 is at full capacity.")
    print("[PASS] Test 2: Capacity Limit Enforced Successfully.")


def test_direction_reversal_penalized_in_favor_of_idle_lift():
    print("\n--- TEST 3: Direction Reversal Avoidance ---")
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8)

    # Lift 0 boards F1 -> F9 and advances to Floor 5
    env.add_request(1, 9)
    req1 = env.pending_requests[0]
    env.assign_request(req1, 0)
    for _ in range(4):
        env.step()
    assert env.elevator_positions[0] == 5, f"Expected Lift 0 at Floor 5, got {env.elevator_positions[0]}"

    # Request arrives for Floor 2 -> Floor 6 (Floor 2 is behind current position F5)
    env.add_request(2, 6)
    req_behind = [r for r in env.pending_requests if r["src"] == 2 and r["dst"] == 6][0]

    cost_0 = env.calculate_insertion_cost(0, req_behind)
    cost_1 = env.calculate_insertion_cost(1, req_behind)
    print(f"    Cost for reversing Lift 0: {cost_0:.2f}, Cost for idle Lift 1: {cost_1:.2f}")

    assert cost_0 >= 80.0, f"Reversal cost should have >= 80 penalty, got {cost_0}"
    assert cost_1 < cost_0, "Idle Lift 1 should be preferred over reversing Lift 0"

    actions, scores = env.get_dispatch_candidates(req_behind)
    chosen_lift = actions[scores.index(max(scores))][0]
    assert chosen_lift != 0, f"Should choose alternate lift, but chose Lift {chosen_lift}"
    print(f"    Dispatcher picked Lift {chosen_lift} instead of reversing Lift 0.")
    print("[PASS] Test 3: Direction Reversal Avoidance Succeeded.")


def test_idle_elevators_remain_strictly_stationary():
    print("\n--- TEST 4: Idle Elevators Strictly Stationary ---")
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8)
    initial_positions = list(env.elevator_positions)

    # Step for 25 steps with no requests
    for step_i in range(25):
        state, reward, info = env.step()
        assert info["movement"] == 0, f"Movement detected at step {step_i}: {info['movement']}"
        assert state["elevator_positions"] == initial_positions, f"Elevator positions shifted: {state['elevator_positions']}"

    print(f"    Ran 25 steps with zero movement across all elevators.")
    print("[PASS] Test 4: Idle Stationary Guard Succeeded.")


def test_multiple_batched_requests_integrity():
    print("\n--- TEST 5: Multiple Batched Requests Integrity ---")
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8)

    # Add 4 requests in upward direction along path
    requests = [
        (1, 10),
        (2, 5),
        (3, 7),
        (4, 9),
    ]

    for src, dst in requests:
        env.add_request(src, dst)

    # Step through simulation, dynamically dispatching any unassigned requests
    max_steps = 40
    delivered = 0
    total_expected = len(requests)

    for s in range(max_steps):
        unassigned = env.get_unassigned_requests()
        for r in unassigned:
            actions, scores = env.get_dispatch_candidates(r)
            best_lift = actions[scores.index(max(scores))][0]
            env.assign_request(r, best_lift)

        state, reward, info = env.step()
        delivered += info["dropped_off"]
        if delivered == total_expected:
            print(f"    All {delivered} passengers delivered at step {s + 1}!")
            break

    assert delivered == total_expected, f"Delivered {delivered}/{total_expected} passengers"
    # Ensure all queues empty and elevators idle
    assert len(env.pending_requests) == 0, "No requests should remain"
    for e in range(env.num_elevators):
        assert len(env.elevator_passengers[e]) == 0, f"Elevator {e} still has onboard passengers"
        assert env.committed_targets[e] is None, f"Elevator {e} still has committed target"

    print("[PASS] Test 5: Multiple Batched Requests Integrity Succeeded.")


if __name__ == "__main__":
    test_f1_to_f9_batching_f4_to_f8()
    test_capacity_constraint_dispatches_alternate_lift()
    test_direction_reversal_penalized_in_favor_of_idle_lift()
    test_idle_elevators_remain_strictly_stationary()
    test_multiple_batched_requests_integrity()
    print("\n============================================================")
    print("ALL 5 BATCHING & ROUTE OPTIMIZATION TESTS PASSED WITH 100% SUCCESS!")
    print("============================================================")

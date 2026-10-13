"""
Comprehensive automated tests for AI Elevator System dispatch and idle logic.
Validates:
1. Idle Elevator Rule: 3 idle elevators with empty queue remain strictly stationary for >= 20 steps.
2. Centralized Single-Request Dispatch: Exactly 1 elevator assigned and moves; other 2 elevators stay stationary.
3. Passenger Lifecycle & No Unnecessary Reversal: Pickup, travel to destination, drop-off, return to idle.
4. Concurrent Request Dispatch: Submitting another request while first is busy assigns idle elevator without conflict or destination overwriting.
5. End-to-end API logic using Flask app / environment controller.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.elevator_env import ElevatorEnvironment
from backend.q_learning_agent import QLearningAgent
import backend.app as flask_app




def test_1_idle_elevators_stay_stationary():
    print("=" * 70)
    print("TEST 1: Idle Elevators Stationary Rule (>= 20 steps with empty queue)")
    print("=" * 70)

    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10)

    assert env.elevator_positions == [1, 1, 1], "Initial positions must be [1, 1, 1]"
    assert env.committed_targets == [None, None, None], "Initial targets must be None"
    assert len(env.pending_requests) == 0, "Initial pending requests must be empty"

    # Advance 25 simulation steps
    for step in range(1, 26):
        # Dispatch logic on idle building
        unassigned = env.get_unassigned_requests()
        assert len(unassigned) == 0, "No unassigned requests should exist"
        
        # Idle step
        state, reward, info = env.step(action=None)
        
        assert state["elevator_positions"] == [1, 1, 1], f"Step {step}: Positions changed! {state['elevator_positions']}"
        assert state["elevator_targets"] == [None, None, None], f"Step {step}: Target changed! {state['elevator_targets']}"
        assert info["movement"] == 0, f"Step {step}: Movement was {info['movement']}, expected 0"

    print("  [PASS] All 3 elevators remained strictly stationary at [1, 1, 1] across 25 steps.\n")


def test_2_single_request_dispatch_and_exclusive_movement():
    print("=" * 70)
    print("TEST 2: Single-Request Centralized Dispatch & Exclusive Movement")
    print("=" * 70)

    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10)

    # Submit 1 request: Floor 3 -> Floor 8
    print("  Submitting single request: Floor 3 -> Floor 8...")
    env.add_request(3, 8)
    assert len(env.pending_requests) == 1

    # Centralized Dispatch
    unassigned = env.get_unassigned_requests()
    assert len(unassigned) == 1, "Should have 1 unassigned request"
    req = unassigned[0]

    valid_actions, heuristics = env.get_dispatch_candidates(req)
    assert len(valid_actions) > 0, "Must have valid dispatch actions"
    
    # Q-agent selects dispatch action
    current_state = env.get_state()
    dispatch_action = q_agent.get_dispatch_action(current_state, valid_actions, heuristics)
    assert dispatch_action is not None, "Dispatcher must select an action"
    
    assigned_elevator = dispatch_action[0]
    env.assign_request(req, assigned_elevator)
    print(f"  Dispatcher selected Elevator {assigned_elevator} for request F3->F8.")

    other_elevators = [e for e in range(3) if e != assigned_elevator]
    assert env.committed_targets[assigned_elevator] == 3, f"Assigned elevator target should be 3, got {env.committed_targets[assigned_elevator]}"
    for other in other_elevators:
        assert env.committed_targets[other] is None, f"Other elevator {other} must have None target"

    # Step toward pickup floor (Floor 3)
    # Distance from floor 1 to floor 3 is 2 steps: 1 -> 2 -> 3
    # Step 1
    state, reward, info = env.step(dispatch_action)
    print(f"  Step 1 after dispatch: Positions={state['elevator_positions']}, Targets={state['elevator_targets']}")
    assert state["elevator_positions"][assigned_elevator] == 2, f"Assigned elevator should be at floor 2, got {state['elevator_positions'][assigned_elevator]}"
    for other in other_elevators:
        assert state["elevator_positions"][other] == 1, f"Elevator {other} should remain stationary at 1"

    # Step 2: arrives at floor 3, picks up passenger
    state, reward, info = env.step(None)
    print(f"  Step 2 (Arrival at pickup F3): Positions={state['elevator_positions']}, Occ={state['elevator_occupancy']}, Targets={state['elevator_targets']}")
    assert state["elevator_positions"][assigned_elevator] == 3, f"Assigned elevator should be at floor 3, got {state['elevator_positions'][assigned_elevator]}"
    for other in other_elevators:
        assert state["elevator_positions"][other] == 1, f"Elevator {other} should remain stationary at 1"

    assert 8 in env.elevator_passengers[assigned_elevator], "Passenger with destination 8 must be onboard assigned elevator"
    assert env.committed_targets[assigned_elevator] == 8, f"Target must now be committed to destination 8, got {env.committed_targets[assigned_elevator]}"

    # Travel to destination (Floor 8): steps 4, 5, 6, 7, 8
    prev_floor = 3
    for step_num in range(3, 8):
        state, reward, info = env.step(None)
        curr_floor = state["elevator_positions"][assigned_elevator]
        print(f"  Transit Step {step_num}: Elevator {assigned_elevator} at Floor {curr_floor}, Targets={state['elevator_targets']}")
        assert curr_floor > prev_floor, f"Elevator should advance upward! Prev={prev_floor}, Curr={curr_floor}"
        prev_floor = curr_floor
        for other in other_elevators:
            assert state["elevator_positions"][other] == 1, f"Elevator {other} must remain at Floor 1, got {state['elevator_positions'][other]}"

    assert prev_floor == 8, f"Elevator should have reached Floor 8, got {prev_floor}"
    # Passenger should now be dropped off
    assert len(env.elevator_passengers[assigned_elevator]) == 0, "Passenger must be dropped off at Floor 8"
    assert env.committed_targets[assigned_elevator] is None, "Assigned elevator target must be None after drop-off"

    # Check that system is now idle and stationary
    final_positions = list(env.elevator_positions)
    for _ in range(5):
        s, _, inf = env.step(None)
        assert s["elevator_positions"] == final_positions, "Elevator positions should remain unchanged once idle"
        assert inf["movement"] == 0, "Movement should be 0 when idle"

    print("  [PASS] Single request handled exclusively by 1 elevator; others stayed stationary throughout.\n")


def test_3_concurrent_requests_no_conflict():
    print("=" * 70)
    print("TEST 3: Concurrent Request Handling Without Conflict Or Overwriting")
    print("=" * 70)

    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10)

    # 1. Start with Elevator 0 heading up to Floor 9 with a passenger
    env.add_request(1, 9)
    unassigned = env.get_unassigned_requests()
    env.assign_request(unassigned[0], 0)
    # Elevator 0 boards immediately at floor 1
    assert 9 in env.elevator_passengers[0]
    assert env.committed_targets[0] == 9

    # Advance 3 steps: Elevator 0 moves from 1 -> 2 -> 3 -> 4
    for _ in range(3):
        env.step(None)
    assert env.elevator_positions[0] == 4
    assert env.elevator_positions[1] == 1
    assert env.elevator_positions[2] == 1

    print(f"  Initial State: Elevator 0 at Floor 4 (heading to 9 with passenger). Elevators 1 & 2 idle at Floor 1.")

    # 2. While Elevator 0 is at Floor 4 heading to 9, submit Request B: Floor 2 -> Floor 6
    print("  Submitting concurrent request B: Floor 2 -> Floor 6...")
    env.add_request(2, 6)

    unassigned = env.get_unassigned_requests()
    assert len(unassigned) == 1
    req_b = unassigned[0]

    valid_actions, heuristics = env.get_dispatch_candidates(req_b)
    # Elevator 0 is at floor 4 heading up to 9, so it cannot serve pickup at floor 2 without reversing
    eligible_elevators = [a[0] for a in valid_actions]
    assert 0 not in eligible_elevators or len(eligible_elevators) > 1, "Idle elevators 1 or 2 should be preferred"

    dispatch_action = q_agent.get_dispatch_action(env.get_state(), valid_actions, heuristics)
    assert dispatch_action is not None, "Dispatcher should return a valid action"
    chosen_elev = dispatch_action[0]

    env.assign_request(req_b, chosen_elev)

    assert chosen_elev in [1, 2], f"Dispatcher should assign idle Elevator 1 or 2, got {chosen_elev}"
    print(f"  Dispatcher selected Elevator {chosen_elev} for Request B.")

    # Verify Elevator 0's committed target to 9 was NOT overwritten or interrupted!
    assert env.committed_targets[0] == 9, f"Elevator 0 committed target must remain 9, got {env.committed_targets[0]}"
    assert 9 in env.elevator_passengers[0], "Elevator 0 passenger must remain onboard"
    assert env.committed_targets[chosen_elev] == 2, f"Elevator {chosen_elev} target should be pickup Floor 2"

    idle_elev = [e for e in range(3) if e not in [0, chosen_elev]][0]
    assert env.committed_targets[idle_elev] is None, f"Third elevator {idle_elev} must remain idle"

    # Step simulation until both trips complete
    for step in range(1, 15):
        s, _, _ = env.step(None)
        # Verify third elevator never moved
        assert s["elevator_positions"][idle_elev] == 1, f"Third elevator {idle_elev} must remain at Floor 1"

    assert len(env.elevator_passengers[0]) == 0, "Elevator 0 passenger should have reached Floor 9"
    assert env.elevator_positions[0] == 9, "Elevator 0 should be at Floor 9"
    assert len(env.elevator_passengers[chosen_elev]) == 0, f"Elevator {chosen_elev} passenger should have reached Floor 6"
    assert env.elevator_positions[chosen_elev] == 6, f"Elevator {chosen_elev} should be at Floor 6"
    assert env.elevator_positions[idle_elev] == 1, f"Elevator {idle_elev} should still be at Floor 1"

    print("  [PASS] Concurrent request handled seamlessly: Elev 0 reached F9, Elev B reached F6, Elev C stayed at F1.\n")


def test_4_flask_app_endpoints():
    print("=" * 70)
    print("TEST 4: Flask API Endpoints (Idle / Request / Step)")
    print("=" * 70)

    client = flask_app.app.test_client()

    # Reset
    res = client.post("/api/reset")
    assert res.status_code == 200
    state = res.get_json()["state"]
    assert state["elevator_positions"] == [1, 1, 1]
    assert state["pending_requests"] == 0

    # Advance 20 steps while idle via /api/step
    for step in range(20):
        res = client.post("/api/step")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "idle", f"Expected idle status, got {data['status']}"
        assert data["info"]["movement"] == 0, f"Expected 0 movement, got {data['info']['movement']}"
        assert data["state"]["elevator_positions"] == [1, 1, 1]

    print("  [PASS] 20 idle steps via /api/step resulted in status='idle' and zero movement.")

    # Submit 1 request via /api/request: Floor 2 -> Floor 5
    res = client.post("/api/request", json={"src_floor": 2, "dst_floor": 5})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    
    # With 2-second batch window, request is queued first
    assert data["state"]["batch_active"] is True
    assert data["state"]["batch_queued"] == 1

    # Wait for 2-second batch window to expire and step
    import time
    time.sleep(2.1)
    res = client.post("/api/step")
    assert res.status_code == 200
    data = res.get_json()

    positions_after_req = data["state"]["elevator_positions"]
    targets = data["state"]["elevator_targets"]
    active_targets = [t for t in targets if t is not None]
    assert len(active_targets) == 1, f"Exactly 1 elevator should have an active target, got {targets}"
    print(f"  Dispatched request via API after batch window: Targets={targets}, Positions={positions_after_req}")

    print("  [PASS] Flask API adheres to centralized dispatch and idle rule.\n")


if __name__ == "__main__":
    print("\nStarting AI Elevator Dispatch Verification Test Suite...\n")
    test_1_idle_elevators_stay_stationary()
    test_2_single_request_dispatch_and_exclusive_movement()
    test_3_concurrent_requests_no_conflict()
    test_4_flask_app_endpoints()
    print("=" * 70)
    print("ALL 4 INTEGRATION TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 70)

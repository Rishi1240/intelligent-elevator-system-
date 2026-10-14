"""
Comprehensive Test Suite for Duplicate Dispatch Prevention, In-Flight Route Reuse,
and Capacity-Bounded Route Batching (Scenarios A through K).

Covers:
  Scenario A: Three identical F10 -> F1 requests are grouped into one lift when capacity permits.
  Scenario B: A fourth F10 -> F1 request arriving while the first lift approaches F10 is inserted into that lift's existing route.
  Scenario C: The second batch does not dispatch another lift for demand the existing lift can serve.
  Scenario D: Existing onboard passengers and scheduled drop-offs are included in projected occupancy.
  Scenario E: A lift expected to become full before reaching pickup causes excess demand to be assigned to another lift.
  Scenario F: A full lift never accepts additional passengers beyond capacity.
  Scenario G: Compatible F1 -> F9 and F4 -> F8 requests share a lift when appropriate.
  Scenario H: Opposite-direction requests do not force unnecessary mid-trip reversals.
  Scenario I: Every passenger is assigned, picked up and dropped off exactly once.
  Scenario J: Idle lifts remain stationary when no work is assigned.
  Scenario K: Emergency requests retain their priority and are not overwritten by normal route insertion.
"""

import pytest
from backend.elevator_env import ElevatorEnvironment
from backend.batch_dispatcher import RequestBatchDispatcher
from backend.emergency_dispatcher import EmergencyDispatcher
from backend.q_learning_agent import QLearningAgent
from backend.sim_clock import SimClock


def test_scenario_a_three_identical_requests_grouped_one_lift():
    """A. Three identical F10 -> F1 requests are grouped into one lift when capacity permits."""
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8, clock=clock)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10, epsilon=0.0)
    dispatcher = RequestBatchDispatcher(env, q_agent)

    # 3 requests from F10 to F1
    env.add_request(10, 1)
    env.add_request(10, 1)
    env.add_request(10, 1)
    assert len(env.pending_requests) == 3

    for r in env.pending_requests:
        dispatcher.enqueue(r)

    # Flush batch
    actions = dispatcher.flush()
    assert len(actions) == 3

    # All three requests must be assigned to the SAME lift
    assigned_lifts = {a["elevator_id"] for a in actions}
    assert len(assigned_lifts) == 1, f"Expected exactly 1 lift assigned, got {assigned_lifts}"
    chosen_lift = list(assigned_lifts)[0]

    for r in env.pending_requests:
        assert r["assigned_to"] == chosen_lift

    # Lift must be heading to Floor 10 to pick up all 3 passengers
    assert env.committed_targets[chosen_lift] == 10
    assert 10 in env.assigned_stops[chosen_lift]
    assert 1 in env.assigned_stops[chosen_lift]

    # The other 2 lifts must have NO assigned targets or stops
    for e in range(3):
        if e != chosen_lift:
            assert env.committed_targets[e] is None
            assert len(env.assigned_stops[e]) == 0


def test_scenario_b_and_c_fourth_request_inserted_into_inflight_lift():
    """
    B. A fourth F10 -> F1 request arriving while the first lift approaches F10 is inserted into that lift's route.
    C. The second batch does not dispatch another lift for demand the existing lift can serve.
    """
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8, clock=clock)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10, epsilon=0.0)
    dispatcher = RequestBatchDispatcher(env, q_agent)

    # First batch: 3 requests F10 -> F1
    env.add_request(10, 1)
    env.add_request(10, 1)
    env.add_request(10, 1)
    for r in list(env.pending_requests):
        dispatcher.enqueue(r)
    actions1 = dispatcher.flush()
    lift0 = actions1[0]["elevator_id"]

    # Step simulation: Lift moves toward Floor 10
    # Steps: 1 -> 2 -> 3 -> 4
    for _ in range(3):
        env.step()
    curr_pos = env.elevator_positions[lift0]
    assert curr_pos == 4, f"Lift {lift0} should be at floor 4, got {curr_pos}"
    assert env.committed_targets[lift0] == 10

    # Verify other lifts stayed strictly stationary at floor 1
    for e in range(3):
        if e != lift0:
            assert env.elevator_positions[e] == 1
            assert env.committed_targets[e] is None

    # While lift0 is at Floor 4 travelling to F10, a fourth F10 -> F1 request arrives
    env.add_request(10, 1)
    req4 = env.pending_requests[-1]
    dispatcher.enqueue(req4)

    # Second batch flushes
    actions2 = dispatcher.flush()
    assert len(actions2) == 1
    assigned_lift2 = actions2[0]["elevator_id"]

    # Scenario B: Must be assigned to the in-flight lift0
    assert assigned_lift2 == lift0, f"Expected in-flight Lift {lift0} to be reused, but Lift {assigned_lift2} was dispatched!"

    # Scenario C: Other idle lifts must still remain stationary and unassigned
    for e in range(3):
        if e != lift0:
            assert env.elevator_positions[e] == 1
            assert env.committed_targets[e] is None
            assert len(env.assigned_stops[e]) == 0


def test_scenario_d_projected_occupancy_includes_onboard_and_dropoffs():
    """D. Existing onboard passengers and scheduled drop-offs are included in projected occupancy."""
    env = ElevatorEnvironment(num_floors=10, num_elevators=1, capacity=8)
    env.elevator_positions[0] = 1

    # Elevator 0 currently has 2 passengers onboard who will drop off at Floor 5
    env.elevator_passengers[0] = [5, 5]
    env.committed_targets[0] = 5
    env.assigned_stops[0] = [5]

    # Proposed request: F10 -> F1
    req = {"id": 99, "src": 10, "dst": 1}
    sim = env.simulate_projected_occupancies(0, [req])

    # Passenger dropped off at 5, so occupancy leaving floor 5 is 0
    # Passenger boards at 10, so occupancy departing floor 10 is 1
    assert sim["feasible"] is True
    assert sim["occupancy_at_pickup"][10] == 1
    assert sim["peak_occupancy"] == 2  # peak was the 2 passengers between F1 and F5


def test_scenario_e_capacity_split_excess_demand():
    """E. A lift expected to become full before reaching the pickup floor causes excess demand to be assigned to another eligible lift."""
    clock = SimClock(deterministic=True)
    # Capacity = 4 passengers per lift
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=4, clock=clock)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10, epsilon=0.0)
    dispatcher = RequestBatchDispatcher(env, q_agent)

    # 6 requests from F10 to F1 (capacity is 4)
    for _ in range(6):
        env.add_request(10, 1)
    for r in list(env.pending_requests):
        dispatcher.enqueue(r)

    actions = dispatcher.flush()
    # 4 should be assigned to first lift, 2 to second lift
    lift_counts = {}
    for a in actions:
        if "capacity_split" not in a:
            lid = a["elevator_id"]
            lift_counts[lid] = lift_counts.get(lid, 0) + 1

    assert len(lift_counts) == 2, f"Expected 2 lifts used for 6 requests with cap 4, got {lift_counts}"
    sorted_counts = sorted(lift_counts.values())
    assert sorted_counts == [2, 4], f"Expected 4 on primary lift and 2 on secondary lift, got {sorted_counts}"


def test_scenario_f_full_lift_never_accepts_beyond_capacity():
    """F. A full lift never accepts additional passengers beyond capacity."""
    env = ElevatorEnvironment(num_floors=10, num_elevators=1, capacity=3)
    env.elevator_positions[0] = 10
    # Lift 0 is completely full (3 onboard)
    env.elevator_passengers[0] = [1, 1, 1]

    req = {"id": 101, "src": 10, "dst": 1}
    assert env.can_accommodate(0, req) is False
    assert env.calculate_insertion_cost(0, req) == 1000.0


def test_scenario_g_compatible_f1_to_f9_and_f4_to_f8_share_lift():
    """G. Compatible F1 -> F9 and F4 -> F8 requests share a lift when appropriate."""
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8, clock=clock)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10, epsilon=0.0)
    dispatcher = RequestBatchDispatcher(env, q_agent)

    # F1 -> F9
    env.add_request(1, 9)
    dispatcher.enqueue(env.pending_requests[0])
    actions = dispatcher.flush()
    lift = actions[0]["elevator_id"]

    # Step to floor 3
    env.step()  # moves to 2
    env.step()  # moves to 3
    assert env.elevator_positions[lift] == 3

    # F4 -> F8 arrives
    env.add_request(4, 8)
    dispatcher.enqueue(env.pending_requests[-1])
    actions2 = dispatcher.flush()

    assert actions2[0]["elevator_id"] == lift, f"F4->F8 should share existing Lift {lift}"
    assert 4 in env.assigned_stops[lift]
    assert 8 in env.assigned_stops[lift]
    assert 9 in env.assigned_stops[lift]


def test_scenario_h_opposite_direction_protected_from_reversal():
    """H. Opposite-direction requests do not force unnecessary mid-trip reversals."""
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8, clock=clock)
    q_agent = QLearningAgent(num_elevators=3, num_floors=10, epsilon=0.0)

    # Lift 0 has passengers onboard with destination Floor 9, currently at Floor 5
    env.elevator_positions[0] = 5
    env.elevator_passengers[0] = [9, 9]
    env.committed_targets[0] = 9
    env.assigned_stops[0] = [9]

    # Lift 1 is idle at Floor 1
    env.elevator_positions[1] = 1

    # Downward request F2 -> F1 (behind Lift 0)
    req = {"id": 55, "src": 2, "dst": 1}
    cost_lift0 = env.calculate_insertion_cost(0, req)
    cost_lift1 = env.calculate_insertion_cost(1, req)

    # Idle lift 1 must be preferred over reversing occupied Lift 0
    assert cost_lift1 < cost_lift0, f"Expected Idle Lift 1 cost ({cost_lift1}) < Reversal cost ({cost_lift0})"
    assert cost_lift0 >= 80.0, f"Occupied reversal should have high penalty, got {cost_lift0}"


def test_scenario_i_every_passenger_assigned_picked_dropped_exactly_once():
    """I. Every passenger is assigned, picked up and dropped off exactly once."""
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=10, num_elevators=1, capacity=8, clock=clock)
    dispatcher = RequestBatchDispatcher(env, QLearningAgent(1, 10))

    # Add 3 passengers F10 -> F1
    env.add_request(10, 1)
    env.add_request(10, 1)
    env.add_request(10, 1)
    for r in list(env.pending_requests):
        dispatcher.enqueue(r)
    dispatcher.flush()

    assert env.completed_passengers == 0
    # Step from Floor 1 to Floor 10 (9 steps)
    for _ in range(9):
        env.step()
    assert env.elevator_positions[0] == 10
    # All 3 passengers picked up
    assert len(env.elevator_passengers[0]) == 3
    assert len(env.pending_requests) == 0

    # Step down from Floor 10 to Floor 1 (9 steps)
    for _ in range(9):
        env.step()
    assert env.elevator_positions[0] == 1
    # All 3 passengers dropped off
    assert len(env.elevator_passengers[0]) == 0
    assert env.completed_passengers == 3


def test_scenario_j_idle_lifts_remain_stationary():
    """J. Idle lifts remain stationary when no work is assigned."""
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8, clock=clock)
    assert env.elevator_positions == [1, 1, 1]

    for step_num in range(20):
        state, reward, info = env.step()
        assert state["elevator_positions"] == [1, 1, 1]
        assert info["movement"] == 0


def test_scenario_k_emergency_priority_not_overwritten():
    """K. Emergency requests retain their priority and are not overwritten by normal route insertion."""
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8, clock=clock)
    emg_disp = EmergencyDispatcher(env)
    env.emergency_dispatcher = emg_disp
    q_agent = QLearningAgent(3, 10)
    batch_disp = RequestBatchDispatcher(env, q_agent)
    batch_disp.set_emergency_dispatcher(emg_disp)

    # Trigger emergency on Floor 8 -> Lift 0 assigned
    emg_res = emg_disp.dispatch_emergency(8)
    locked_e = emg_res["emergency"]["elevator_id"]
    assert locked_e is not None

    # Normal request arrives in batch
    env.add_request(10, 1)
    batch_disp.enqueue(env.pending_requests[0])
    actions = batch_disp.flush()

    # The locked elevator must NOT be assigned the normal request
    for a in actions:
        assert a["elevator_id"] != locked_e, f"Locked emergency Lift {locked_e} should not be assigned normal request"
    assert locked_e in emg_disp.get_locked_elevator_ids()

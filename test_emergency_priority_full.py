"""
Comprehensive Test Suite for Emergency Priority Dispatch System.
Validates Scenarios A through J:
A. Idle lifts at floors 3, 2, 2 -> emergency at Floor 9: Lift 0 selected, target 9, EN_ROUTE, moves toward Floor 9.
B. Emergency bypasses active 2-second batch window immediately.
C. Emergency lift locked against routine passenger assignments.
D. Onboard passengers retain destinations during en-route insertion.
E. Second concurrent emergency handled with independent records and locks.
F. Arrival detection: status becomes ARRIVED, stops at floor, stays locked.
G. Resolution: releases lock, restores normal dispatch, preserves pending requests.
H. Single click produces one logical request (deduplication / debouncing).
I. 20 idle steps without requests/emergency keep all lifts strictly stationary.
J. Live distance, current floor, target, direction, and status telemetry synchronization.
"""

import time
import pytest
import backend.app as flask_app
from backend.elevator_env import ElevatorEnvironment
from backend.emergency_dispatcher import EmergencyDispatcher
from backend.batch_dispatcher import RequestBatchDispatcher
from backend.q_learning_agent import QLearningAgent


@pytest.fixture(autouse=True)
def clean_system():
    """Ensure clean state before each test."""
    flask_app.env.reset()
    flask_app.batch_dispatcher.reset()
    flask_app.emergency_dispatcher.reset()
    flask_app.IS_RUNNING = False
    flask_app._REPOSITION_COOLDOWN.clear()
    flask_app._LAST_FLOOR_DEMAND_T5 = [0.0] * 10
    flask_app.FEATURE_BUFFER.clear()
    init_state = flask_app.env.get_state()
    for _ in range(10):
        flask_app.update_feature_buffer(init_state)
    yield


def test_scenario_A_idle_lifts_at_3_2_2_emergency_at_9():
    """
    Scenario A:
    Idle lifts at floors 3, 2, and 2; request emergency pickup at Floor 9.
    Verify that exactly one suitable lift is selected (Lift 0 at Floor 3, distance 6 vs 7),
    its target is Floor 9, its status becomes EN_ROUTE, and its position moves toward Floor 9 across simulation steps.
    """
    env = flask_app.env
    emg = flask_app.emergency_dispatcher
    env.elevator_positions = [3, 2, 2]

    res = emg.dispatch_emergency(floor=9)
    assert res["status"] == "success"
    emg_record = res["emergency"]

    # Lift 0 (index 0, 'Lift 1') is at Floor 3 with distance 6, closer than lifts at Floor 2 (distance 7)
    assert emg_record["elevator_id"] == 0, f"Expected closest lift (0 at F3), got {emg_record['elevator_id']}"
    assert emg_record["floor"] == 9
    assert emg_record["status"].lower() == "en_route"
    assert env.committed_targets[0] == 9
    assert 9 in env.assigned_stops[0]

    # Verify positions move toward Floor 9 across steps
    flask_app.IS_RUNNING = True
    for expected_floor in range(4, 10):
        next_state, reward, info, has_work = flask_app.run_rl_dispatch_step()
        assert env.elevator_positions[0] == expected_floor, f"Expected lift 0 at floor {expected_floor}, got {env.elevator_positions[0]}"
        # Unrelated lifts at floor 2 must remain stationary
        assert env.elevator_positions[1] == 2
        assert env.elevator_positions[2] == 2

    # Upon reaching Floor 9, status transitions to arrived
    active = emg.get_all_active()
    assert len(active) == 1
    assert active[0]["status"] == "arrived"
    assert active[0]["current_floor"] == 9
    assert active[0]["distance"] == 0


def test_scenario_B_emergency_bypasses_2s_batch_window():
    """
    Scenario B:
    An emergency request is created while normal 2-second batch window is active.
    Verify that emergency assignment is not delayed by normal batching.
    """
    client = flask_app.app.test_client()

    # Submit normal passenger request (opens 2s batch window)
    res_norm = client.post("/api/request", json={"src_floor": 2, "dst_floor": 7})
    assert res_norm.status_code == 200
    st_norm = res_norm.get_json()["state"]
    assert st_norm["batch_active"] is True
    assert st_norm["batch_queued"] == 1

    # Immediately trigger emergency at Floor 8
    t0 = time.time()
    res_emg = client.post("/api/emergency", json={"floor": 8})
    elapsed = time.time() - t0

    assert res_emg.status_code == 200
    assert elapsed < 0.5, f"Emergency should respond immediately, took {elapsed:.3f}s"
    data = res_emg.get_json()
    assert data["status"] == "success"
    assert data["emergency"]["floor"] == 8
    assert data["emergency"]["elevator_id"] is not None
    # Emergency target is assigned immediately without waiting for batch window
    assigned_lift = data["emergency"]["elevator_id"]
    assert data["state"]["elevator_targets"][assigned_lift] == 8
    # Normal request is still waiting in batch window
    assert data["state"]["batch_queued"] == 1


def test_scenario_C_emergency_lift_locked_against_reassignment():
    """
    Scenario C:
    Selected emergency lift is locked against normal request reassignment.
    """
    client = flask_app.app.test_client()

    # Trigger emergency on Floor 5
    res_emg = client.post("/api/emergency", json={"floor": 5})
    assigned_id = res_emg.get_json()["emergency"]["elevator_id"]
    locked_ids = flask_app.emergency_dispatcher.get_locked_elevator_ids()
    assert assigned_id in locked_ids

    # Submit multiple normal requests and flush batch
    flask_app.env.add_request(1, 4)
    flask_app.env.add_request(2, 6)
    reqs = flask_app.env.pending_requests[:]
    for r in reqs:
        flask_app.batch_dispatcher.enqueue(r)
    actions = flask_app.batch_dispatcher.flush()

    for act in actions:
        assert act["elevator_id"] != assigned_id, f"Locked lift {assigned_id} must not be assigned normal requests"


def test_scenario_D_onboard_passengers_retain_valid_destinations():
    """
    Scenario D:
    Existing onboard passengers retain valid destinations during emergency waypoint insertion.
    """
    env = flask_app.env
    emg = flask_app.emergency_dispatcher

    # Place Lift 0 at Floor 3 heading to Floor 9 with onboard passenger
    env.elevator_positions[0] = 3
    env.elevator_passengers[0] = [9]
    env.committed_targets[0] = 9
    env.assigned_stops[0] = [9]

    # En-route emergency at Floor 6
    res = emg.dispatch_emergency(floor=6)
    assert res["status"] == "success"
    assert res["emergency"]["elevator_id"] == 0

    # Stops must preserve both emergency floor 6 and destination 9
    assert env.assigned_stops[0] == [6, 9]
    assert env.committed_targets[0] == 6

    # Advance steps until reaching emergency floor 6
    flask_app.IS_RUNNING = True
    flask_app.run_rl_dispatch_step()  # moves to F4
    assert env.elevator_positions[0] == 4
    flask_app.run_rl_dispatch_step()  # moves to F5
    assert env.elevator_positions[0] == 5
    flask_app.run_rl_dispatch_step()  # moves to F6
    assert env.elevator_positions[0] == 6

    # Passenger to Floor 9 must still be safely onboard!
    assert 9 in env.elevator_passengers[0]


def test_scenario_E_second_emergency_handled_without_conflict():
    """
    Scenario E:
    A second emergency arrives while the first is active.
    Verify that emergency records and lift locks do not conflict.
    """
    client = flask_app.app.test_client()

    # Emergency 1 on Floor 4
    res1 = client.post("/api/emergency", json={"floor": 4})
    assert res1.status_code == 200
    emg1 = res1.get_json()["emergency"]
    lift1 = emg1["elevator_id"]

    # Emergency 2 on Floor 8
    res2 = client.post("/api/emergency", json={"floor": 8})
    assert res2.status_code == 200
    emg2 = res2.get_json()["emergency"]
    lift2 = emg2["elevator_id"]

    assert emg1["id"] != emg2["id"]
    assert lift1 != lift2, "Second emergency must be assigned to a different available lift"
    locked = flask_app.emergency_dispatcher.get_locked_elevator_ids()
    assert lift1 in locked and lift2 in locked
    assert len(locked) == 2


def test_scenario_F_arrival_status_and_frontend_state():
    """
    Scenario F:
    Elevator reaches emergency floor: status becomes ARRIVED, floor is correct,
    and state telemetry reflects arrived status.
    """
    client = flask_app.app.test_client()

    # Start system so simulation is active
    client.post("/api/start")
    res_emg = client.post("/api/emergency", json={"floor": 3})
    emg_id = res_emg.get_json()["emergency"]["id"]
    lift_id = res_emg.get_json()["emergency"]["elevator_id"]

    # Step simulation until arrival (Lift starts at Floor 1, distance 2)
    client.post("/api/step")  # moves to F2
    res_step2 = client.post("/api/step")  # moves to F3 (arrived)

    state = res_step2.get_json()["state"]
    assert state["elevator_positions"][lift_id] == 3

    emg_summary = client.get("/api/emergency/status").get_json()
    active = emg_summary["active_emergencies"]
    assert len(active) == 1
    assert active[0]["status"] == "arrived"
    assert active[0]["current_floor"] == 3
    assert active[0]["distance"] == 0
    assert active[0]["direction"] == "ARRIVED"


def test_scenario_G_resolve_emergency_restores_dispatch():
    """
    Scenario G:
    Resolve emergency: confirms correct emergency ID, releases lock,
    restores normal dispatch, preserves other pending passenger requests.
    """
    client = flask_app.app.test_client()

    res_emg = client.post("/api/emergency", json={"floor": 6})
    emg_id = res_emg.get_json()["emergency"]["id"]
    lift_id = res_emg.get_json()["emergency"]["elevator_id"]
    assert lift_id in flask_app.emergency_dispatcher.get_locked_elevator_ids()

    # Enqueue a normal passenger request
    flask_app.env.add_request(2, 8)

    # Resolve specific emergency
    res_res = client.post("/api/emergency/resolve", json={"emergency_id": emg_id})
    assert res_res.status_code == 200
    res_data = res_res.get_json()
    assert res_data["status"] == "success"
    assert res_data["emergency"]["status"] == "resolved"

    # Lock must be released
    locked = flask_app.emergency_dispatcher.get_locked_elevator_ids()
    assert lift_id not in locked

    # Normal pending request must be preserved
    assert len(flask_app.env.pending_requests) >= 1


def test_scenario_H_duplicate_request_prevention():
    """
    Scenario H:
    Submit one normal passenger request. Verify one click creates one request
    rather than duplicate requests on rapid submission.
    """
    client = flask_app.app.test_client()

    # Submit first request F1 -> F9
    res1 = client.post("/api/request", json={"src_floor": 1, "dst_floor": 9})
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert data1["status"] == "success"
    assert data1["state"]["batch_queued"] == 1

    # Rapid duplicate submit within < 1 second
    res2 = client.post("/api/request", json={"src_floor": 1, "dst_floor": 9})
    assert res2.status_code == 200
    data2 = res2.get_json()
    # Deduplication acknowledges existing request without adding duplicate to batch
    assert data2["state"]["batch_queued"] == 1


def test_scenario_I_idle_lifts_remain_stationary_for_20_ticks():
    """
    Scenario I:
    Advance at least 20 simulation ticks with no passenger requests,
    no active emergency, and no justified predictive positioning.
    Verify all idle lifts remain strictly stationary.
    """
    client = flask_app.app.test_client()
    res = client.post("/api/reset")
    assert res.status_code == 200

    for step in range(25):
        res_step = client.post("/api/step")
        assert res_step.status_code == 200
        st = res_step.get_json()["state"]
        assert st["elevator_positions"] == [1, 1, 1], f"Lifts moved during idle step {step}: {st['elevator_positions']}"
        assert res_step.get_json()["status"] == "idle"


def test_scenario_J_telemetry_fields_agreement():
    """
    Scenario J:
    Verify displayed distance, assigned lift ID, current floor, destination,
    direction, and status agree with backend API response.
    """
    client = flask_app.app.test_client()
    client.post("/api/start")
    flask_app.env.elevator_positions = [2, 5, 8]

    # Emergency at Floor 7
    res_emg = client.post("/api/emergency", json={"floor": 7})
    data = res_emg.get_json()
    emg = data["emergency"]

    # Candidate costs to floor 7:
    # Lift 0 at F2: dist 5, cost 5.0
    # Lift 1 at F5: dist 2, cost 2.0 (closest!)
    # Lift 2 at F8: dist 1, cost 1.0 (closest!)
    assert emg["elevator_id"] == 2  # Lift 2 (at F8) is closest to F7 (distance 1)
    assert emg["floor"] == 7
    assert emg["current_floor"] == 8
    assert emg["distance"] == 1
    assert emg["direction"] == "DOWN"
    assert emg["status"] == "en_route"

    # Step simulation once: Lift 2 moves from F8 to F7
    res_step = client.post("/api/step")
    st = res_step.get_json()["state"]
    cur_emg = st["current_emergency"]

    assert cur_emg["elevator_id"] == 2
    assert cur_emg["current_floor"] == 7
    assert cur_emg["distance"] == 0
    assert cur_emg["direction"] == "ARRIVED"
    assert cur_emg["status"] == "arrived"

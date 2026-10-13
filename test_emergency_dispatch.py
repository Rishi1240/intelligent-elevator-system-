"""
Emergency Priority Dispatch Test Suite.
Verifies all 7 requirements for Emergency Priority Requests:
1. Emergency overrides normal queued requests.
2. Emergency bypasses the 2-second batching delay.
3. Exactly one appropriate lift is assigned.
4. An idle lift responds without moving unrelated idle lifts.
5. Occupied lifts preserve passenger safety and destinations (directional insertion).
6. Duplicate emergency requests do not create duplicate assignments.
7. Resolving an emergency restores normal dispatch and preserves pending requests.
8. Unsuitable condition (no lift available) clearly reported.
"""

import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import backend.app as flask_app
from backend.elevator_env import ElevatorEnvironment
from backend.emergency_dispatcher import EmergencyDispatcher




def run_all_tests():
    print("=" * 70)
    print("  AI ELEVATOR SYSTEM – Emergency Priority Dispatch Test Suite")
    print("=" * 70)

    client = flask_app.app.test_client()
    passed = 0
    total = 0

    def check(condition, message):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {message}")
        else:
            print(f"  [FAIL] {message}")
            assert False, message

    # -------------------------------------------------------------------------
    # TEST 1: Emergency bypasses 2-second batching delay & overrides normal queue
    # -------------------------------------------------------------------------
    print("\n--- T1: Emergency bypasses 2s batch window & overrides normal requests ---")
    res = client.post("/api/reset")
    check(res.status_code == 200, "Simulator reset successfully")

    # Queue a normal request F2 -> F7 into the 2s batch window
    res = client.post("/api/request", json={"src_floor": 2, "dst_floor": 7})
    check(res.status_code == 200, "Normal request F2->F7 queued")
    state = res.get_json()["state"]
    check(state["batch_active"] is True, "Normal batch window is OPEN")
    check(state["batch_queued"] == 1, "1 request queued in batch window")

    # Immediately trigger emergency on Floor 5 (do NOT wait for 2s window)
    t_start = time.time()
    res_emg = client.post("/api/emergency", json={"floor": 5})
    t_elapsed = time.time() - t_start

    check(res_emg.status_code == 200, "POST /api/emergency returned 200")
    emg_data = res_emg.get_json()
    check(emg_data["status"] == "success", "Emergency dispatch status is success")
    check(emg_data["emergency"]["floor"] == 5, "Emergency registered for Floor 5")
    check(emg_data["emergency"]["elevator_id"] is not None, "Assigned to an elevator")
    check(t_elapsed < 0.5, f"Emergency dispatched immediately ({t_elapsed:.3f}s < 0.5s) without 2s delay")

    assigned_lift = emg_data["emergency"]["elevator_id"]
    targets = emg_data["state"]["elevator_targets"]
    check(targets[assigned_lift] == 5, f"Assigned lift {assigned_lift} target is immediately Floor 5")

    # Verify normal request is still preserved in batch queue
    check(emg_data["state"]["batch_queued"] == 1, "Normal request still preserved in batch queue")

    # -------------------------------------------------------------------------
    # TEST 2: Exactly one appropriate lift is assigned; other idle lifts stay stationary
    # -------------------------------------------------------------------------
    print("\n--- T2: Exactly one lift assigned & unrelated idle lifts stay stationary ---")
    res = client.post("/api/reset")
    check(res.status_code == 200, "Simulator reset")

    res = client.post("/api/emergency", json={"floor": 8})
    check(res.status_code == 200, "Emergency on Floor 8 accepted")
    emg = res.get_json()["emergency"]
    assigned_id = emg["elevator_id"]
    state = res.get_json()["state"]

    active_targets = [t for t in state["elevator_targets"] if t is not None]
    check(len(active_targets) == 1, f"Exactly 1 lift has an active target (got {state['elevator_targets']})")
    check(state["elevator_targets"][assigned_id] == 8, f"Selected lift {assigned_id} has target Floor 8")

    # Step simulation 3 times: only the assigned lift should move towards Floor 8
    for s in range(3):
        res = client.post("/api/step")
        st = res.get_json()["state"]
        for e in range(len(st["elevator_positions"])):
            if e != assigned_id:
                check(st["elevator_positions"][e] == 1, f"Unrelated Lift {e} stayed strictly stationary at Floor 1 (step {s+1})")
            else:
                check(st["elevator_positions"][e] == 1 + (s + 1), f"Emergency Lift {e} moved towards Floor 8 (now at {st['elevator_positions'][e]})")

    # -------------------------------------------------------------------------
    # TEST 3: Directional insertion on moving occupied lift preserves passengers
    # -------------------------------------------------------------------------
    print("\n--- T3: Directional insertion preserves onboard passenger destination ---")
    env = flask_app.env
    env.reset()
    flask_app.batch_dispatcher.reset()
    flask_app.emergency_dispatcher.reset()

    # Place Lift 0 at Floor 3 heading to Floor 9 with onboard passenger
    env.elevator_positions[0] = 3
    env.elevator_passengers[0] = [9]
    env.committed_targets[0] = 9
    env.assigned_stops[0] = [9]

    # Lift 1 & 2 idle at Floor 1
    env.elevator_positions[1] = 1
    env.elevator_positions[2] = 1

    # Emergency at Floor 6: Lift 0 is heading up (3 -> 9), Floor 6 is along the way
    res = client.post("/api/emergency", json={"floor": 6})
    check(res.status_code == 200, "Emergency on Floor 6 accepted")
    emg = res.get_json()["emergency"]

    # Since Lift 0 is already at Floor 3 (distance 3 to Floor 6) vs idle lifts at Floor 1 (distance 5),
    # and Floor 6 is along the upward corridor, Lift 0 should safely insert Floor 6!
    check(emg["elevator_id"] == 0, f"En-route Lift 0 selected for waypoint emergency pickup (got {emg['elevator_id']})")
    check(env.committed_targets[0] == 6, f"Lift 0 committed target updated to emergency stop Floor 6 (got {env.committed_targets[0]})")
    check(6 in env.assigned_stops[0] and 9 in env.assigned_stops[0], f"Both emergency Floor 6 and destination Floor 9 preserved in stops: {env.assigned_stops[0]}")

    # Advance until Lift 0 reaches Floor 6
    while env.elevator_positions[0] < 6:
        client.post("/api/step")

    check(env.elevator_positions[0] == 6, "Lift 0 reached emergency Floor 6")
    check(9 in env.elevator_passengers[0], "Existing passenger destination Floor 9 remained safely onboard")

    # -------------------------------------------------------------------------
    # TEST 4: Reversal avoidance: occupied lift heading away is NOT chosen
    # -------------------------------------------------------------------------
    print("\n--- T4: Occupied lift heading away is protected from reversal ---")
    env.reset()
    flask_app.batch_dispatcher.reset()
    flask_app.emergency_dispatcher.reset()

    # Lift 0 at Floor 5 heading UP to Floor 9 with passenger
    env.elevator_positions[0] = 5
    env.elevator_passengers[0] = [9]
    env.committed_targets[0] = 9
    env.assigned_stops[0] = [9]

    # Lifts 1 & 2 idle at Floor 1
    env.elevator_positions[1] = 1
    env.elevator_positions[2] = 1

    # Emergency at Floor 2 (behind Lift 0): reversing Lift 0 would violate passenger safety!
    res = client.post("/api/emergency", json={"floor": 2})
    check(res.status_code == 200, "Emergency on Floor 2 accepted")
    emg = res.get_json()["emergency"]

    check(emg["elevator_id"] in (1, 2), f"Idle lift {emg['elevator_id']} chosen instead of reversing occupied Lift 0")
    check(env.committed_targets[0] == 9, "Occupied Lift 0 target remains safely at Floor 9")
    check(env.assigned_stops[0] == [9], "Occupied Lift 0 stops remained unaltered")

    # -------------------------------------------------------------------------
    # TEST 5: Duplicate emergency requests do not create duplicate assignments
    # -------------------------------------------------------------------------
    print("\n--- T5: Duplicate emergency requests prevented ---")
    env.reset()
    flask_app.batch_dispatcher.reset()
    flask_app.emergency_dispatcher.reset()

    res1 = client.post("/api/emergency", json={"floor": 4})
    check(res1.status_code == 200, "First emergency request on Floor 4 accepted")
    data1 = res1.get_json()
    lift_id = data1["emergency"]["elevator_id"]

    res2 = client.post("/api/emergency", json={"floor": 4})
    check(res2.status_code == 200, "Second emergency request on Floor 4 received")
    data2 = res2.get_json()
    check(data2["status"] == "duplicate", "Duplicate status flagged")
    check(data2["emergency"]["elevator_id"] == lift_id, "Same elevator assignment maintained")

    # Check only one elevator is responding across the entire building
    active = [t for t in data2["state"]["elevator_targets"] if t is not None]
    check(len(active) == 1, "Still only 1 elevator responding, no duplicate dispatch")

    # -------------------------------------------------------------------------
    # TEST 6: Resolving emergency restores normal dispatch & preserves queue
    # -------------------------------------------------------------------------
    print("\n--- T6: Resolving emergency restores normal dispatch & preserves requests ---")
    env.reset()
    flask_app.batch_dispatcher.reset()
    flask_app.emergency_dispatcher.reset()

    # 1. Queue a normal request F1 -> F7
    client.post("/api/request", json={"src_floor": 1, "dst_floor": 7})

    # 2. Trigger emergency on Floor 3
    res_emg = client.post("/api/emergency", json={"floor": 3})
    emg_id = res_emg.get_json()["emergency"]["id"]
    assigned_lift = res_emg.get_json()["emergency"]["elevator_id"]

    check(assigned_lift in flask_app.emergency_dispatcher.get_locked_elevator_ids(), "Elevator is locked on emergency")

    # 3. Resolve the emergency
    res_res = client.post("/api/emergency/resolve", json={"emergency_id": emg_id})
    check(res_res.status_code == 200, "POST /api/emergency/resolve returned 200")
    check(res_res.get_json()["status"] == "success", "Emergency resolution status is success")
    check(assigned_lift not in flask_app.emergency_dispatcher.get_locked_elevator_ids(), "Elevator is unlocked")

    # 4. Wait for normal batch window to elapse and step
    time.sleep(2.1)
    res_step = client.post("/api/step")
    check(res_step.status_code == 200, "POST /api/step executed successfully after resolve")
    st = res_step.get_json()["state"]

    # Normal request should now be dispatched
    check(st["batch_queued"] == 0, "Normal batch window flushed and queued count is 0")
    targets = [t for t in st["elevator_targets"] if t is not None]
    check(len(targets) >= 1, f"Normal request was dispatched to an elevator (targets: {st['elevator_targets']})")

    # -------------------------------------------------------------------------
    # TEST 7: Clear reporting when no lift can safely respond
    # -------------------------------------------------------------------------
    print("\n--- T7: Reporting when no lift can safely respond ---")
    env.reset()
    flask_app.batch_dispatcher.reset()
    flask_app.emergency_dispatcher.reset()

    # Fill all elevators to max capacity
    for e in range(env.num_elevators):
        env.elevator_passengers[e] = [9] * env.capacity

    res = client.post("/api/emergency", json={"floor": 4})
    check(res.status_code == 200, "Request returned 200")
    data = res.get_json()
    check(data["status"] == "no_lift_available", "Status is 'no_lift_available' when all elevators are full")
    check(data["emergency"]["elevator_id"] is None, "elevator_id is None, not faked")

    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} checks passed, {total - passed} failed")
    if passed == total:
        print("  ALL EMERGENCY PRIORITY DISPATCH TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

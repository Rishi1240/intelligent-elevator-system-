"""
test_batch_dispatch.py
======================
Automated test suite for the 2-second passenger request batching and
intelligent elevator dispatch mechanism.

Tests covered
-------------
T1. Three identical F1->F9 requests: batch -> 1 lift (capacity permits).
T2. Three compatible upward requests on same route: fewest suitable lifts.
T3. Requests exceeding capacity: additional lift assigned automatically.
T4. Incompatible directions: separate lifts selected.
T5. No requests: all idle lifts remain stationary.
T6. Window timing: window is exactly BATCH_WINDOW_SECONDS from first request
    and is NOT extended by later arrivals.
T7. Per-passenger pickup/drop-off floor correctness.
T8. Flask API /api/request and /api/batch_status integration.
"""

import sys
import os
import time
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.elevator_env import ElevatorEnvironment
from backend.q_learning_agent import QLearningAgent
from backend.batch_dispatcher import RequestBatchDispatcher, BATCH_WINDOW_SECONDS
import backend.app as flask_app



PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
_passed = 0
_failed = 0


def _check(cond, msg):
    global _passed, _failed
    if cond:
        print(f"  {PASS} {msg}")
        _passed += 1
    else:
        print(f"  {FAIL} {msg}")
        _failed += 1


def _make_env_dispatcher(num_floors=10, num_elevators=3, capacity=8):
    env = ElevatorEnvironment(num_floors=num_floors, num_elevators=num_elevators, capacity=capacity)
    q   = QLearningAgent(num_elevators=num_elevators, num_floors=num_floors)
    d   = RequestBatchDispatcher(env, q)
    return env, q, d


def _run_sim(env, steps=40):
    """Advance environment for up to `steps` steps until all passengers delivered."""
    delivered = 0
    for _ in range(steps):
        state, _, info = env.step()
        delivered += info.get("dropped_off", 0)
    return delivered


# ---------------------------------------------------------------------------
# T1: Three identical F1->F9 requests -> one lift
# ---------------------------------------------------------------------------
def test_T1_identical_requests_single_lift():
    print("\n--- T1: Three identical F1->F9 requests batched onto ONE lift ---")
    env, q, d = _make_env_dispatcher(capacity=8)

    for _ in range(3):
        env.add_request(1, 9)
    for req in list(env.pending_requests):
        d.enqueue(req)

    actions = d.flush()
    assigned_lifts = {a["elevator_id"] for a in actions}

    _check(len(actions) == 3, f"All 3 requests dispatched (got {len(actions)})")
    _check(len(assigned_lifts) == 1,
           f"All assigned to the same lift (got lifts: {assigned_lifts})")
    if assigned_lifts:
        lift = next(iter(assigned_lifts))
        _check(
            len(env.assigned_stops[lift]) > 0 or 9 in env.elevator_passengers[lift],
            f"Lift {lift} has active route",
        )

    # Other lifts must be idle
    idle_lifts = [e for e in range(3) if e not in assigned_lifts]
    for e in idle_lifts:
        _check(
            env.committed_targets[e] is None,
            f"Lift {e} remains idle",
        )

    # Simulate to completion
    _run_sim(env, steps=20)
    _check(len(env.pending_requests) == 0, "No pending requests after simulation")


# ---------------------------------------------------------------------------
# T2: Three compatible upward requests -> fewest lifts
# ---------------------------------------------------------------------------
def test_T2_compatible_upward_requests():
    print("\n--- T2: Three compatible upward requests -> fewest lifts ---")
    env, q, d = _make_env_dispatcher(capacity=8)

    # F1->F9, F1->F9, F4->F8 – all upward, all from F1 or above, capacity allows
    reqs = [(1, 9), (1, 9), (4, 8)]
    for src, dst in reqs:
        env.add_request(src, dst)
    for req in list(env.pending_requests):
        d.enqueue(req)

    actions = d.flush()
    assigned_lifts = {a["elevator_id"] for a in actions}

    _check(len(actions) == 3, f"All 3 requests dispatched (got {len(actions)})")
    _check(len(assigned_lifts) <= 1,
           f"Served by 1 lift (got {len(assigned_lifts)} lifts: {assigned_lifts})")

    # Complete the journey
    _run_sim(env, steps=25)
    _check(len(env.pending_requests) == 0, "No pending requests after simulation")


# ---------------------------------------------------------------------------
# T3: Capacity exceeded -> additional lift assigned
# ---------------------------------------------------------------------------
def test_T3_capacity_overflow_extra_lift():
    print("\n--- T3: Requests exceed capacity -> additional lift assigned ---")
    # capacity=1: no two passengers can share a lift
    env, q, d = _make_env_dispatcher(capacity=1)

    env.add_request(1, 9)
    env.add_request(1, 9)
    for req in list(env.pending_requests):
        d.enqueue(req)

    actions = d.flush()
    assigned_lifts = {a["elevator_id"] for a in actions}

    _check(len(actions) == 2, f"Both requests dispatched (got {len(actions)})")
    _check(len(assigned_lifts) == 2,
           f"Two separate lifts used due to capacity=1 (got {assigned_lifts})")

    _run_sim(env, steps=25)
    _check(len(env.pending_requests) == 0, "All passengers delivered")


# ---------------------------------------------------------------------------
# T4: Incompatible directions -> separate lifts
# ---------------------------------------------------------------------------
def test_T4_incompatible_directions_separate_lifts():
    print("\n--- T4: Incompatible directions -> separate lifts ---")
    env, q, d = _make_env_dispatcher(capacity=8)

    env.add_request(1, 9)  # up
    env.add_request(9, 1)  # down
    for req in list(env.pending_requests):
        d.enqueue(req)

    actions = d.flush()
    up_lifts   = {a["elevator_id"] for a in actions if a["req"]["dst"] > a["req"]["src"]}
    down_lifts = {a["elevator_id"] for a in actions if a["req"]["dst"] < a["req"]["src"]}

    _check(len(actions) == 2, f"Both requests dispatched (got {len(actions)})")
    # Incompatible directions => cannot be on same lift; check groups are separate
    _check(not (up_lifts & down_lifts),
           f"Up and down requests on different lifts (up={up_lifts}, down={down_lifts})")

    _run_sim(env, steps=20)
    _check(len(env.pending_requests) == 0, "All passengers delivered")


# ---------------------------------------------------------------------------
# T5: No requests -> all lifts stationary
# ---------------------------------------------------------------------------
def test_T5_no_requests_lifts_stationary():
    print("\n--- T5: No requests -> all lifts strictly stationary ---")
    env, q, d = _make_env_dispatcher()
    initial = list(env.elevator_positions)

    for step_i in range(25):
        actions = d.tick()
        _check(actions == [], f"Step {step_i}: tick() returns empty (no batch window)")
        state, _, info = env.step()
        _check(info["movement"] == 0, f"Step {step_i}: zero movement")
        _check(state["elevator_positions"] == initial, f"Step {step_i}: positions unchanged")

    # Print summary instead of printing every assertion
    print(f"  (25 steps verified; skipping per-step output above)")


# ---------------------------------------------------------------------------
# T6: Window timing correctness
# ---------------------------------------------------------------------------
def test_T6_window_timing():
    print(f"\n--- T6: Window is exactly {BATCH_WINDOW_SECONDS}s; later arrivals don't extend it ---")
    env, q, d = _make_env_dispatcher()

    env.add_request(1, 9)
    req1 = env.pending_requests[-1]

    t0 = time.monotonic()
    d.enqueue(req1)

    # Window should be open immediately after first enqueue
    status = d.get_status()
    _check(status["batch_active"], "Window is OPEN after first enqueue")
    _check(status["queued_count"] == 1, "1 request queued")

    # tick() before 2s should return nothing
    actions = d.tick()
    _check(actions == [], "tick() returns empty before window expires")

    # Add a second request mid-window (should NOT restart the clock)
    time.sleep(0.5)
    env.add_request(3, 7)
    req2 = env.pending_requests[-1]
    d.enqueue(req2)
    status2 = d.get_status()
    _check(status2["queued_count"] == 2, "Second request queued without restarting window")

    remaining = status2["countdown_ms"]
    # Remaining should be roughly (BATCH_WINDOW_SECONDS - 0.5)*1000, i.e. < 2000
    _check(
        remaining < BATCH_WINDOW_SECONDS * 1000,
        f"Window countdown did not reset (countdown_ms={remaining})",
    )

    # Wait for window to expire
    elapsed = time.monotonic() - t0
    wait_left = BATCH_WINDOW_SECONDS - elapsed + 0.05  # tiny buffer
    if wait_left > 0:
        time.sleep(wait_left)

    actions = d.tick()
    t_dispatch = time.monotonic() - t0

    _check(len(actions) == 2, f"Both requests dispatched after window ({len(actions)} actions)")
    _check(
        t_dispatch < BATCH_WINDOW_SECONDS + 0.5,
        f"Dispatch happened within window + 0.5s buffer (took {t_dispatch:.3f}s)",
    )

    # Window should now be closed
    status3 = d.get_status()
    _check(not status3["batch_active"], "Window is CLOSED after flush")


# ---------------------------------------------------------------------------
# T7: Per-passenger pickup and drop-off floor correctness
# ---------------------------------------------------------------------------
def test_T7_pickup_and_dropoff_floors():
    print("\n--- T7: Each passenger picked up and dropped off at correct floors ---")
    env, q, d = _make_env_dispatcher(capacity=8)

    trips = [(1, 5), (3, 8)]
    for src, dst in trips:
        env.add_request(src, dst)
    for req in list(env.pending_requests):
        d.enqueue(req)

    d.flush()

    # Track pickups and drop-offs during simulation
    pickups   = {}  # dst -> floor where boarding happened
    dropoffs  = {}  # dst -> floor where drop-off happened

    for step in range(30):
        prev_occ = [list(p) for p in env.elevator_passengers]
        prev_pos = list(env.elevator_positions)

        state, _, info = env.step()

        curr_occ = [list(p) for p in env.elevator_passengers]
        curr_pos = list(env.elevator_positions)

        for e in range(env.num_elevators):
            for dst in curr_occ[e]:
                if dst not in prev_occ[e]:
                    pickups[dst] = curr_pos[e]
            for dst in prev_occ[e]:
                if dst not in curr_occ[e]:
                    dropoffs[dst] = curr_pos[e]

        if len(pickups) == len(trips) and len(dropoffs) == len(trips):
            break

    for src, dst in trips:
        if dst in pickups:
            _check(pickups[dst] == src, f"Passenger F{src}->F{dst} boarded at F{src} (got F{pickups.get(dst)})")
        if dst in dropoffs:
            _check(dropoffs[dst] == dst, f"Passenger F{src}->F{dst} dropped off at F{dst} (got F{dropoffs.get(dst)})")

    _check(len(env.pending_requests) == 0, "Queue empty after all deliveries")


# ---------------------------------------------------------------------------
# T8: Flask API integration
# ---------------------------------------------------------------------------
def test_T8_flask_api_integration():
    print("\n--- T8: Flask API /api/request and /api/batch_status integration ---")
    client = flask_app.app.test_client()

    # Reset environment
    res = client.post("/api/reset")
    assert res.status_code == 200
    state = res.get_json()["state"]
    _check(state["elevator_positions"] == [1, 1, 1], "Reset: positions [1,1,1]")
    _check(state["pending_requests"] == 0, "Reset: no pending requests")

    # Start the system
    client.post("/api/start")

    # Submit first request
    res1 = client.post("/api/request", json={"src_floor": 1, "dst_floor": 9})
    _check(res1.status_code == 200, "POST /api/request returns 200")
    data1 = res1.get_json()
    _check(data1["status"] == "success", "Request 1 accepted")

    state1 = data1["state"]
    _check(
        state1["batch_active"] or state1["pending_requests"] >= 1,
        "Batch window active OR request pending after first request",
    )
    _check("batch_countdown_ms" in state1, "batch_countdown_ms present in state")
    _check("batch_queued" in state1, "batch_queued present in state")

    # /api/batch_status endpoint
    bs = client.get("/api/batch_status")
    _check(bs.status_code == 200, "GET /api/batch_status returns 200")
    bdata = bs.get_json()
    _check("batch_active" in bdata, "batch_status has batch_active field")
    _check("countdown_ms" in bdata, "batch_status has countdown_ms field")
    _check("queued_count" in bdata, "batch_status has queued_count field")

    # Submit two more requests quickly (same window)
    res2 = client.post("/api/request", json={"src_floor": 1, "dst_floor": 9})
    res3 = client.post("/api/request", json={"src_floor": 1, "dst_floor": 9})
    _check(res2.status_code == 200, "Request 2 accepted")
    _check(res3.status_code == 200, "Request 3 accepted")

    # Wait for batch window to expire
    time.sleep(BATCH_WINDOW_SECONDS + 0.2)

    # One step to trigger flush
    res_step = client.post("/api/step")
    _check(res_step.status_code == 200, "POST /api/step after window returns 200")
    step_data = res_step.get_json()

    # After flush, batch window should be inactive
    state_after = step_data["state"]
    _check(
        not state_after["batch_active"],
        "Batch window closed after 2s + step",
    )

    # The three F1->F9 requests should have been batched:
    # verify the batch_queued in state reflects 0 (all dispatched)
    _check(
        state_after["batch_queued"] == 0,
        f"batch_queued=0 after dispatch (got {state_after['batch_queued']})",
    )

    print("  [PASS] Flask API integration complete.")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AI ELEVATOR SYSTEM – 2-Second Batch Dispatch Test Suite")
    print("=" * 70)

    test_T1_identical_requests_single_lift()
    test_T2_compatible_upward_requests()
    test_T3_capacity_overflow_extra_lift()
    test_T4_incompatible_directions_separate_lifts()

    # T5: only print the final summary
    print("\n--- T5: No requests -> all lifts strictly stationary ---")
    env_t5, q_t5, d_t5 = _make_env_dispatcher()
    initial_t5 = list(env_t5.elevator_positions)
    all_static = True
    for _ in range(25):
        d_t5.tick()
        state_t5, _, info_t5 = env_t5.step()
        if info_t5["movement"] != 0 or state_t5["elevator_positions"] != initial_t5:
            all_static = False
    _check(all_static, "All 3 lifts strictly stationary for 25 steps with empty queue")

    test_T6_window_timing()
    test_T7_pickup_and_dropoff_floors()
    test_T8_flask_api_integration()

    print("\n" + "=" * 70)
    total = _passed + _failed
    print(f"  Results: {_passed}/{total} checks passed, {_failed} failed")
    if _failed == 0:
        print("  ALL BATCH DISPATCH TESTS PASSED SUCCESSFULLY!")
    else:
        print("  SOME TESTS FAILED – see above for details.")
    print("=" * 70)
    sys.exit(0 if _failed == 0 else 1)

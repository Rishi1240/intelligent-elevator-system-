"""
Automated Test Suite for User-Controlled, Automatically Advancing Simulation Clock
and Predictive Elevator Control Integration.

Covers all 13 verification points specified in Section 10:
  1. Setting initial simulated date and time to 09:00:00 AM.
  2. Advancing two real-equivalent minutes at 1x to 09:02:00 AM.
  3. Correct advancement at 2x and 5x speeds.
  4. Pausing and resuming without losing time or introducing jumps.
  5. Changing selected date and time cleanly.
  6. Correct minute, hour, midnight, and weekday transitions.
  7. Active simulation calendar time directly driving LSTM feature vector.
  8. Updated time reaching prediction metadata and forecast inputs.
  9. Prediction output passed to real scheduling decisions (predictive positioning).
  10. State polling (GET /api/state) and rapid calls do not advance time twice.
  11. Clock changes preserve active passenger assignments, routes, and emergencies.
  12. Floor travel (1s/floor) and collection-window (3s dwell) timing remain exact.
  13. Existing compatible route sharing, capacity (8 max), and idle lift policies pass.
"""

from datetime import datetime, timedelta
import pytest
from backend.sim_clock import SimClock
from backend.elevator_env import ElevatorEnvironment, DWELL_WINDOW_SECS, TRAVEL_SECS
from backend import app as flask_app
from backend.app import (
    build_current_feature_vector,
    seed_feature_buffer,
    FEATURE_BUFFER,
    _run_lstm_inference,
    _apply_predictive_positioning,
)


# ── 1. Setting Initial Date and Time to 09:00:00 AM ───────────────────────────

def test_1_setting_initial_date_and_time_090000():
    clock = SimClock(
        deterministic=True,
        timezone="Asia/Kolkata",
        initial_date="2026-10-14",
        initial_time="09:00:00 AM",
    )
    dt = clock.get_calendar_datetime()
    assert dt.hour == 9
    assert dt.minute == 0
    assert dt.second == 0
    assert dt.strftime("%Y-%m-%d") == "2026-10-14"
    status = clock.get_status_dict()
    assert status["calendar_time_str"] == "09:00:00 AM"
    assert status["weekday"] == "Wednesday"
    assert "Asia/Kolkata" in status["timezone"]


# ── 2. Advancing Two Real-Equivalent Minutes at 1x to 09:02:00 AM ────────────

def test_2_advancing_two_minutes_at_1x_to_090200():
    clock = SimClock(
        deterministic=True,
        timezone="Asia/Kolkata",
        initial_date="2026-10-14",
        initial_time="09:00:00 AM",
    )
    clock.set_speed(1.0)
    # 2 real minutes at 1x speed = 120 simulated seconds
    clock.advance(120.0)
    dt = clock.get_calendar_datetime()
    assert dt.hour == 9
    assert dt.minute == 2
    assert dt.second == 0
    assert dt.strftime("%I:%M:%S %p") == "09:02:00 AM"
    assert clock.now() == 120.0


# ── 3. Correct Advancement at 2x and 5x Speeds ───────────────────────────────

def test_3_advancement_at_2x_and_5x_speeds():
    # 2x speed: 2 minutes (120 wall-seconds) yields 240 simulated seconds (4 minutes)
    clock_2x = SimClock(
        deterministic=True,
        timezone="Asia/Kolkata",
        initial_date="2026-10-14",
        initial_time="09:00:00 AM",
    )
    clock_2x.set_speed(2.0)
    # In deterministic equivalent: advance by wall_seconds * speed
    wall_seconds = 120.0
    clock_2x.advance(wall_seconds * clock_2x.speed)
    dt_2x = clock_2x.get_calendar_datetime()
    assert dt_2x.hour == 9
    assert dt_2x.minute == 4
    assert dt_2x.second == 0
    assert dt_2x.strftime("%I:%M:%S %p") == "09:04:00 AM"

    # 5x speed: 2 minutes (120 wall-seconds) yields 600 simulated seconds (10 minutes)
    clock_5x = SimClock(
        deterministic=True,
        timezone="Asia/Kolkata",
        initial_date="2026-10-14",
        initial_time="09:00:00 AM",
    )
    clock_5x.set_speed(5.0)
    clock_5x.advance(wall_seconds * clock_5x.speed)
    dt_5x = clock_5x.get_calendar_datetime()
    assert dt_5x.hour == 9
    assert dt_5x.minute == 10
    assert dt_5x.second == 0
    assert dt_5x.strftime("%I:%M:%S %p") == "09:10:00 AM"


# ── 4. Pausing and Resuming Without Losing Time or Introducing Jumps ─────────

def test_4_pause_and_resume_semantics():
    clock = SimClock(
        deterministic=True,
        timezone="Asia/Kolkata",
        initial_date="2026-10-14",
        initial_time="09:00:00 AM",
    )
    clock.start()
    clock.advance(45.0)
    assert clock.now() == 45.0
    dt_pause = clock.get_calendar_datetime()
    assert dt_pause.minute == 0
    assert dt_pause.second == 45

    # Pause clock: time freezes
    clock.pause()
    assert not clock.is_running
    # While paused, querying time returns exact paused point
    assert clock.now() == 45.0
    assert clock.get_calendar_datetime() == dt_pause

    # Resume clock: continues from exact paused moment without jumping
    clock.resume()
    assert clock.is_running
    clock.advance(30.0)
    assert clock.now() == 75.0
    dt_resumed = clock.get_calendar_datetime()
    assert dt_resumed.minute == 1
    assert dt_resumed.second == 15


# ── 5. Changing the Selected Date and Time ───────────────────────────────────

def test_5_changing_selected_date_and_time():
    clock = SimClock(
        deterministic=True,
        timezone="Asia/Kolkata",
        initial_date="2026-10-14",
        initial_time="09:00:00 AM",
    )
    clock.advance(50.0)
    assert clock.now() == 50.0

    # User picks custom afternoon time
    new_dt = clock.set_custom_time(date_str="2026-10-15", time_str="02:30:00 PM")
    assert new_dt.hour == 14
    assert new_dt.minute == 30
    assert new_dt.second == 0
    assert clock.now() == 0.0  # re-anchored elapsed seconds
    assert clock.get_status_dict()["calendar_time_str"] == "02:30:00 PM"
    assert clock.get_status_dict()["weekday"] == "Thursday"


# ── 6. Correct Hour, Midnight, and Weekday Transitions ───────────────────────

def test_6_midnight_and_weekday_transitions():
    # Start on Sunday at 23:58:30 PM
    clock = SimClock(
        deterministic=True,
        timezone="Asia/Kolkata",
        initial_date="2026-10-18",  # Sunday
        initial_time="11:58:30 PM",
    )
    assert clock.get_status_dict()["weekday"] == "Sunday"

    # Advance 120 seconds (2 minutes) -> crosses midnight into Monday 00:00:30 AM
    clock.advance(120.0)
    dt = clock.get_calendar_datetime()
    assert dt.hour == 0
    assert dt.minute == 0
    assert dt.second == 30
    assert dt.strftime("%Y-%m-%d") == "2026-10-19"
    assert clock.get_status_dict()["weekday"] == "Monday"
    assert clock.get_status_dict()["weekday_idx"] == 0


# ── 7. Active Simulation Time Reaching Actual LSTM Feature Vector ─────────────

def test_7_simulation_time_reaches_lstm_feature_vector():
    flask_app.sim_clock.set_custom_time(date_str="2026-10-19", time_str="09:00:00 AM")
    dummy_state = {
        "floor_demand": [4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "pending_requests": 5,
        "elevator_positions": [1, 2, 3],
        "elevator_occupancy": [0, 0, 0],
    }

    # At 09:00:00
    vec_0900 = build_current_feature_vector(dummy_state)
    assert vec_0900[0] == 9   # hour
    assert vec_0900[1] == 0   # minute
    assert vec_0900[2] == 0   # day_of_week (Monday)

    # Advance clock to 09:01:00 AM
    flask_app.sim_clock.advance(60.0)
    vec_0901 = build_current_feature_vector(dummy_state)
    assert vec_0901[0] == 9
    assert vec_0901[1] == 1
    assert vec_0901[2] == 0

    # Advance clock to 09:02:00 AM
    flask_app.sim_clock.advance(60.0)
    vec_0902 = build_current_feature_vector(dummy_state)
    assert vec_0902[0] == 9
    assert vec_0902[1] == 2
    assert vec_0902[2] == 0


# ── 8. Updated Time Reaching Prediction Metadata and Forecast Input ──────────

def test_8_time_reaching_prediction_metadata_and_api():
    client = flask_app.app.test_client()

    # Set custom time via API
    res = client.post("/api/clock/set", json={
        "date": "2026-10-20",
        "time": "09:00:00 AM",
        "mode": "custom"
    })
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["clock"]["calendar_time_str"] == "09:00:00 AM"
    assert payload["clock"]["calendar_date_str"] == "2026-10-20"
    assert payload["clock"]["weekday"] == "Tuesday"

    # Verify FEATURE_BUFFER was seeded with 10 sequential chronological steps
    assert len(FEATURE_BUFFER) == 10
    # Last step should match 09:00:00
    last_frame = FEATURE_BUFFER[-1]
    assert last_frame[0] == 9  # hour
    assert last_frame[1] == 0  # minute
    assert last_frame[2] == 1  # Tuesday

    # First step should match 9 minutes prior (08:51:00)
    first_frame = FEATURE_BUFFER[0]
    assert first_frame[0] == 8
    assert first_frame[1] == 51

    # Verify state response includes authoritative metadata
    st = payload["state"]
    assert st["prediction_horizon_min"] == 5
    assert st["sim_time_display"] == "09:00:00 AM"
    assert st["sim_weekday"] == "Tuesday"


# ── 9. Prediction Output Passed to Real Scheduling Decision Function ─────────

def test_9_prediction_passed_to_predictive_positioning():
    """
    Verifies the new conservative predictive positioning policy:
    - Low demand (< DEMAND_THRESHOLD=8.0) does NOT reposition any lift.
    - High demand (>= 8.0) must be sustained for 3 cycles before reposition.
    - At most 1 lift is repositioned (MAX_IDLE_PREPOSITION=1).
    - At least 1 lift remains untouched as backup.
    """
    # flask_app is imported at module level as: from backend import app as flask_app
    fapp = flask_app

    # ── Part A: Low demand must not trigger repositioning ──────────────────
    low_demand = [4.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
    fapp.env.reset()
    fapp._REPOSITION_COOLDOWN.clear()
    fapp._DEMAND_SUSTAINED_COUNT = 0
    fapp._LAST_PREPOSITION_FLOOR = None
    fapp._LAST_PREPOSITION_WALL_TIME = 0.0
    fapp.env.elevator_positions = [6, 7, 8]
    fapp.env.committed_targets = [None, None, None]
    fapp.env.assigned_stops = [[], [], []]
    fapp.env.elevator_passengers = [[], [], []]
    fapp.env.pending_requests = []

    _apply_predictive_positioning(locked_lifts=set(), floor_demand_t5=low_demand)
    repositioned_low = [e for e in range(3) if fapp.env.committed_targets[e] is not None]
    assert len(repositioned_low) == 0, (
        "Low demand (4.5 < DEMAND_THRESHOLD=8.0) must NOT trigger repositioning"
    )

    # ── Part B: High demand sustained for 3 cycles triggers exactly 1 reposition ──
    # Demand spike at Floor 1 (lobby) above threshold
    high_demand = [9.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
    fapp.env.reset()
    fapp._REPOSITION_COOLDOWN.clear()
    fapp._DEMAND_SUSTAINED_COUNT = 0
    fapp._LAST_PREPOSITION_FLOOR = None
    fapp._LAST_PREPOSITION_WALL_TIME = 0.0
    fapp.env.elevator_positions = [6, 7, 8]
    fapp.env.committed_targets = [None, None, None]
    fapp.env.assigned_stops = [[], [], []]
    fapp.env.elevator_passengers = [[], [], []]
    fapp.env.pending_requests = []

    # First 2 calls: sustained counter builds up but no reposition yet
    _apply_predictive_positioning(locked_lifts=set(), floor_demand_t5=high_demand)
    assert fapp._DEMAND_SUSTAINED_COUNT == 1
    repositioned_after_1 = [e for e in range(3) if fapp.env.committed_targets[e] is not None]
    assert len(repositioned_after_1) == 0, "Reposition must NOT fire until 3 sustained cycles"

    _apply_predictive_positioning(locked_lifts=set(), floor_demand_t5=high_demand)
    assert fapp._DEMAND_SUSTAINED_COUNT == 2
    repositioned_after_2 = [e for e in range(3) if fapp.env.committed_targets[e] is not None]
    assert len(repositioned_after_2) == 0, "Reposition must NOT fire until 3 sustained cycles"

    # Third call: sustained counter reaches DEMAND_SUSTAINED_REQUIRED=3 → reposition fires
    _apply_predictive_positioning(locked_lifts=set(), floor_demand_t5=high_demand)
    repositioned = [e for e in range(3) if fapp.env.committed_targets[e] == 1]
    untouched = [e for e in range(3) if fapp.env.committed_targets[e] is None]

    # Exactly 1 repositioned, at least 1 left as backup
    assert len(repositioned) == 1, f"Exactly 1 lift should be repositioned, got {len(repositioned)}"
    assert len(untouched) >= 1, "At least 1 lift must remain untouched as backup"
    # Sustained count resets after firing
    assert fapp._DEMAND_SUSTAINED_COUNT == 0, "Sustained count must reset after repositioning"



# ── 10. State Polling and Rerenders Do Not Advance Time Twice ─────────────────

def test_10_state_polling_does_not_advance_time():
    client = flask_app.app.test_client()
    # Freeze / reset
    client.post("/api/reset")
    res1 = client.get("/api/state")
    t1 = res1.get_json()["sim_time"]

    # Repeated GET requests must NOT advance simulated time
    for _ in range(5):
        res = client.get("/api/state")
        assert res.get_json()["sim_time"] == t1

    res_status = client.get("/api/clock/status")
    assert res_status.get_json()["clock"]["sim_time_seconds"] == t1


# ── 11. Clock Changes Preserve Active Passenger Assignments and Emergencies ───

def test_11_clock_changes_preserve_active_assignments_and_emergencies():
    client = flask_app.app.test_client()
    client.post("/api/reset")
    # Add passenger request
    client.post("/api/request", json={"src_floor": 2, "dst_floor": 7})
    # Add emergency
    client.post("/api/emergency", json={"floor": 9})

    st_before = client.get("/api/state").get_json()
    assert st_before["has_active_emergency"] is True

    # User adjusts custom simulation time
    res = client.post("/api/clock/set", json={"date": "2026-10-21", "time": "10:15:00 AM"})
    st_after = res.get_json()["state"]

    # Emergency and passenger requests must NOT be wiped out by clock change
    assert st_after["has_active_emergency"] is True
    assert st_after["active_emergencies"][0]["floor"] == 9
    assert st_after["pending_requests"] >= 1 or any(
        7 in occ for occ in st_after["assigned_stops"]
    )


# ── 12. Floor Travel and Collection-Window Timing Preserved ───────────────────

def test_12_floor_travel_and_collection_window_timing():
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=5, num_elevators=1, clock=clock, dwell_window_secs=3.0)
    env.elevator_positions[0] = 1
    env.elevator_passengers[0] = [2]
    env.committed_targets[0] = 2

    # Step 1: Moves from F1 to F2 (1s intermediate travel)
    env.step()
    assert env.elevator_positions[0] == 2
    # At destination: triggers 3-second dwell window
    assert env.dwell_remaining[0] == DWELL_WINDOW_SECS
    assert env.completed_passengers == 1

    # Next step during dwell: stays stationary
    env.committed_targets[0] = 5
    env.step()
    assert env.elevator_positions[0] == 2
    assert env.dwell_remaining[0] == DWELL_WINDOW_SECS - TRAVEL_SECS


# ── 13. Existing Route Sharing, Capacity, and Idle Policy Continuing to Pass ─

def test_13_compatible_route_sharing_and_capacity():
    clock = SimClock(deterministic=True)
    env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8, clock=clock)

    # Req 1: F1 -> F9 assigned to Lift 0
    req1 = {"id": 1, "src": 1, "dst": 9, "assigned_to": None, "wait_time": 0.0}
    env.pending_requests.append(req1)
    env.assign_request(req1, 0)

    # Compatible Req 2: F4 -> F8 along the route
    req2 = {"id": 2, "src": 4, "dst": 8, "assigned_to": None, "wait_time": 0.0}
    env.pending_requests.append(req2)
    assert env.can_accommodate(0, req2) is True
    env.assign_request(req2, 0)

    # Both requests share Lift 0
    assert 4 in env.assigned_stops[0] or 8 in env.assigned_stops[0]
    # Lifts 1 and 2 remain idle and stationary
    assert env.elevator_positions[1] == 1
    assert env.elevator_positions[2] == 1

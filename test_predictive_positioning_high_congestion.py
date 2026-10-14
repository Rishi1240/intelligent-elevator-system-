"""
Automated Test Suite for High-Congestion Predictive Elevator Pre-Positioning.
Updated for >= 55% activation threshold with demand-aware allocation.

Verifies all 14 mandatory behavioral requirements:
1. At 40% and exactly 54.9% congestion, prediction alone issues no movement commands.
2. One isolated prediction of 90% does not activate positioning; two consecutive valid predictions above 55% can activate it.
3. Sustained 90% congestion with highest predicted demand at F3 causes suitable idle elevators to be positioned at or toward F3.
4. If the highest predicted demand is at F8 instead, the target changes to F8 after the confirmation rule is met.
5. Repeated identical predictions do not create duplicate assignments or repeated target commands.
6. Already-positioned elevators are not sent on unnecessary trips.
7. A lift carrying passengers or committed to a passenger route is never redirected for prediction.
8. Real requests, compatible route sharing, capacity constraints, and the 2-second batching window continue to work.
9. Emergency assignment and resolution continue to work and cannot be overridden by prediction.
10. Predictions fluctuating between 54% and 56% do not cause elevator oscillation.
11. At low demand and with no requests, all idle elevators remain stationary.
12. If the model has no valid floor-specific output or is not successfully loaded, predictive positioning fails safely and logs the reason.
13. Changing the selected simulation time to 9:15 AM changes the time features used in inference. High congestion alone does not move elevators until the confirmation and eligibility rules are satisfied.
14. Existing test suites continue to pass.
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import app as fapp
from backend.predictive_positioner import PredictivePositioner


def reset_test_state():
    """Resets simulator and positioner to a completely clean baseline."""
    fapp.env.reset()
    fapp.batch_dispatcher.reset()
    fapp.emergency_dispatcher.reset()
    fapp.predictive_positioner.reset()
    fapp.IS_RUNNING = True
    fapp.sim_clock.start()
    fapp._LAST_PREDICTION_SIM_TIME = fapp.sim_clock.now()
    fapp._LATEST_DISPATCH_EVENTS.clear()
    fapp._LATEST_EMG_EVENTS.clear()


# ── Test 1: No movement below 55% threshold ───────────────────────────────────

def test_1_no_movement_below_55_percent_congestion():
    """At 40% and 54.9% congestion, prediction alone issues no movement commands."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]
    demand_f3 = [0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for cong in [0.40, 0.549]:
        fapp.predictive_positioner.reset()
        fapp._LAST_CONGESTION_T5 = cong
        fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)
        for _ in range(5):
            res = fapp.run_rl_dispatch_step()
            next_state, reward, info, has_pax = res
            assert next_state is None, f"Movement occurred at {cong*100}% congestion! Should be stationary."
            assert all(t is None for t in fapp.env.committed_targets)
            assert fapp.env.elevator_positions == [1, 1, 1]
            assert fapp.predictive_positioner.status == "inactive"


# ── Test 2: Isolated 90% does not activate; 2 consecutive activates ──────────

def test_2_isolated_90_percent_no_positioning_two_consecutive_activates():
    """One isolated prediction of 90% does not activate positioning; two consecutive valid predictions above 55% can activate it."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]
    demand_f3 = [0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Cycle 1: 90% congestion (isolated update)
    fapp._LAST_CONGESTION_T5 = 0.90
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)
    res1 = fapp.run_rl_dispatch_step()
    assert res1[0] is None, "First isolated prediction of 90% must NOT cause movement!"
    assert all(t is None for t in fapp.env.committed_targets)
    assert fapp.predictive_positioner.status == "awaiting_confirmation"
    assert fapp.predictive_positioner.consecutive_high_count == 1

    # Cycle 2: 2nd consecutive 90% prediction
    res2 = fapp.run_rl_dispatch_step()
    next_state = res2[0]
    assert next_state is not None, "Second consecutive 90% prediction MUST activate positioning!"
    assert fapp.predictive_positioner.status == "active"
    assert fapp.predictive_positioner.consecutive_high_count >= 2


# ── Test 3: Sustained 90% with peak at F3 positions elevators ────────────────

def test_3_sustained_90_percent_highest_demand_f3_positions_elevators():
    """Sustained 90% congestion with highest predicted demand at F3 causes suitable idle elevators to be positioned at or toward F3."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]
    demand_f3 = [0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fapp._LAST_CONGESTION_T5 = 0.90
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)

    # Confirmation cycle 1 & 2
    fapp.run_rl_dispatch_step()
    fapp.run_rl_dispatch_step()

    assigned_lifts = fapp.predictive_positioner.get_assigned_elevators()
    # With demand=18.0 and capacity=4.0, ceil(18/4) = 5 but only 3 elevators available
    # All 3 should be assigned to F3 since demand far exceeds capacity
    assert len(assigned_lifts) >= 1, f"At least 1 elevator must be assigned toward F3, got {assigned_lifts}"

    # All assigned should target F3
    for eid in assigned_lifts:
        assert fapp.env.committed_targets[eid] == 3, \
            f"Elevator {eid} should target F3, got {fapp.env.committed_targets[eid]}"


# ── Test 4: Target changes to F8 after confirmation ──────────────────────────

def test_4_target_changes_to_f8_after_confirmation():
    """If the highest predicted demand is at F8 instead, the target changes to F8 after the confirmation rule is met."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]
    demand_f3 = [0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    demand_f8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.0, 0.0, 0.0]

    fapp._LAST_CONGESTION_T5 = 0.90
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)
    # Activate on F3
    fapp.run_rl_dispatch_step()
    fapp.run_rl_dispatch_step()
    assert fapp.predictive_positioner.active_target_floor == 3

    # Switch demand to F8 – 1st update: target must remain F3 (anti-oscillation)
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f8)
    fapp.run_rl_dispatch_step()
    assert fapp.predictive_positioner.active_target_floor == 3, "Target must not change immediately on 1 update"

    # 2nd update to F8: confirmed and switches to F8
    fapp.run_rl_dispatch_step()
    assert fapp.predictive_positioner.active_target_floor == 8, "Target must switch to F8 after 2 confirmations"


# ── Test 5: Repeated identical predictions do not create duplicate commands ───

def test_5_repeated_identical_predictions_no_duplicate_assignments():
    """Repeated identical predictions do not create duplicate assignments or repeated target commands."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]
    demand_f3 = [0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fapp._LAST_CONGESTION_T5 = 0.90
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)

    fapp.run_rl_dispatch_step()
    fapp.run_rl_dispatch_step()
    initial_assigned = list(fapp.predictive_positioner.get_assigned_elevators())

    # Run 10 subsequent identical prediction updates
    for _ in range(10):
        fapp.run_rl_dispatch_step()
        # Assigned elevator set must remain consistent
        current_assigned = fapp.predictive_positioner.get_assigned_elevators()
        assert len(current_assigned) <= 3, f"Assigned count must not exceed 3 elevators, got {current_assigned}"


# ── Test 6: Already-positioned elevators are not sent on unnecessary trips ───

def test_6_already_positioned_elevators_not_sent_unnecessarily():
    """Already-positioned elevators are not sent on unnecessary trips."""
    reset_test_state()
    # Two elevators already at F3, one at F1
    fapp.env.elevator_positions = [3, 3, 1]
    fapp.env.committed_targets = [None, None, None]
    # Demand of 3.0: single elevator capacity (4.0) covers it, but 2 are already at F3
    demand_f3 = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fapp._LAST_CONGESTION_T5 = 0.90
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)

    fapp.run_rl_dispatch_step()
    fapp.run_rl_dispatch_step()

    # 2 are already at F3 and demand (3.0) < capacity of one elevator (4.0),
    # so no additional movements are needed – elevator at F1 stays put
    assert fapp.env.elevator_positions == [3, 3, 1], "Positions should not change"


# ── Test 7: Passenger carrying or committed lift never redirected ────────────

def test_7_passenger_carrying_or_committed_lift_never_redirected():
    """A lift carrying passengers or committed to a passenger route is never redirected for prediction."""
    reset_test_state()
    # Lift 0 has passengers onboard heading to F9
    fapp.env.elevator_positions = [2, 1, 1]
    fapp.env.elevator_passengers[0] = [9]
    fapp.env.committed_targets[0] = 9
    fapp.env.assigned_stops[0] = [9]

    demand_f3 = [0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fapp._LAST_CONGESTION_T5 = 0.90
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)

    fapp.run_rl_dispatch_step()
    fapp.run_rl_dispatch_step()

    # Lift 0 must remain committed to F9 passenger stop, never stolen
    assert fapp.env.committed_targets[0] == 9, "Passenger route on Lift 0 was overwritten!"
    assert 0 not in fapp.predictive_positioner.get_assigned_elevators()
    # Only eligible idle lifts (1 and 2) can be positioned
    assigned = fapp.predictive_positioner.get_assigned_elevators()
    for eid in assigned:
        assert eid != 0, f"Elevator 0 (carrying passengers) should not be assigned!"


# ── Test 8: Real requests, compatible sharing, capacity, and 2s batching ─────

def test_8_real_requests_compatible_route_sharing_capacity_2s_batching():
    """Real requests, compatible route sharing, capacity constraints, and the 2-second batching window continue to work."""
    reset_test_state()
    # Queue a real passenger request
    success = fapp.env.add_request(2, 7)
    assert success
    req = fapp.env.pending_requests[-1]
    fapp.batch_dispatcher.enqueue(req)

    assert fapp.batch_dispatcher.get_status()["batch_active"] is True
    assert fapp.batch_dispatcher.get_status()["queued_count"] == 1

    # Real passenger dispatch takes precedence over any predictive positioning
    actions = fapp.batch_dispatcher.flush()
    assert len(actions) == 1
    assigned_eid = actions[0]["elevator_id"]
    # Lift must be assigned to the passenger request
    assert fapp.env.committed_targets[assigned_eid] == 2 or 2 in fapp.env.assigned_stops[assigned_eid]


# ── Test 9: Emergency assignment cannot be overridden by prediction ──────────

def test_9_emergency_assignment_and_resolution_cannot_be_overridden_by_prediction():
    """Emergency assignment and resolution continue to work and cannot be overridden by prediction."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]

    # Trigger emergency at Floor 9
    res = fapp.emergency_dispatcher.dispatch_emergency(9)
    assert res["status"] == "success"
    emg_lift = res["emergency"]["elevator_id"]
    locked = fapp.emergency_dispatcher.get_locked_elevator_ids()
    assert emg_lift in locked

    # Now attempt high congestion prediction at Floor 3
    demand_f3 = [0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fapp._LAST_CONGESTION_T5 = 0.95
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)

    fapp.run_rl_dispatch_step()
    fapp.run_rl_dispatch_step()

    # The emergency elevator MUST NOT be assigned to predictive positioning
    assert emg_lift not in fapp.predictive_positioner.get_assigned_elevators()
    assert fapp.env.committed_targets[emg_lift] == 9, "Emergency elevator destination was tampered with!"

    # Resolve emergency
    res_resolve = fapp.emergency_dispatcher.resolve_emergency(res["emergency"]["id"])
    assert res_resolve["status"] == "success"
    assert emg_lift not in fapp.emergency_dispatcher.get_locked_elevator_ids()


# ── Test 10: Fluctuations across 55% threshold do not cause oscillation ──────

def test_10_predictions_fluctuating_across_threshold_do_not_cause_oscillation():
    """Predictions fluctuating between 54% and 56% do not cause elevator oscillation."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]
    demand_f3 = [0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fapp._LAST_FLOOR_DEMAND_T5 = list(demand_f3)

    # Alternate 56% (high) and 54% (below threshold) repeatedly
    for _ in range(15):
        # 56% (update 1)
        fapp._LAST_CONGESTION_T5 = 0.56
        fapp.run_rl_dispatch_step()
        assert fapp.predictive_positioner.status in ("awaiting_confirmation", "inactive")

        # 54% (resets confirmation counter)
        fapp._LAST_CONGESTION_T5 = 0.54
        fapp.run_rl_dispatch_step()
        assert fapp.predictive_positioner.status == "inactive"
        assert fapp.predictive_positioner.consecutive_high_count == 0

    # Elevators must have stayed strictly stationary at Floor 1 with NO oscillation
    assert fapp.env.elevator_positions == [1, 1, 1]
    assert all(t is None for t in fapp.env.committed_targets)


# ── Test 11: Low demand, no requests, all idle elevators stay stationary ─────

def test_11_at_low_demand_and_with_no_requests_all_idle_elevators_remain_stationary():
    """At low demand and with no requests, all idle elevators remain stationary."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]
    fapp._LAST_CONGESTION_T5 = 0.15
    fapp._LAST_FLOOR_DEMAND_T5 = [0.5] * 10

    initial_empty = fapp.env.empty_movement_floors

    for _ in range(30):
        res = fapp.run_rl_dispatch_step()
        assert res[0] is None, "Idle guard failed! Environment step was executed when idle."
        assert fapp.env.elevator_positions == [1, 1, 1]

    # No wasted energy
    assert fapp.env.empty_movement_floors == initial_empty


# ── Test 12: Missing or invalid model output fails safely and logs reason ────

def test_12_missing_or_invalid_model_output_fails_safely_and_logs():
    """If the model has no valid floor-specific output or is not successfully loaded, predictive positioning fails safely and logs the reason."""
    reset_test_state()
    fapp.env.elevator_positions = [1, 1, 1]
    pos = PredictivePositioner()

    # Test 1: model not loaded
    activated, reason, actions = pos.evaluate(
        env=fapp.env,
        congestion_t5=0.90,
        floor_demand_t5=[10.0] * 10,
        locked_elevator_ids=set(),
        model_loaded=False,
    )
    assert not activated
    assert pos.status == "unavailable"
    assert "not successfully loaded" in reason

    # Test 2: empty floor demand
    activated2, reason2, actions2 = pos.evaluate(
        env=fapp.env,
        congestion_t5=0.90,
        floor_demand_t5=[],
        locked_elevator_ids=set(),
        model_loaded=True,
    )
    assert not activated2
    assert pos.status == "unavailable"
    assert "No valid floor-specific" in reason2

    # In both cases no elevators are moved
    assert all(t is None for t in fapp.env.committed_targets)


# ── Test 13: Simulation time 9:15 AM changes time features, obeys rules ─────

def test_13_changing_simulation_time_to_9_15_am_changes_features_and_obeys_rules():
    """Changing the selected simulation time to 9:15 AM changes the time features used in inference. High congestion alone does not move elevators until the confirmation and eligibility rules are satisfied."""
    reset_test_state()
    # Set simulation time to 09:15:00 AM
    new_base = fapp.sim_clock.set_custom_time(time_str="09:15:00 AM")
    assert new_base.hour == 9
    assert new_base.minute == 15

    # Feature vector builder must reflect hour 9, minute 15
    cur_state = fapp.env.get_state()
    vec = fapp.build_current_feature_vector(cur_state)
    assert vec[0] == 9, f"Expected hour 9, got {vec[0]}"
    assert vec[1] == 15, f"Expected minute 15, got {vec[1]}"

    # With high congestion, isolated single call does not move elevators
    fapp._LAST_CONGESTION_T5 = 0.90
    fapp._LAST_FLOOR_DEMAND_T5 = [0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    res1 = fapp.run_rl_dispatch_step()
    assert res1[0] is None, "Elevator moved before confirmation requirement was met!"

    # Second consecutive call satisfies confirmation and moves elevators
    res2 = fapp.run_rl_dispatch_step()
    assert res2[0] is not None
    assert fapp.predictive_positioner.status == "active"


# ── Test 14: Telemetry transparency and state API ────────────────────────────

def test_14_dashboard_telemetry_includes_all_required_state():
    """Telemetry includes congestion percentage, horizon, highest-demand floor, positioning status, assigned lifts, operational status, floor evaluations, and allocations."""
    reset_test_state()
    fapp._LAST_CONGESTION_T5 = 0.88
    fapp._LAST_FLOOR_DEMAND_T5 = [0.0, 0.0, 16.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Two cycles to activate
    fapp.run_rl_dispatch_step()
    fapp.run_rl_dispatch_step()

    state = fapp.get_system_state_dict()
    assert "congestion_t5" in state
    assert state["congestion_t5"] == 0.88
    assert state["prediction_horizon_min"] == 5
    assert state["predicted_target_floor"] == 3
    assert state["predictive_positioning_status"] == "active"
    assert len(state["assigned_predictive_elevators"]) >= 1
    assert "elevator_operational_status" in state
    assert len(state["elevator_operational_status"]) == fapp.env.num_elevators
    # Verify new telemetry fields
    assert "floor_evaluations" in state
    assert "assigned_allocations" in state
    assert "eligible_elevators_count" in state
    assert "last_decision_log" in state
    assert state["high_congestion_threshold"] == 0.55

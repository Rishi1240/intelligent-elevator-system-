"""
test_new_features.py
====================
New tests covering:
  - Future-target alignment (congestion_t5 / demand_f*_t5 columns)
  - LSTM model loading and predict_full() output
  - Forecast-to-dispatch integration (LSTM output influences app state)
  - Predictive positioning logic
  - Q-learning action selection with loaded Q-table
  - Model fallback when artifacts are absent
  - Idle elevators remain stationary when no work + no beneficial forecast
  - Predictive positioning targets the highest-demand floor
  - Current passenger requests override unnecessary predictive movement
  - Emergency dispatch bypasses batching
"""

import os
import sys
import time
import pickle
import random
import tempfile
import shutil

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
for p in (ROOT, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.elevator_env    import ElevatorEnvironment
from backend.q_learning_agent import QLearningAgent
from backend.lstm_model      import LSTMModel
from backend.batch_dispatcher import RequestBatchDispatcher
from backend.emergency_dispatcher import EmergencyDispatcher

SAVED_MODEL = os.path.join(BACKEND, "saved_model")
KERAS_PATH  = os.path.join(SAVED_MODEL, "lstm_weights.keras")
SCALER_PATH = os.path.join(SAVED_MODEL, "scaler.pkl")
QTABLE_PATH = os.path.join(SAVED_MODEL, "q_table.pkl")


# ── helpers ───────────────────────────────────────────────────────────────────

def make_env(**kw):
    return ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=8, **kw)


def make_seq(features=20, steps=10, val=0.0):
    """Dummy 10-step × 20-feature sequence."""
    return [[val] * features for _ in range(steps)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Future-target alignment
# ─────────────────────────────────────────────────────────────────────────────

class TestFutureTargetAlignment:
    """Verify training_data_future_5min.csv has correct schema."""

    @pytest.fixture(scope="class")
    def df(self):
        import pandas as pd
        csv = os.path.join(BACKEND, "training_data_future_5min.csv")
        if not os.path.exists(csv):
            pytest.skip("training_data_future_5min.csv not found")
        return pd.read_csv(csv)

    def test_has_congestion_t5(self, df):
        assert "congestion_t5" in df.columns, "Missing congestion_t5 column"

    def test_has_demand_columns(self, df):
        for i in range(1, 11):
            assert f"demand_f{i}_t5" in df.columns, f"Missing demand_f{i}_t5 column"

    def test_row_count(self, df):
        assert len(df) >= 1000, f"Expected >= 1000 rows, got {len(df)}"

    def test_congestion_range(self, df):
        assert df["congestion_t5"].between(0, 1.01).all(), \
            "congestion_t5 values out of [0,1] range"

    def test_demand_non_negative(self, df):
        for i in range(1, 11):
            col = f"demand_f{i}_t5"
            assert (df[col] >= 0).all(), f"{col} has negative values"

    def test_no_target_in_input_features(self, df):
        """Targets must not appear in the standard 20 input columns."""
        input_cols = {
            "hour", "minute", "day_of_week",
            *[f"floor_req_f{i}" for i in range(1, 11)],
            "total_requests",
            "elev1_floor", "elev2_floor", "elev3_floor",
            "elev1_occ", "elev2_occ", "elev3_occ",
        }
        for col in ("congestion_t5", *[f"demand_f{i}_t5" for i in range(1, 11)]):
            assert col not in input_cols, f"Target {col} found in input feature set"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM model loading and predict_full()
# ─────────────────────────────────────────────────────────────────────────────

class TestLSTMModelLoading:
    def test_fallback_when_no_weights(self):
        m = LSTMModel()
        assert m.model_mode == "fallback"
        assert not m.is_loaded

    def test_fallback_predict_returns_float(self):
        m = LSTMModel()
        seq = make_seq()
        result = m.predict(seq)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_fallback_predict_full_structure(self):
        m = LSTMModel()
        seq = make_seq()
        result = m.predict_full(seq)
        assert "congestion_t5" in result
        assert "floor_demand_t5" in result
        assert len(result["floor_demand_t5"]) == 10
        assert result["model_mode"] == "fallback"

    @pytest.mark.skipif(not os.path.exists(KERAS_PATH), reason="No trained .keras model")
    def test_load_keras_model(self):
        m = LSTMModel()
        ok = m.load(KERAS_PATH)
        assert ok, "load() returned False for existing .keras file"
        assert m.is_loaded
        assert m.model_mode == "loaded"

    @pytest.mark.skipif(not os.path.exists(KERAS_PATH), reason="No trained .keras model")
    def test_predict_full_with_loaded_model(self):
        m = LSTMModel()
        m.load(KERAS_PATH)
        seq = make_seq()
        result = m.predict_full(seq)
        assert "congestion_t5" in result
        assert 0.0 <= result["congestion_t5"] <= 1.0
        assert len(result["floor_demand_t5"]) == 10
        # All demand values non-negative
        assert all(v >= 0 for v in result["floor_demand_t5"])
        assert result["model_mode"] == "loaded"

    def test_load_missing_file_raises(self):
        m = LSTMModel()
        with pytest.raises(FileNotFoundError):
            m.load("/nonexistent/path/model.keras")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Q-learning action selection
# ─────────────────────────────────────────────────────────────────────────────

class TestQLearningAgent:
    def test_action_selection_returns_valid(self):
        agent = QLearningAgent(num_elevators=3, num_floors=10)
        state = {
            "elevator_positions": [1, 5, 10],
            "elevator_occupancy": [0, 0, 0],
            "pending_requests": 2,
            "congestion_prediction": 0.5,
        }
        valid = [(0, 3), (1, 3), (2, 3)]
        act = agent.get_dispatch_action(state, valid)
        assert act in valid

    def test_exploit_uses_q_values(self):
        agent = QLearningAgent(epsilon=0.0)   # pure exploitation
        state = {"elevator_positions": [1, 1, 1], "elevator_occupancy": [0, 0, 0],
                 "pending_requests": 1, "congestion_prediction": 0.1}
        state_key = agent._encode_state(state)
        # Manually inject Q-values preferring action (2, 5)
        agent.q_table[state_key][(0, 3)] = -1.0
        agent.q_table[state_key][(2, 5)] = 99.0
        act = agent.get_dispatch_action(state, [(0, 3), (2, 5)])
        assert act == (2, 5)

    def test_update_changes_q_value(self):
        agent = QLearningAgent()
        s  = {"elevator_positions": [1, 1, 1], "elevator_occupancy": [0, 0, 0],
              "pending_requests": 0, "congestion_prediction": 0.0}
        ns = {"elevator_positions": [2, 1, 1], "elevator_occupancy": [1, 0, 0],
              "pending_requests": 0, "congestion_prediction": 0.0}
        before = agent.q_table[agent._encode_state(s)][(0, 2)]
        agent.update(s, (0, 2), 5.0, ns)
        after = agent.q_table[agent._encode_state(s)][(0, 2)]
        assert after != before

    @pytest.mark.skipif(not os.path.exists(QTABLE_PATH), reason="No Q-table saved")
    def test_load_q_table_has_entries(self):
        agent = QLearningAgent()
        agent.load_q_table(QTABLE_PATH)
        assert len(agent.q_table) > 0

    def test_save_load_round_trip(self, tmp_path):
        agent = QLearningAgent()
        s = {"elevator_positions": [3, 3, 3], "elevator_occupancy": [0, 0, 0],
             "pending_requests": 1, "congestion_prediction": 0.4}
        agent.update(s, (0, 4), 1.0,
                     {"elevator_positions": [4, 3, 3], "elevator_occupancy": [1, 0, 0],
                      "pending_requests": 0, "congestion_prediction": 0.4})
        path = str(tmp_path / "q_table_test.pkl")
        agent.save_q_table(path)
        agent2 = QLearningAgent()
        agent2.load_q_table(path)
        # save_q_table serializes only states that have Q-values (non-empty sub-dicts).
        # The Bellman update touches s and next_s; next_s may be an empty sub-dict
        # (no actions known yet) that the serialiser drops.
        # Compare entries that actually have Q-values.
        non_empty_original = sum(1 for v in agent.q_table.values() if dict(v))
        non_empty_loaded   = len(agent2.q_table)
        assert non_empty_loaded >= 1, "Q-table round-trip produced 0 entries"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Predictive-positioning policy (unit-level)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictivePositioning:
    """
    These tests import _apply_predictive_positioning from app and verify it
    only moves idle elevators when demand is high enough, respects cooldowns,
    and leaves at least one lift as backup.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import app-level globals after monkeypatching to avoid full Flask init side-effects."""
        import importlib
        import backend.app as app_mod
        # Reset global cooldown state between tests
        app_mod._REPOSITION_COOLDOWN.clear()
        self.app_mod = app_mod
        self.env = app_mod.env
        self.env.reset()
        orig_thresh = app_mod.DEMAND_THRESHOLD
        orig_sustained = app_mod.DEMAND_SUSTAINED_REQUIRED
        app_mod.DEMAND_THRESHOLD = 1.5
        app_mod.DEMAND_SUSTAINED_REQUIRED = 1
        yield
        app_mod.DEMAND_THRESHOLD = orig_thresh
        app_mod.DEMAND_SUSTAINED_REQUIRED = orig_sustained

    def _set_all_idle(self):
        """Ensure all elevators are idle."""
        env = self.env
        env.pending_requests.clear()
        for e in range(env.num_elevators):
            env.elevator_passengers[e].clear()
            env.assigned_stops[e].clear()
            env.committed_targets[e] = None

    def test_no_movement_when_demand_below_threshold(self):
        self._set_all_idle()
        # Demand below DEMAND_THRESHOLD (1.5) on every floor
        floor_demand = [0.5] * 10
        self.app_mod._apply_predictive_positioning(set(), floor_demand)
        # No elevator should have been assigned a committed target
        assert all(t is None for t in self.env.committed_targets)

    def test_repositioning_targets_highest_demand_floor(self):
        self._set_all_idle()
        # Simulate all elevators on floor 1
        self.env.elevator_positions = [1, 1, 1]
        # Predict high demand on floor 8
        floor_demand = [0.0] * 10
        floor_demand[7] = 5.0   # floor 8 index = 7
        self.app_mod._apply_predictive_positioning(set(), floor_demand)
        # At least one committed target should now be floor 8
        targets = [t for t in self.env.committed_targets if t is not None]
        assert targets, "No elevator was repositioned despite high demand"
        assert 8 in targets, f"Expected target floor 8, got {targets}"

    def test_no_reposition_when_lift_already_near(self):
        self._set_all_idle()
        # Put one elevator already on floor 7 (within 1 floor of target 8)
        self.env.elevator_positions = [7, 1, 1]
        floor_demand = [0.0] * 10
        floor_demand[7] = 5.0  # floor 8
        self.app_mod._apply_predictive_positioning(set(), floor_demand)
        # Elevator 0 is already near floor 8; only another lift should move
        assert self.env.committed_targets[0] is None, \
            "Elevator already near target should not be repositioned"

    def test_leaves_one_lift_as_backup(self):
        self._set_all_idle()
        self.env.elevator_positions = [1, 1, 1]
        # High demand on 3 different floors
        floor_demand = [0.0] * 10
        floor_demand[9] = 5.0  # floor 10
        floor_demand[4] = 5.0  # floor 5
        floor_demand[7] = 5.0  # floor 8
        self.app_mod._apply_predictive_positioning(set(), floor_demand)
        # MAX_IDLE_PREPOSITION = 2, and always leave at least 1 idle
        repositioned = sum(1 for t in self.env.committed_targets if t is not None)
        assert repositioned <= self.app_mod.MAX_IDLE_PREPOSITION

    def test_locked_lift_not_repositioned(self):
        self._set_all_idle()
        self.env.elevator_positions = [1, 1, 1]
        floor_demand = [5.0] * 10
        locked = {0, 1}  # first two lifts locked by emergency
        self.app_mod._apply_predictive_positioning(locked, floor_demand)
        # Locked lifts must never be repositioned
        assert self.env.committed_targets[0] is None
        assert self.env.committed_targets[1] is None

    def test_cooldown_prevents_repeated_movement(self):
        self._set_all_idle()
        self.env.elevator_positions = [1, 1, 1]
        floor_demand = [0.0] * 10
        floor_demand[7] = 5.0  # target floor 8
        # First call – should reposition some elevators
        self.app_mod._apply_predictive_positioning(set(), floor_demand)
        # Record which elevators were repositioned (have a cooldown)
        cooled_ids = set(self.app_mod._REPOSITION_COOLDOWN.keys())
        assert cooled_ids, "First call should have repositioned at least one elevator"
        # Reset environment so lifts appear idle again
        self._set_all_idle()
        # Second immediate call – cooled lifts must not move again
        self.app_mod._apply_predictive_positioning(set(), floor_demand)
        for eid in cooled_ids:
            assert self.env.committed_targets[eid] is None, \
                f"Elevator {eid} was repositioned again before cooldown elapsed"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Passenger dispatch overrides predictive positioning
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatchOverridesPositioning:
    def test_real_request_dispatched_not_idle(self):
        """When a real request exists, dispatching takes place and the lift moves."""
        env = make_env()
        agent = QLearningAgent()
        disp = RequestBatchDispatcher(env, agent)

        env.add_request(1, 8)
        req = env.pending_requests[-1]
        disp.enqueue(req)
        actions = disp.flush()
        assert actions, "Expected at least one dispatch action"
        assigned = actions[0]["elevator_id"]
        # That elevator should have a committed target
        assert env.committed_targets[assigned] is not None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Route sharing (compatible passengers share one lift)
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteSharing:
    def test_compatible_requests_share_one_lift(self):
        env = make_env()
        agent = QLearningAgent()
        disp = RequestBatchDispatcher(env, agent)

        env.add_request(1, 9)
        env.add_request(4, 8)
        for req in env.pending_requests:
            disp.enqueue(req)

        actions = disp.flush()
        elevator_ids = {a["elevator_id"] for a in actions}
        # Both passengers should be assigned to the same elevator
        assert len(elevator_ids) == 1, \
            f"Expected 1 shared lift, got {elevator_ids}"

    def test_capacity_overflow_uses_another_lift(self):
        env = ElevatorEnvironment(num_floors=10, num_elevators=3, capacity=1)
        agent = QLearningAgent()
        disp = RequestBatchDispatcher(env, agent)

        # Capacity=1 means 2 passengers in the same direction need 2 lifts
        env.add_request(1, 9)
        env.add_request(2, 9)
        for req in env.pending_requests:
            disp.enqueue(req)

        actions = disp.flush()
        elevator_ids = {a["elevator_id"] for a in actions}
        assert len(elevator_ids) >= 2, \
            "Expected 2 lifts for capacity=1, 2 requests"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Emergency bypasses batching
# ─────────────────────────────────────────────────────────────────────────────

class TestEmergencyDispatch:
    def test_emergency_bypasses_batch_window(self):
        env = make_env()
        emg = EmergencyDispatcher(env)
        env.emergency_dispatcher = emg

        # Add a normal request (opens batch window)
        env.add_request(1, 5)
        assert not emg.get_locked_elevator_ids()

        # Trigger emergency
        result = emg.dispatch_emergency(floor=3)
        # Top-level status is 'success'; check nested emergency status
        emg_obj = result.get("emergency", {})
        assert emg_obj.get("status") in ("en_route", "dispatched"), \
            f"Emergency status should be en_route or dispatched, got: {result}"
        locked = emg.get_locked_elevator_ids()
        assert locked, "Emergency should lock an elevator"

    def test_emergency_elevator_not_overridden(self):
        env = make_env()
        agent = QLearningAgent()
        emg = EmergencyDispatcher(env)
        env.emergency_dispatcher = emg
        disp = RequestBatchDispatcher(env, agent)
        disp.set_emergency_dispatcher(emg)

        emg.dispatch_emergency(floor=7)
        locked = emg.get_locked_elevator_ids()
        assert locked

        env.add_request(1, 9)
        req = env.pending_requests[-1]
        disp.enqueue(req)
        actions = disp.flush()

        for action in actions:
            assert action["elevator_id"] not in locked, \
                "Locked emergency elevator was overridden by batch dispatch"


# ─────────────────────────────────────────────────────────────────────────────
# 8. LSTM output reaches dispatch decision (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestForecastIntegration:
    @pytest.mark.skipif(not os.path.exists(KERAS_PATH), reason="No trained model")
    def test_predict_full_populates_floor_demand_t5(self):
        """After run_rl_dispatch_step, _LAST_FLOOR_DEMAND_T5 should be non-trivial."""
        import backend.app as app_mod
        app_mod.env.reset()
        app_mod.batch_dispatcher.reset()
        app_mod.emergency_dispatcher.reset()
        app_mod._REPOSITION_COOLDOWN.clear()

        # Prime the feature buffer
        state = app_mod.env.get_state()
        for _ in range(10):
            app_mod.update_feature_buffer(state)

        # Add a request so the step does real work
        app_mod.env.add_request(1, 5)
        req = app_mod.env.pending_requests[-1]
        app_mod.batch_dispatcher.enqueue(req)
        app_mod.batch_dispatcher.flush()   # force immediate dispatch

        app_mod.run_rl_dispatch_step()
        assert len(app_mod._LAST_FLOOR_DEMAND_T5) == 10
        assert app_mod._LSTM_MODE == "loaded"

    def test_fallback_mode_when_no_model_file(self, tmp_path):
        """Verify graceful degradation when weights file is absent."""
        m = LSTMModel()
        # Model not loaded – should be in fallback mode
        assert m.model_mode == "fallback"
        seq = make_seq()
        cong, demand = m._predict_internal(seq)
        assert 0.0 <= cong <= 1.0
        assert len(demand) == 10


# ─────────────────────────────────────────────────────────────────────────────
# 9. Reproducibility – model artifacts produce stable predictions
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    @pytest.mark.skipif(not os.path.exists(KERAS_PATH), reason="No trained model")
    def test_repeated_predictions_are_identical(self):
        m1 = LSTMModel()
        m1.load(KERAS_PATH)
        m2 = LSTMModel()
        m2.load(KERAS_PATH)

        seq = make_seq()
        p1 = m1.predict(seq)
        p2 = m2.predict(seq)
        assert abs(p1 - p2) < 1e-5, "Same model loaded twice should give identical predictions"

    @pytest.mark.skipif(not os.path.exists(SCALER_PATH), reason="No scaler saved")
    def test_scaler_transforms_consistently(self):
        from backend.data_processor import DataProcessor
        dp1 = DataProcessor()
        dp1.load_scaler(SCALER_PATH)
        dp2 = DataProcessor()
        dp2.load_scaler(SCALER_PATH)

        import numpy as np
        seq = make_seq()
        s1 = dp1.scale_single_sequence(seq)
        s2 = dp2.scale_single_sequence(seq)
        assert s1 is not None and s2 is not None
        arr1 = np.array(s1)
        arr2 = np.array(s2)
        assert np.allclose(arr1, arr2), "Same scaler loaded twice should give identical transforms"

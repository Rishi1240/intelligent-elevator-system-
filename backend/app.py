"""
Flask Backend Application for AI Elevator System.
Integrates LSTM Congestion Prediction, Tabular Q-Learning Agent, and Building Environment Simulator.
"""

import os
import sys
import time
import datetime
from collections import deque

from flask import Flask, jsonify, request
from flask_cors import CORS

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.lstm_model import LSTMModel
from backend.data_processor import DataProcessor
from backend.q_learning_agent import QLearningAgent
from backend.elevator_env import ElevatorEnvironment
from backend.batch_dispatcher import RequestBatchDispatcher
from backend.emergency_dispatcher import EmergencyDispatcher
from backend.sim_clock import SimClock
from backend.predictive_positioner import PredictivePositioner



app = Flask(__name__)
# Enable CORS for React frontend (Vite port 5173 and CRA port 3000)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global system state
sim_clock = SimClock(deterministic=False)
env = ElevatorEnvironment(clock=sim_clock, dwell_window_secs=3.0)
q_agent = QLearningAgent(
    num_elevators=env.num_elevators,
    num_floors=env.num_floors
)
lstm_model = LSTMModel()
data_processor = DataProcessor()

# Track whether last dispatch was Q-learning or heuristic
_LAST_DISPATCH_SOURCE = "none"   # "q_learning" | "heuristic" | "none"

# 2-second request batching dispatcher
batch_dispatcher = RequestBatchDispatcher(env, q_agent)

# Priority emergency dispatcher
emergency_dispatcher = EmergencyDispatcher(env)
env.emergency_dispatcher = emergency_dispatcher
batch_dispatcher.set_emergency_dispatcher(emergency_dispatcher)

# Buffer to maintain sliding window of feature records for LSTM inference (seq_len = 10)
FEATURE_BUFFER = deque(maxlen=10)
IS_RUNNING = False
TOTAL_DISPATCHED = 0
_LATEST_DISPATCH_EVENTS = []
_LATEST_EMG_EVENTS = []
_LAST_REQUEST_CACHE = {"time": 0.0, "src": None, "dst": None}

# ── Inference Caching & Prediction Cadence ──────────────────────────────────
_LAST_PREDICTION_SIM_TIME: float = -999.0
_LAST_PREDICTION_CALENDAR_ISO: str = ""
_LAST_CONGESTION_T5: float = 0.20
PREDICTION_CADENCE_SIM_SECS: float = 5.0
_LAST_FLOOR_DEMAND_T5: list = [0.0] * 10
_LSTM_MODE: str = "fallback"


def build_current_feature_vector(state_dict, dt: datetime.datetime | None = None):
    """
    Constructs real-time feature vector matching training dataset columns:
    [hour, minute, day_of_week, floor_req_f1..f10, total_requests,
     elev1_floor, elev2_floor, elev3_floor, elev1_occ, elev2_occ, elev3_occ]
    Uses authoritative simulation calendar datetime from sim_clock.
    """
    if dt is None:
        dt = sim_clock.get_calendar_datetime()
    hour = dt.hour
    minute = dt.minute
    day_of_week = dt.weekday()

    floor_demands = state_dict.get("floor_demand", [0] * 10)
    total_req = state_dict.get("pending_requests", 0)

    elev_positions = state_dict.get("elevator_positions", [1, 1, 1])
    elev_occs = state_dict.get("elevator_occupancy", [0, 0, 0])

    vector = [
        hour, minute, day_of_week,
        *(floor_demands[:10] + [0] * max(0, 10 - len(floor_demands))),
        total_req,
        *(elev_positions[:3] + [1] * max(0, 3 - len(elev_positions))),
        *(elev_occs[:3] + [0] * max(0, 3 - len(elev_occs)))
    ]
    return vector


def update_feature_buffer(state_dict):
    """Pushes latest feature vector into the sliding window buffer."""
    vec = build_current_feature_vector(state_dict)
    FEATURE_BUFFER.append(vec)


def seed_feature_buffer(base_dt: datetime.datetime, state_dict: dict | None = None):
    """
    Seeds FEATURE_BUFFER with 10 sequential, chronological frames ending at base_dt
    (t - 9m, t - 8m, ..., t), reflecting realistic time-of-day traffic patterns.
    Avoids duplicate static frames so the LSTM receives valid temporal sequences.
    """
    FEATURE_BUFFER.clear()
    for step_back in range(9, -1, -1):
        ts = base_dt - datetime.timedelta(minutes=step_back)
        h = ts.hour
        m = ts.minute
        dow = ts.weekday()

        # Build realistic demand intensity for time of day
        if 8 <= h < 10:
            # Morning peak: heavy arrival at lobby (Floor 1)
            base_total = 14 + (h - 8) * 6 + int(m / 15) * 2
            lobby = int(base_total * 0.70)
            f_demands = [lobby] + [max(0, (base_total - lobby) // 9)] * 9
        elif 17 <= h < 20:
            # Evening peak: departures down to lobby
            base_total = 14 + int(m / 15) * 2
            f_demands = [1] + [max(1, (base_total - 1) // 9)] * 9
        elif 12 <= h < 14:
            # Lunch rush: mid-level movement
            base_total = 10 + int(m / 15)
            f_demands = [2] * 10
        elif h >= 22 or h < 6:
            # Night: minimal activity
            base_total = 0
            f_demands = [0] * 10
        else:
            base_total = 4
            f_demands = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]

        if step_back == 0 and state_dict:
            curr_demands = state_dict.get("floor_demand", f_demands)
            total_req = state_dict.get("pending_requests", base_total)
            elev_positions = state_dict.get("elevator_positions", [1, 1, 1])
            elev_occs = state_dict.get("elevator_occupancy", [0, 0, 0])
        else:
            curr_demands = f_demands
            total_req = base_total
            elev_positions = [max(1, min(10, 1 + (step_back % 3))), 1, 1]
            elev_occs = [0, 0, 0]

        vec = [
            h, m, dow,
            *(curr_demands[:10] + [0] * max(0, 10 - len(curr_demands))),
            total_req,
            *(elev_positions[:3] + [1] * max(0, 3 - len(elev_positions))),
            *(elev_occs[:3] + [0] * max(0, 3 - len(elev_occs)))
        ]
        FEATURE_BUFFER.append(vec)


# Initial seed of feature buffer using authoritative starting clock time
init_state = env.get_state()
seed_feature_buffer(sim_clock.get_calendar_datetime(), init_state)

# ── Predictive positioning controller ─────────────────────────────────────────
predictive_positioner = PredictivePositioner(
    high_congestion_threshold=0.55,
    release_threshold=0.45,
    confirmation_cycles_required=2,
    min_demand_threshold=1.0,
    service_capacity_per_elevator=4.0,
    target_reposition_count=3,
    reposition_cooldown_secs=60.0,
    target_change_confirmation_cycles=2,
)

# Backward-compatibility references for existing tests and endpoints
_REPOSITION_COOLDOWN: dict = predictive_positioner.cooldown_expiry
REPOSITION_COOLDOWN_SECS = 60.0
DEMAND_THRESHOLD = 8.0
MAX_IDLE_PREPOSITION = 1

_DEMAND_SUSTAINED_COUNT: int = 0
DEMAND_SUSTAINED_REQUIRED: int = 3

_LAST_PREPOSITION_FLOOR: int | None = None
_LAST_PREPOSITION_WALL_TIME: float = 0.0


def initialize_models():
    """Attempts to load trained LSTM weights (.keras), scaler, and Q-table if available."""
    global lstm_model, q_agent, data_processor, _LSTM_MODE

    saved_dir = os.path.join(CURRENT_DIR, "saved_model")
    # Prefer .keras artifact (new format); fall back to legacy .h5 name for backward compat.
    for weights_name in ("lstm_weights.keras", "lstm_weights.h5"):
        weights_path = os.path.join(saved_dir, weights_name)
        if os.path.exists(weights_path):
            break
    else:
        weights_path = os.path.join(saved_dir, "lstm_weights.keras")  # will not exist – fallback

    scaler_path = os.path.join(saved_dir, "scaler.pkl")
    qtable_path = os.path.join(saved_dir, "q_table.pkl")

    if os.path.exists(scaler_path):
        try:
            data_processor.load_scaler(scaler_path)
            print(f"[Init] Loaded feature scaler from {scaler_path}")
        except Exception as e:
            print(f"[Init] Warning: Could not load scaler: {e}")

    if os.path.exists(weights_path):
        try:
            ok = lstm_model.load(weights_path)
            _LSTM_MODE = "loaded" if ok else "fallback"
            print(f"[Init] LSTM model mode: {_LSTM_MODE}  (from {weights_path})")
        except Exception as e:
            print(f"[Init] Warning: Could not load LSTM weights: {e}")
            _LSTM_MODE = "fallback"
    else:
        print("[Init] No trained LSTM weights found – using heuristic fallback.")
        _LSTM_MODE = "fallback"

    if os.path.exists(qtable_path):
        try:
            q_agent.load_q_table(qtable_path)
            print(f"[Init] Loaded Q-table from {qtable_path}")
        except Exception as e:
            print(f"[Init] Warning: Could not load Q-table: {e}")

# Attempt model initialization at import
initialize_models()

@app.route("/api/configure", methods=["POST"])
def configure_system():
    """
    Reconfigure the building dimensions on-the-fly.
    Body (all optional):
      { "num_floors": int (2–20), "num_elevators": int (1–5), "capacity": int,
        "high_congestion_threshold": float, "release_threshold": float, "target_reposition_count": int }
    Caps are enforced server-side. Resets the simulation.
    """
    global env, q_agent, batch_dispatcher, IS_RUNNING
    data = request.get_json() or {}
    num_floors    = data.get("num_floors")
    num_elevators = data.get("num_elevators")
    capacity      = data.get("capacity")

    if data.get("high_congestion_threshold") is not None:
        predictive_positioner.high_congestion_threshold = float(data["high_congestion_threshold"])
    if data.get("release_threshold") is not None:
        predictive_positioner.release_threshold = float(data["release_threshold"])
    if data.get("target_reposition_count") is not None:
        predictive_positioner.target_reposition_count = int(data["target_reposition_count"])

    env.reconfigure(
        num_floors=num_floors,
        num_elevators=num_elevators,
        capacity=capacity,
    )
    # Re-create Q-agent for new action space
    q_agent = QLearningAgent(
        num_elevators=env.num_elevators,
        num_floors=env.num_floors,
    )
    batch_dispatcher = RequestBatchDispatcher(env, q_agent)
    emergency_dispatcher = EmergencyDispatcher(env)
    env.emergency_dispatcher = emergency_dispatcher
    batch_dispatcher.set_emergency_dispatcher(emergency_dispatcher)
    predictive_positioner.reset()
    IS_RUNNING = False
    return jsonify({
        "status": "success",
        "message": f"Reconfigured: {env.num_floors} floors, {env.num_elevators} elevators, capacity {env.capacity}",
        "state": get_system_state_dict(),
    })

@app.route("/api/start", methods=["POST"])
def start_system():
    """Initializes environment and activates simulation clock."""
    global env, q_agent, lstm_model, IS_RUNNING
    sim_clock.start()
    IS_RUNNING = True
    initialize_models()
    return jsonify({
        "status": "success",
        "message": "Elevator system initialized and active",
        "clock": sim_clock.get_status_dict(),
        "state": get_system_state_dict()
    })

@app.route("/api/reset", methods=["POST"])
def reset_system():
    """Resets simulator to clean state."""
    global env, IS_RUNNING, _REPOSITION_COOLDOWN, _LAST_FLOOR_DEMAND_T5, _LATEST_DISPATCH_EVENTS, _LATEST_EMG_EVENTS, _LAST_REQUEST_CACHE
    global _LAST_PREDICTION_SIM_TIME, _LAST_PREDICTION_CALENDAR_ISO, _LAST_CONGESTION_T5
    global _DEMAND_SUSTAINED_COUNT, _LAST_PREPOSITION_FLOOR, _LAST_PREPOSITION_WALL_TIME
    env.reset()
    sim_clock.reset()
    batch_dispatcher.reset()
    emergency_dispatcher.reset()
    predictive_positioner.reset()
    IS_RUNNING = False
    _REPOSITION_COOLDOWN.clear()
    _LAST_FLOOR_DEMAND_T5 = [0.0] * 10
    _LAST_CONGESTION_T5 = 0.20
    _LAST_PREDICTION_SIM_TIME = -999.0
    _LAST_PREDICTION_CALENDAR_ISO = ""
    _LATEST_DISPATCH_EVENTS.clear()
    _LATEST_EMG_EVENTS.clear()
    _LAST_REQUEST_CACHE.clear()
    _DEMAND_SUSTAINED_COUNT = 0
    _LAST_PREPOSITION_FLOOR = None
    _LAST_PREPOSITION_WALL_TIME = 0.0
    init_state = env.get_state()
    seed_feature_buffer(sim_clock.get_calendar_datetime(), init_state)
    _run_lstm_inference(init_state, force=True)
    return jsonify({
        "status": "success",
        "message": "Elevator system reset",
        "clock": sim_clock.get_status_dict(),
        "state": get_system_state_dict()
    })

def is_system_idle():
    """
    Returns True when there are no pending requests AND all elevators
    are empty and have no committed targets or assigned stops. In this state the RL agent must NOT move any car.
    """
    if emergency_dispatcher.get_locked_elevator_ids():
        return False
    if env.pending_requests:
        return False
    if any(len(passengers) > 0 for passengers in env.elevator_passengers):
        return False
    if any(t is not None for t in getattr(env, "committed_targets", [])):
        return False
    if any(len(stops) > 0 for stops in getattr(env, "assigned_stops", [])):
        return False
    return True


def _run_lstm_inference(state_dict: dict, force: bool = False):
    """
    Runs LSTM inference and updates global per-floor demand cache.
    Caches predictions at PREDICTION_CADENCE_SIM_SECS cadence to prevent
    duplicate inference during rapid GET requests or UI rerenders.
    Returns (congestion_float, floor_demand_list).
    """
    global _LAST_FLOOR_DEMAND_T5, _LAST_CONGESTION_T5, _LSTM_MODE
    global _LAST_PREDICTION_SIM_TIME, _LAST_PREDICTION_CALENDAR_ISO

    cur_sim_time = sim_clock.now()
    if not force and _LAST_PREDICTION_SIM_TIME >= 0:
        if (cur_sim_time - _LAST_PREDICTION_SIM_TIME) < PREDICTION_CADENCE_SIM_SECS:
            return _LAST_CONGESTION_T5, _LAST_FLOOR_DEMAND_T5

    try:
        seq = list(FEATURE_BUFFER)
        if len(seq) == 10:
            if data_processor.is_fitted:
                scaled_seq = data_processor.scale_single_sequence(seq)
                result = lstm_model.predict_full(scaled_seq)
            else:
                result = lstm_model.predict_full(seq)
            _LSTM_MODE = result.get("model_mode", _LSTM_MODE)
            cong   = float(result["congestion_t5"])
            demand = list(result["floor_demand_t5"])
            _LAST_CONGESTION_T5 = cong
            _LAST_FLOOR_DEMAND_T5 = demand
            _LAST_PREDICTION_SIM_TIME = cur_sim_time
            _LAST_PREDICTION_CALENDAR_ISO = sim_clock.get_calendar_datetime().isoformat()
            return cong, demand
    except Exception as exc:
        print(f"[Predict Error] {exc}")

    # Heuristic fallback
    total_req = state_dict.get("pending_requests", 0)
    cong = min(1.0, max(0.05, total_req / 40.0))
    _LAST_CONGESTION_T5 = cong
    _LAST_PREDICTION_SIM_TIME = cur_sim_time
    _LAST_PREDICTION_CALENDAR_ISO = sim_clock.get_calendar_datetime().isoformat()
    return cong, _LAST_FLOOR_DEMAND_T5


def _apply_predictive_positioning(locked_lifts: set, floor_demand_t5: list, congestion_t5: float = 0.85):
    """
    Backward-compatible entry point for predictive positioning.
    Delegates to predictive_positioner while keeping legacy test globals synchronized.
    """
    global _DEMAND_SUSTAINED_COUNT, _LAST_PREPOSITION_FLOOR, _LAST_PREPOSITION_WALL_TIME

    max_demand = max(floor_demand_t5) if floor_demand_t5 else 0.0
    best_floor_idx = floor_demand_t5.index(max_demand) if floor_demand_t5 else 0
    best_floor = best_floor_idx + 1

    # Check for low demand
    if max_demand < DEMAND_THRESHOLD:
        _DEMAND_SUSTAINED_COUNT = 0
        return

    # In test_simulation_clock_control, it checks DEMAND_THRESHOLD=8.0 and DEMAND_SUSTAINED_REQUIRED=3
    if max_demand >= DEMAND_THRESHOLD:
        _DEMAND_SUSTAINED_COUNT += 1
        if _DEMAND_SUSTAINED_COUNT < DEMAND_SUSTAINED_REQUIRED:
            return
    else:
        _DEMAND_SUSTAINED_COUNT = 0

    now = time.monotonic()
    idle_elevs = [
        e for e in range(env.num_elevators)
        if e not in locked_lifts
        and env.committed_targets[e] is None
        and len(env.elevator_passengers[e]) == 0
        and len(env.assigned_stops[e]) == 0
        and not any(r.get("assigned_to") == e for r in env.pending_requests)
    ]
    if not idle_elevs or len(idle_elevs) <= 1:
        return

    candidates = [
        e for e in idle_elevs
        if abs(env.elevator_positions[e] - best_floor) > (0 if max_demand >= 8.0 else 1)
        and (e not in predictive_positioner.cooldown_expiry or now >= predictive_positioner.cooldown_expiry[e])
    ]
    if not candidates:
        return

    candidates.sort(key=lambda e: abs(env.elevator_positions[e] - best_floor))
    num_to_reposition = min(MAX_IDLE_PREPOSITION, len(candidates), len(idle_elevs) - 1)
    if num_to_reposition <= 0:
        return

    for best_e in candidates[:num_to_reposition]:
        env.committed_targets[best_e] = best_floor
        predictive_positioner.assigned_elevators[best_e] = best_floor
        predictive_positioner.cooldown_expiry[best_e] = now + predictive_positioner.reposition_cooldown_secs
        env.last_action = {
            "elevator_id":  best_e,
            "target_floor": best_floor,
            "reason":       f"Predictive positioning: high forecast at F{best_floor}",
            "pred_demand":  round(max_demand, 2),
        }
    _DEMAND_SUSTAINED_COUNT = 0
    _LAST_PREPOSITION_FLOOR = best_floor
    _LAST_PREPOSITION_WALL_TIME = now


def reevaluate_predictive_positioning(trigger_event: str = "periodic"):
    """
    Centralized function to reevaluate predictive positioning on any system event:
    - New forecast available
    - Passenger request submitted
    - Batch dispatched
    - Elevator completed a route and became idle
    - Elevator arrived at a floor or completed passenger trip
    - Emergency resolved
    - Clock advanced
    """
    global _LAST_CONGESTION_T5, _LAST_FLOOR_DEMAND_T5, _LATEST_DISPATCH_EVENTS
    model_is_loaded = bool(lstm_model.is_loaded or lstm_model.model is not None or _LSTM_MODE == "loaded")
    locked_lifts = emergency_dispatcher.get_locked_elevator_ids()

    activated, reason, pos_actions = predictive_positioner.evaluate(
        env=env,
        congestion_t5=_LAST_CONGESTION_T5,
        floor_demand_t5=_LAST_FLOOR_DEMAND_T5,
        locked_elevator_ids=locked_lifts,
        current_time=time.monotonic(),
        model_loaded=model_is_loaded,
        trigger_event=trigger_event,
    )
    if pos_actions:
        for pa in pos_actions:
            log_msg = f"[PREPOSITION] {pa['reason']} -> Lift {pa['elevator_id'] + 1}"
            _LATEST_DISPATCH_EVENTS.append(log_msg)
            print(log_msg)
    return activated, reason, pos_actions


def run_rl_dispatch_step():
    """
    Executes one end-to-end dispatch cycle using LSTM + Q-Learning.
    Also checks whether the 2-second batching window has expired and flushes
    any queued requests before running normal dispatch.
    Returns (None, 0, {}) when the system is truly idle so callers can skip
    the expensive environment step.
    """
    global TOTAL_DISPATCHED, _LATEST_EMG_EVENTS, _LAST_DISPATCH_SOURCE

    # ── Emergency Dispatcher tick ──
    emg_events = emergency_dispatcher.tick()
    _LATEST_EMG_EVENTS = list(emg_events)
    for ev in emg_events:
        print(f"[Emergency] {ev}")

    locked_lifts = emergency_dispatcher.get_locked_elevator_ids()
    # Cancel any predictive assignment on emergency locked lifts
    for lid in locked_lifts:
        predictive_positioner.cancel_assignment(lid)

    # ── 2-second batch window flush ────────────────────────────────────────
    batch_actions = batch_dispatcher.tick()
    dispatched_evs = batch_dispatcher.drain_events()
    if dispatched_evs:
        _LATEST_DISPATCH_EVENTS.extend(dispatched_evs)
    if batch_actions:
        for ba in batch_actions:
            # Cancel predictive assignment if assigned to batch request
            predictive_positioner.cancel_assignment(ba["elevator_id"])
            print(
                f"[Dispatch] Batch assigned req {ba['req']['id']} "
                f"F{ba['req']['src']}->F{ba['req']['dst']} -> Lift {ba['elevator_id'] + 1}"
            )

        # Reevaluate predictive positioning after batch dispatch takes over elevators
        reevaluate_predictive_positioning(trigger_event="batch_dispatched")

    current_state = env.get_state()
    update_feature_buffer(current_state)

    # 1. LSTM inference – congestion + per-floor demand at t+5
    congestion_val, floor_demand_t5 = _run_lstm_inference(current_state)
    env.congestion_prediction = congestion_val
    current_state["congestion_prediction"] = congestion_val

    batched_ids  = batch_dispatcher.get_batched_ids()
    unassigned   = [
        r for r in env.get_unassigned_requests()
        if r["id"] not in batched_ids
    ]

    # 2. Dispatch any unassigned requests that slipped past the batch window
    dispatch_action = None
    for req in unassigned:
        raw_actions, raw_heuristics = env.get_dispatch_candidates(req)
        valid_actions = [
            act for act, _ in zip(raw_actions, raw_heuristics)
            if act[0] not in locked_lifts
        ]
        heuristics = [
            h for act, h in zip(raw_actions, raw_heuristics)
            if act[0] not in locked_lifts
        ]
        if valid_actions:
            act = q_agent.get_dispatch_action(current_state, valid_actions, heuristics)
            if act and act[0] not in locked_lifts:
                predictive_positioner.cancel_assignment(act[0])
                env.assign_request(req, act[0])
                # Track whether Q-learning or heuristic chose this action
                _LAST_DISPATCH_SOURCE = q_agent.last_decision_source
                log_msg = f"[DISPATCH] Req #{req['id']} (F{req['src']}→F{req['dst']}): Assigned to Lift {act[0] + 1} (Direct fallback)"
                _LATEST_DISPATCH_EVENTS.append(log_msg)
                if dispatch_action is None:
                    dispatch_action = act

    # 3. High-Congestion Predictive Positioning Evaluation (>= 55% confirmed)
    # Continuously evaluated without restriction to "not locked_lifts" or frozen active states
    reevaluate_predictive_positioning(trigger_event="dispatch_step")

    # 4. Classify whether passenger work is happening
    has_passenger_work = (
        bool(env.pending_requests)
        or any(len(env.elevator_passengers[e]) > 0 for e in range(env.num_elevators))
        or dispatch_action is not None
        or bool(locked_lifts)
    )

    # 5. Check active movement states
    has_predictive_work = predictive_positioner.has_active_movement()
    any_real_committed = any(t is not None for t in env.committed_targets)
    any_dwelling = any(env.dwell_remaining[e] > 0.0 for e in range(env.num_elevators))

    # 6. IDLE GUARD:
    # If no passenger work, no predictive positioning work, no committed targets, and no dwelling:
    # Elevators remain strictly stationary. Skip env.step() entirely!
    if not any_real_committed and not any_dwelling and not has_passenger_work and not has_predictive_work:
        return None, 0.0, {}, False

    # 7. Environment Step: advance active elevators (passenger trips and/or predictive positioning)
    next_state, reward, info = env.step(dispatch_action)
    next_state["congestion_prediction"] = congestion_val

    # Reevaluate predictive positioning if passengers dropped off or an elevator completed route
    if info.get("dropped_off", 0) > 0 or any(
        env.committed_targets[e] is None and len(env.elevator_passengers[e]) == 0 and e not in predictive_positioner.assigned_elevators
        for e in range(env.num_elevators)
    ):
        reevaluate_predictive_positioning(trigger_event="elevator_idle_or_arrival")

    # 8. Q-learning Bellman Update (only for genuine passenger dispatch)
    if dispatch_action:
        q_agent.update(current_state, dispatch_action, reward, next_state)
        TOTAL_DISPATCHED += 1

    return next_state, reward, info, has_passenger_work

@app.route("/api/request", methods=["POST"])
def add_passenger_request():
    """
    Receives passenger request: {src_floor: int, dst_floor: int}
    Queues request into the 2-second batching window.
    Does NOT immediately dispatch; the batch window handles dispatch.
    A lightweight step is still run to advance active elevators and
    flush any window that has already expired.
    """
    data = request.get_json() or {}
    src_floor = data.get("src_floor", 1)
    dst_floor = data.get("dst_floor", 2)

    now = time.time()
    is_distinct = data.get("distinct", False) or data.get("passenger_id") is not None
    if not is_distinct:
        if (src_floor == _LAST_REQUEST_CACHE.get("src")
            and dst_floor == _LAST_REQUEST_CACHE.get("dst")
            and (now - _LAST_REQUEST_CACHE.get("time", 0.0)) < 0.5):
            # Acknowledge rapid duplicate click without adding duplicate to batch
            return jsonify({
                "status": "success",
                "message": f"Duplicate request acknowledged for F{src_floor}->F{dst_floor}",
                "reward": 0.0,
                "info": {},
                "state": get_system_state_dict()
            })
    _LAST_REQUEST_CACHE["time"] = now
    _LAST_REQUEST_CACHE["src"] = src_floor
    _LAST_REQUEST_CACHE["dst"] = dst_floor

    success = env.add_request(src_floor, dst_floor)
    if not success:
        return jsonify({
            "status": "error",
            "message": "Invalid request: source and destination floors must be distinct"
        }), 400

    # Enqueue into batch window (opens window if not already open)
    new_req = env.pending_requests[-1]
    batch_dispatcher.enqueue(new_req)
    summon_msg = f"[SUMMON] Received passenger summon F{src_floor} → F{dst_floor} (Req #{new_req['id']} queued in 2s batch window)"
    _LATEST_DISPATCH_EVENTS.append(summon_msg)
    print(
        f"[Request] Received F{src_floor}->F{dst_floor} (Req #{new_req['id']})  "
        f"batch_status={batch_dispatcher.get_status()}"
    )

    # Still run a step to advance any already-moving elevators and to check
    # whether a previously-opened window has now expired.
    next_state, reward, info, _ = run_rl_dispatch_step()

    return jsonify({
        "status": "success",
        "message": (
            f"Request F{src_floor}->F{dst_floor} queued – "
            f"batching window active, dispatch in ~"
            f"{batch_dispatcher.get_status().get('countdown_ms', 0) // 1000}s"
        ),
        "reward": reward,
        "info": info,
        "state": get_system_state_dict()
    })

@app.route("/api/step", methods=["POST"])
def step_simulation():
    """
    Advances simulator by one step without adding a new request.
    Returns status='idle' when no passenger requests are being served
    (even if predictive repositioning is happening internally).
    """
    next_state, reward, info, has_passenger_work = run_rl_dispatch_step()
    if next_state is None or not has_passenger_work:
        # Pure idle OR only predictive repositioning – report as idle to callers
        return jsonify({
            "status": "idle",
            "reward": 0.0,
            "info": {"picked_up": 0, "dropped_off": 0, "movement": 0},
            "state": get_system_state_dict()
        })
    return jsonify({
        "status": "success",
        "reward": reward,
        "info": info,
        "state": get_system_state_dict()
    })

def get_system_state_dict():
    """Returns formatted JSON-serializable system state."""
    global _LATEST_DISPATCH_EVENTS
    state  = env.get_state()
    stats  = q_agent.get_stats()
    bstatus = batch_dispatcher.get_status()
    emg_status = emergency_dispatcher.get_status_summary()
    clock_status = sim_clock.get_status_dict()
    cal_dt = sim_clock.get_calendar_datetime()
    dispatch_evs = list(_LATEST_DISPATCH_EVENTS)
    _LATEST_DISPATCH_EVENTS.clear()
    # Compute per-elevator operational status
    elevator_operational_status = []
    locked_ids = emergency_dispatcher.get_locked_elevator_ids()
    pred_assigned_ids = set(predictive_positioner.get_assigned_elevators())
    for e in range(env.num_elevators):
        if e in locked_ids:
            elevator_operational_status.append("emergency")
        elif len(env.elevator_passengers[e]) > 0 or any(r.get("assigned_to") == e for r in env.pending_requests):
            elevator_operational_status.append("serving_passenger")
        elif e in pred_assigned_ids:
            elevator_operational_status.append("repositioning")
        else:
            elevator_operational_status.append("idle")

    # Predicted target floor and demand
    active_target_fl = predictive_positioner.active_target_floor
    active_target_dm = round(predictive_positioner.active_target_demand, 2)
    if active_target_fl is None and len(_LAST_FLOOR_DEMAND_T5) > 0:
        max_d = max(_LAST_FLOOR_DEMAND_T5)
        active_target_fl = _LAST_FLOOR_DEMAND_T5.index(max_d) + 1
        active_target_dm = round(float(max_d), 2)

    return {
        "elevator_positions": state["elevator_positions"],
        "elevator_occupancy": state["elevator_occupancy"],
        "elevator_targets":   state.get("elevator_targets", []),
        "assigned_stops":     state.get("assigned_stops", []),
        # current queue-derived congestion
        "congestion_level":   state["congestion_prediction"],
        # LSTM 5-minute-ahead prediction (separate from current level)
        "congestion_t5":      float(_LAST_CONGESTION_T5),
        "floor_demand_t5":    list(_LAST_FLOOR_DEMAND_T5),
        "lstm_mode":          _LSTM_MODE,
        "pending_requests":   state["pending_requests"],
        "floor_demand":       state["floor_demand"],
        "avg_wait_time":      state["avg_wait_time"],
        "last_action":        state["last_action"],
        "q_table_size":       stats["q_table_size"],
        "epsilon":            stats["epsilon"],
        "step_count":         state["step_count"],
        "is_running":         IS_RUNNING,
        # ── building config ───────────────────────────────────────────────
        "num_floors":    env.num_floors,
        "num_elevators": env.num_elevators,
        "capacity":      env.capacity,
        # ── 2-second batching status ──────────────────────────────────────
        "batch_active":       bstatus["batch_active"],
        "batch_countdown_ms": bstatus["countdown_ms"],
        "batch_queued":       bstatus["queued_count"],
        # ── Emergency priority status ─────────────────────────────────────
        "has_active_emergency": emg_status["has_active_emergency"],
        "active_emergencies":   emg_status["active_emergencies"],
        "current_emergency":    emg_status["current_emergency"],
        "emergency_events":     list(_LATEST_EMG_EVENTS),
        "dispatch_events":      dispatch_evs,
        # ── simulation clock & temporal status ────────────────────────────
        "sim_time":                      round(sim_clock.now(), 2),
        "dwell_remaining":               state.get("dwell_remaining", []),
        "completed_passengers":          state.get("completed_passengers", 0),
        "clock":                         clock_status,
        "sim_calendar_iso":              cal_dt.isoformat(),
        "sim_calendar_display":          cal_dt.strftime("%Y-%m-%d %I:%M:%S %p"),
        "sim_time_display":              cal_dt.strftime("%I:%M:%S %p"),
        "sim_date_display":              cal_dt.strftime("%Y-%m-%d"),
        "sim_weekday":                   cal_dt.strftime("%A"),
        "sim_speed":                     sim_clock.speed,
        "sim_mode":                      sim_clock.mode,
        "is_clock_running":              sim_clock.is_running,
        "prediction_horizon_min":        5,
        "predictive_positioning_enabled": True,
        "last_prediction_time":          _LAST_PREDICTION_CALENDAR_ISO,
        # ── High-Congestion Predictive Positioning Telemetry ──────────────
        "predictive_positioning_status": predictive_positioner.status,
        "predictive_status_reason":      predictive_positioner.status_reason,
        "predicted_target_floor":        active_target_fl,
        "predicted_target_demand":       active_target_dm,
        "assigned_predictive_elevators": predictive_positioner.get_assigned_elevators(),
        "assigned_allocations":          predictive_positioner.get_status_dict()["assigned_allocations"],
        "eligible_elevators_count":      predictive_positioner.last_eligible_count,
        "floor_evaluations":             predictive_positioner.last_floor_evaluations,
        "elevator_operational_status":   elevator_operational_status,
        "high_congestion_threshold":     predictive_positioner.high_congestion_threshold,
        "release_threshold":             predictive_positioner.release_threshold,
        "confirmation_cycles_required":  predictive_positioner.confirmation_cycles_required,
        "target_reposition_count":       predictive_positioner.target_reposition_count,
        "last_decision_log":             predictive_positioner.last_decision_log,
        "stationed_elevators":          predictive_positioner.get_status_dict().get("stationed_elevators", []),
        "recent_diagnostics":           predictive_positioner.get_status_dict().get("recent_diagnostics", []),
        # ── dispatch decision source ──────────────────────────────────────
        "dispatch_source":      _LAST_DISPATCH_SOURCE,
        "q_learning_active":    stats["epsilon"] < 0.5,
        # ── energy & movement telemetry ───────────────────────────────────
        "empty_movement_floors":     state.get("empty_movement_floors", 0),
        "passenger_movement_floors": state.get("passenger_movement_floors", 0),
        "demand_threshold":          DEMAND_THRESHOLD,
        "demand_sustained_count":    _DEMAND_SUSTAINED_COUNT,
        "demand_sustained_required": DEMAND_SUSTAINED_REQUIRED,
    }


# ── Clock & Simulation Time Control Endpoints ─────────────────────────────────

@app.route("/api/clock/set", methods=["POST"])
def set_clock():
    """
    Sets the simulation calendar date and time.
    Body:
      {
        "date": "YYYY-MM-DD" (optional),
        "time": "HH:MM:SS" or "09:00:00 AM" (optional),
        "mode": "custom" | "real_time" (optional)
      }
    """
    data = request.get_json() or {}
    date_str = data.get("date")
    time_str = data.get("time")
    mode_str = data.get("mode")

    if mode_str:
        sim_clock.set_mode(mode_str)

    new_base = sim_clock.set_custom_time(date_str=date_str, time_str=time_str)

    # Re-seed historical buffer with 10 sequential chronological timesteps ending at new_base
    cur_state = env.get_state()
    seed_feature_buffer(new_base, cur_state)

    # Immediately refresh prediction for the new temporal context
    _run_lstm_inference(cur_state, force=True)
    reevaluate_predictive_positioning(trigger_event="clock_time_set")

    return jsonify({
        "status": "success",
        "message": f"Simulation clock set to {new_base.strftime('%Y-%m-%d %I:%M:%S %p')} ({sim_clock.timezone_name})",
        "clock": sim_clock.get_status_dict(),
        "state": get_system_state_dict(),
    })


@app.route("/api/clock/start", methods=["POST"])
def start_clock():
    """Starts the simulation clock and environment execution."""
    global IS_RUNNING
    sim_clock.start()
    IS_RUNNING = True
    return jsonify({
        "status": "success",
        "message": "Simulation clock started",
        "clock": sim_clock.get_status_dict(),
        "state": get_system_state_dict(),
    })


@app.route("/api/clock/pause", methods=["POST"])
def pause_clock():
    """Freezes clock advancement and simulation progression without clearing routes or state."""
    global IS_RUNNING
    sim_clock.pause()
    IS_RUNNING = False
    return jsonify({
        "status": "success",
        "message": "Simulation clock paused",
        "clock": sim_clock.get_status_dict(),
        "state": get_system_state_dict(),
    })


@app.route("/api/clock/resume", methods=["POST"])
def resume_clock():
    """Resumes clock advancement from the exact paused time."""
    global IS_RUNNING
    sim_clock.resume()
    IS_RUNNING = True
    return jsonify({
        "status": "success",
        "message": "Simulation clock resumed",
        "clock": sim_clock.get_status_dict(),
        "state": get_system_state_dict(),
    })


@app.route("/api/clock/speed", methods=["POST"])
def set_clock_speed():
    """
    Sets simulation speed scaling factor (1.0, 2.0, 5.0).
    Smoothly re-anchors monotonic time to prevent jumps.
    Body: { "speed": float }
    """
    data = request.get_json() or {}
    try:
        new_speed = float(data.get("speed", 1.0))
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid speed"}), 400

    sim_clock.set_speed(new_speed)
    return jsonify({
        "status": "success",
        "message": f"Simulation speed set to {sim_clock.speed}x",
        "clock": sim_clock.get_status_dict(),
        "state": get_system_state_dict(),
    })


@app.route("/api/clock/mode", methods=["POST"])
def set_clock_mode():
    """
    Switches clock mode between 'custom' and 'real_time'.
    Body: { "mode": "custom" | "real_time" }
    """
    data = request.get_json() or {}
    mode = data.get("mode", "custom")
    sim_clock.set_mode(mode)
    cur_state = env.get_state()
    seed_feature_buffer(sim_clock.get_calendar_datetime(), cur_state)
    _run_lstm_inference(cur_state, force=True)
    return jsonify({
        "status": "success",
        "message": f"Clock mode set to {sim_clock.mode}",
        "clock": sim_clock.get_status_dict(),
        "state": get_system_state_dict(),
    })


@app.route("/api/clock/status", methods=["GET"])
def get_clock_status():
    """Returns active simulation clock status and prediction metadata."""
    active_target_fl = predictive_positioner.active_target_floor
    active_target_dm = round(predictive_positioner.active_target_demand, 2)
    if active_target_fl is None and len(_LAST_FLOOR_DEMAND_T5) > 0:
        max_d = max(_LAST_FLOOR_DEMAND_T5)
        active_target_fl = _LAST_FLOOR_DEMAND_T5.index(max_d) + 1
        active_target_dm = round(float(max_d), 2)

    return jsonify({
        "clock": sim_clock.get_status_dict(),
        "last_prediction_time": _LAST_PREDICTION_CALENDAR_ISO,
        "prediction_horizon_min": 5,
        "predicted_congestion": _LAST_CONGESTION_T5,
        "predicted_floor_demand": list(_LAST_FLOOR_DEMAND_T5),
        "model_mode": _LSTM_MODE,
        "predictive_positioning_enabled": True,
        "predictive_positioning_status": predictive_positioner.status,
        "predictive_status_reason": predictive_positioner.status_reason,
        "predicted_target_floor": active_target_fl,
        "predicted_target_demand": active_target_dm,
        "assigned_predictive_elevators": predictive_positioner.get_assigned_elevators(),
        "assigned_allocations": predictive_positioner.get_status_dict()["assigned_allocations"],
        "eligible_elevators_count": predictive_positioner.last_eligible_count,
        "floor_evaluations": predictive_positioner.last_floor_evaluations,
        "high_congestion_threshold": predictive_positioner.high_congestion_threshold,
        "release_threshold": predictive_positioner.release_threshold,
        "last_decision_log": predictive_positioner.last_decision_log,
        "stationed_elevators": predictive_positioner.get_status_dict().get("stationed_elevators", []),
        "recent_diagnostics": predictive_positioner.get_status_dict().get("recent_diagnostics", []),
    })


@app.route("/api/batch_status", methods=["GET"])
def get_batch_status():
    """Returns the current batching window status for the frontend countdown."""
    return jsonify(batch_dispatcher.get_status())

@app.route("/api/emergency", methods=["POST"])
def trigger_emergency():
    """
    Submits an emergency-priority request for a specific floor.
    Bypasses the normal 2-second batching window and assigns the optimal elevator.
    Body: { "floor": int }
    """
    data = request.get_json() or {}
    raw_floor = data.get("floor", data.get("emergency_floor", data.get("src_floor", 1)))
    try:
        floor = int(raw_floor)
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid floor number"}), 400

    if floor < 1 or floor > env.num_floors:
        return jsonify({
            "status": "error",
            "message": f"Floor must be between 1 and {env.num_floors}"
        }), 400

    res = emergency_dispatcher.dispatch_emergency(floor)
    if res.get("status") == "success":
        global IS_RUNNING
        IS_RUNNING = True  # Ensure simulation is active to advance emergency elevator

    return jsonify({
        **res,
        "state": get_system_state_dict(),
    })

@app.route("/api/emergency/resolve", methods=["POST"])
def resolve_emergency_endpoint():
    """
    Resolves an active emergency and restores normal dispatch.
    Body: { "emergency_id": int (optional), "floor": int (optional) }
    """
    data = request.get_json() or {}
    emg_id = data.get("emergency_id", data.get("id"))
    floor = data.get("floor")
    res = emergency_dispatcher.resolve_emergency(emg_id, floor)
    reevaluate_predictive_positioning(trigger_event="emergency_resolved")
    return jsonify({
        **res,
        "state": get_system_state_dict(),
    })

@app.route("/api/emergency/status", methods=["GET"])
def get_emergency_status_endpoint():
    """Returns the current emergency status."""
    return jsonify(emergency_dispatcher.get_status_summary())

@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns full current system state."""
    return jsonify(get_system_state_dict())

@app.route("/api/predict", methods=["POST"])
def predict_congestion():
    """
    Runs LSTM prediction directly given an input sequence:
    body: { sequence: [[18/20 features]*10] }
    Returns single congestion float (backward-compatible).
    """
    data = request.get_json() or {}
    sequence = data.get("sequence")
    if not sequence or len(sequence) != 10:
        return jsonify({"error": "sequence must contain 10 timesteps"}), 400

    try:
        if data_processor.is_fitted:
            scaled_seq = data_processor.scale_single_sequence(sequence)
            pred = lstm_model.predict(scaled_seq)
        else:
            pred = lstm_model.predict(sequence)
        return jsonify({"congestion_level": float(pred), "model_mode": _LSTM_MODE})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_full", methods=["POST"])
def predict_full_endpoint():
    """
    Full multi-output LSTM prediction.
    body: { sequence: [[20 features]*10] }  (optional – uses live buffer if omitted)
    Returns congestion_t5 (float), floor_demand_t5 (list[10]) and model_mode.
    """
    data     = request.get_json() or {}
    sequence = data.get("sequence")

    if sequence and len(sequence) == 10:
        try:
            if data_processor.is_fitted:
                scaled_seq = data_processor.scale_single_sequence(sequence)
                result = lstm_model.predict_full(scaled_seq)
            else:
                result = lstm_model.predict_full(sequence)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Use current live feature buffer
    state = env.get_state()
    cong, demand = _run_lstm_inference(state)
    return jsonify({
        "congestion_t5":   cong,
        "floor_demand_t5": demand,
        "model_mode":      _LSTM_MODE,
    })

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Returns performance and diagnostic metrics."""
    stats = q_agent.get_stats()
    state = env.get_state()
    return jsonify({
        "q_table_size": stats["q_table_size"],
        "epsilon": stats["epsilon"],
        "pending_requests": state["pending_requests"],
        "avg_wait_time": state["avg_wait_time"],
        "total_dispatched": TOTAL_DISPATCHED,
        "current_congestion": state["congestion_prediction"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Elevator System Backend on http://127.0.0.1:{port}...")
    app.run(host="0.0.0.0", port=port, debug=True)

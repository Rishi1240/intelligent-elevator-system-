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
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from lstm_model import LSTMModel
from data_processor import DataProcessor
from q_learning_agent import QLearningAgent
from elevator_env import ElevatorEnvironment

app = Flask(__name__)
# Enable CORS for React frontend (Vite port 5173 and CRA port 3000)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global system state
env = ElevatorEnvironment()
q_agent = QLearningAgent(
    num_elevators=env.num_elevators,
    num_floors=env.num_floors
)
lstm_model = LSTMModel()
data_processor = DataProcessor()

# Buffer to maintain sliding window of feature records for LSTM inference (seq_len = 10)
FEATURE_BUFFER = deque(maxlen=10)
IS_RUNNING = False
TOTAL_DISPATCHED = 0

def build_current_feature_vector(state_dict):
    """
    Constructs real-time feature vector matching training dataset columns:
    [hour, minute, day_of_week, floor_req_f1..f10, total_requests,
     elev1_floor, elev2_floor, elev3_floor, elev1_occ, elev2_occ, elev3_occ]
    """
    now = datetime.datetime.now()
    hour = now.hour
    minute = now.minute
    day_of_week = now.weekday()

    floor_demands = state_dict.get("floor_demand", [0] * 10)
    total_req = state_dict.get("pending_requests", 0)

    elev_positions = state_dict.get("elevator_positions", [1, 1, 1])
    elev_occs = state_dict.get("elevator_occupancy", [0, 0, 0])

    vector = [
        hour, minute, day_of_week,
        *floor_demands,
        total_req,
        *elev_positions,
        *elev_occs
    ]
    return vector

def update_feature_buffer(state_dict):
    """Pushes latest feature vector into the sliding window buffer."""
    vec = build_current_feature_vector(state_dict)
    FEATURE_BUFFER.append(vec)

# Pre-populate feature buffer with initial baseline frames
init_state = env.get_state()
for _ in range(10):
    update_feature_buffer(init_state)

def initialize_models():
    """Attempts to load trained LSTM weights, scaler, and Q-table if available."""
    global lstm_model, q_agent, data_processor

    saved_dir = os.path.join(CURRENT_DIR, "saved_model")
    weights_path = os.path.join(saved_dir, "lstm_weights.h5")
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
            lstm_model.load(weights_path)
            print(f"[Init] Loaded LSTM model from {weights_path}")
        except Exception as e:
            print(f"[Init] Warning: Could not load LSTM weights: {e}")

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
      { "num_floors": int (2–20), "num_elevators": int (1–5), "capacity": int }
    Caps are enforced server-side. Resets the simulation.
    """
    global env, q_agent, IS_RUNNING
    data = request.get_json() or {}
    num_floors    = data.get("num_floors")
    num_elevators = data.get("num_elevators")
    capacity      = data.get("capacity")
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
    IS_RUNNING = False
    return jsonify({
        "status": "success",
        "message": f"Reconfigured: {env.num_floors} floors, {env.num_elevators} elevators, capacity {env.capacity}",
        "state": get_system_state_dict(),
    })

@app.route("/api/start", methods=["POST"])
def start_system():
    """Initializes environment and reloads models."""
    global env, q_agent, lstm_model, IS_RUNNING
    env.reset()
    IS_RUNNING = True
    initialize_models()
    return jsonify({
        "status": "success",
        "message": "Elevator system initialized and active",
        "state": get_system_state_dict()
    })

@app.route("/api/reset", methods=["POST"])
def reset_system():
    """Resets simulator to clean state."""
    global env, IS_RUNNING
    env.reset()
    IS_RUNNING = False
    return jsonify({
        "status": "success",
        "message": "Elevator system reset",
        "state": get_system_state_dict()
    })

def is_system_idle():
    """
    Returns True when there are no pending requests AND all elevators
    are empty and have no committed targets or assigned stops. In this state the RL agent must NOT move any car.
    """
    if env.pending_requests:
        return False
    if any(len(passengers) > 0 for passengers in env.elevator_passengers):
        return False
    if any(t is not None for t in getattr(env, "committed_targets", [])):
        return False
    if any(len(stops) > 0 for stops in getattr(env, "assigned_stops", [])):
        return False
    return True


def run_rl_dispatch_step():
    """
    Executes one end-to-end dispatch cycle using LSTM + Q-Learning.
    Returns (None, 0, {}) when the system is idle so callers can skip movement.
    """
    global TOTAL_DISPATCHED

    # ── IDLE GUARD: do not dispatch when building has no work ──
    if is_system_idle():
        return None, 0.0, {}

    current_state = env.get_state()
    update_feature_buffer(current_state)

    # 1. Predict congestion with LSTM using current feature sequence
    congestion_val = 0.20
    try:
        seq = list(FEATURE_BUFFER)
        if len(seq) == 10:
            if data_processor.is_fitted:
                scaled_seq = data_processor.scale_single_sequence(seq)
                congestion_val = lstm_model.predict(scaled_seq)
            else:
                congestion_val = lstm_model.predict(seq)
    except Exception as e:
        print(f"[Predict Error] {e}")

    env.congestion_prediction = congestion_val
    current_state["congestion_prediction"] = congestion_val

    # 2. Centralized Dispatch for any unassigned requests
    unassigned = env.get_unassigned_requests()
    dispatch_action = None
    for req in unassigned:
        valid_actions, heuristics = env.get_dispatch_candidates(req)
        if valid_actions:
            act = q_agent.get_dispatch_action(current_state, valid_actions, heuristics)
            if act:
                env.assign_request(req, act[0])
                if dispatch_action is None:
                    dispatch_action = act

    # 3. Environment Step: advance active elevators
    next_state, reward, info = env.step(dispatch_action)
    next_state["congestion_prediction"] = congestion_val

    # 4. Q-learning Bellman Update
    if dispatch_action:
        q_agent.update(current_state, dispatch_action, reward, next_state)
        TOTAL_DISPATCHED += 1

    return next_state, reward, info

@app.route("/api/request", methods=["POST"])
def add_passenger_request():
    """
    Receives passenger request: {src_floor: int, dst_floor: int}
    Queues request, performs one dispatch decision step, and returns state.
    """
    data = request.get_json() or {}
    src_floor = data.get("src_floor", 1)
    dst_floor = data.get("dst_floor", 2)

    success = env.add_request(src_floor, dst_floor)
    if not success:
        return jsonify({
            "status": "error",
            "message": "Invalid request: source and destination floors must be distinct (1-10)"
        }), 400

    next_state, reward, info = run_rl_dispatch_step()

    return jsonify({
        "status": "success",
        "message": f"Dispatched request from Floor {src_floor} to {dst_floor}",
        "reward": reward,
        "info": info,
        "state": get_system_state_dict()
    })

@app.route("/api/step", methods=["POST"])
def step_simulation():
    """Advances simulator by one step without adding a new request."""
    next_state, reward, info = run_rl_dispatch_step()
    if next_state is None:
        # System is idle – no elevators were moved
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
    state = env.get_state()
    stats = q_agent.get_stats()
    return {
        "elevator_positions": state["elevator_positions"],
        "elevator_occupancy": state["elevator_occupancy"],
        "elevator_targets":   state.get("elevator_targets", []),
        "assigned_stops":     state.get("assigned_stops", []),
        "congestion_level":   state["congestion_prediction"],
        "pending_requests":   state["pending_requests"],
        "floor_demand":       state["floor_demand"],
        "avg_wait_time":      state["avg_wait_time"],
        "last_action":        state["last_action"],
        "q_table_size":       stats["q_table_size"],
        "epsilon":            stats["epsilon"],
        "step_count":         state["step_count"],
        "is_running":         IS_RUNNING,
        # ── building config ────────────────────────────────────────────────
        "num_floors":    env.num_floors,
        "num_elevators": env.num_elevators,
        "capacity":      env.capacity,
    }

@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns full current system state."""
    return jsonify(get_system_state_dict())

@app.route("/api/predict", methods=["POST"])
def predict_congestion():
    """
    Runs LSTM prediction directly given an input sequence:
    body: { sequence: [[18/20 features]*10] }
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
        return jsonify({"congestion_level": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

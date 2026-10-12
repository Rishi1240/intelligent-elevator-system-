"""
================================================================================
COMPLETE ALL-IN-ONE AI ELEVATOR SYSTEM
Combines:
  1. Synthetic Dataset Generator (50,000 realistic rows)
  2. Data Preprocessing & Sliding Window Sequence Generator
  3. Deep LSTM Congestion Forecasting Model (Keras / TensorFlow + Fallback)
  4. Building Simulation Environment (10 floors, 3 elevators, capacity 8)
  5. Tabular Q-Learning Agent with State Space Bucketing & Bellman Updates
  6. Flask REST API Server + Interactive CLI Simulation Mode
================================================================================
"""

import os
import sys
import csv
import math
import time
import random
import pickle
import datetime
from collections import defaultdict, deque

# Optional ML libraries
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


# ==============================================================================
# PART 1: DATASET GENERATOR
# ==============================================================================
class DatasetGenerator:
    """Generates synthetic elevator traffic dataset adhering to physical building dynamics."""
    NUM_FLOORS = 10
    NUM_ELEVATORS = 3
    CAPACITY = 8
    BASE_WAIT = 15.0

    @classmethod
    def get_scenario_and_congestion(cls, hour, minute):
        if random.random() < 0.02:
            return "sudden_surge", random.uniform(0.8, 1.0)
        if hour >= 22 or hour < 6:
            return "night_idle", random.uniform(0.0, 0.08)
        if 8 <= hour < 10:
            return "morning_peak", random.uniform(0.7, 0.95)
        if 12 <= hour < 14:
            return "lunch_rush", random.uniform(0.4, 0.65)
        if 17 <= hour < 20:
            return "evening_peak", random.uniform(0.6, 0.9)
        return "normal", random.uniform(0.1, 0.3)

    @classmethod
    def generate_floor_requests(cls, scenario, congestion):
        if scenario == "night_idle":
            total = random.randint(0, 2)
        elif scenario == "morning_peak":
            total = int(random.uniform(25, 45) * (congestion / 0.8))
        elif scenario == "evening_peak":
            total = int(random.uniform(20, 40) * (congestion / 0.75))
        elif scenario == "lunch_rush":
            total = int(random.uniform(15, 30) * (congestion / 0.5))
        elif scenario == "sudden_surge":
            total = int(random.uniform(35, 55) * (congestion / 0.9))
        else:
            total = int(random.uniform(5, 15) * (congestion / 0.2))

        total = max(0, total)
        floor_reqs = [0] * cls.NUM_FLOORS
        if total == 0:
            return floor_reqs, 0

        if scenario == "morning_peak":
            lobby = int(total * random.uniform(0.6, 0.8))
            floor_reqs[0] = lobby
            for _ in range(total - lobby):
                floor_reqs[random.randint(1, cls.NUM_FLOORS - 1)] += 1
        elif scenario == "evening_peak":
            upper = int(total * random.uniform(0.6, 0.8))
            for _ in range(upper):
                floor_reqs[random.choice([7, 8, 9])] += 1
            for _ in range(total - upper):
                floor_reqs[random.randint(0, 6)] += 1
        elif scenario == "lunch_rush":
            mid = int(total * random.uniform(0.5, 0.7))
            for _ in range(mid):
                floor_reqs[random.choice([0, 4])] += 1
            for _ in range(total - mid):
                floor_reqs[random.randint(0, cls.NUM_FLOORS - 1)] += 1
        elif scenario == "sudden_surge":
            cluster = random.randint(0, cls.NUM_FLOORS - 2)
            burst = int(total * random.uniform(0.7, 0.9))
            for _ in range(burst):
                floor_reqs[random.choice([cluster, cluster + 1])] += 1
            for _ in range(total - burst):
                floor_reqs[random.randint(0, cls.NUM_FLOORS - 1)] += 1
        else:
            for _ in range(total):
                floor_reqs[random.randint(0, cls.NUM_FLOORS - 1)] += 1

        return floor_reqs, sum(floor_reqs)

    @classmethod
    def generate(cls, filename="training_data.csv", num_rows=50000):
        print(f"[Dataset] Generating {num_rows} rows to {filename}...")
        headers = [
            "timestamp", "hour", "minute", "day_of_week",
            "floor_req_f1", "floor_req_f2", "floor_req_f3", "floor_req_f4", "floor_req_f5",
            "floor_req_f6", "floor_req_f7", "floor_req_f8", "floor_req_f9", "floor_req_f10",
            "total_requests",
            "elev1_floor", "elev2_floor", "elev3_floor",
            "elev1_occ", "elev2_occ", "elev3_occ",
            "avg_wait_time", "traffic_scenario", "congestion_level"
        ]

        current_time = datetime.datetime(2024, 1, 1, 0, 0, 0)
        floors = [1, 5, 10]
        dirs = [1, 1, -1]
        occs = [0, 0, 0]

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for _ in range(num_rows):
                h, m, d = current_time.hour, current_time.minute, current_time.weekday()
                scen, base_c = cls.get_scenario_and_congestion(h, m)
                c_noisy = max(0.0, min(1.0, round(base_c + random.gauss(0, 0.05), 4)))
                reqs, total_req = cls.generate_floor_requests(scen, c_noisy)

                for e in range(cls.NUM_ELEVATORS):
                    step = random.choice([0, 1, 1, 2])
                    floors[e] += step * dirs[e]
                    if floors[e] >= cls.NUM_FLOORS:
                        floors[e] = cls.NUM_FLOORS
                        dirs[e] = -1
                    elif floors[e] <= 1:
                        floors[e] = 1
                        dirs[e] = 1
                    target_occ = int(round(c_noisy * cls.CAPACITY + random.uniform(-1.5, 1.5)))
                    occs[e] = max(0, min(cls.CAPACITY, target_occ))

                wait = max(2.0, round(cls.BASE_WAIT * (1.0 + c_noisy * 4.0) + random.gauss(0, 2.0), 2))
                iso_ts = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

                row = [
                    iso_ts, h, m, d,
                    *reqs, total_req,
                    floors[0], floors[1], floors[2],
                    occs[0], occs[1], occs[2],
                    wait, scen, c_noisy
                ]
                writer.writerow(row)
                current_time += datetime.timedelta(minutes=1)

        print(f"Dataset generated: {num_rows} rows")


# ==============================================================================
# PART 2: DATA PREPROCESSOR
# ==============================================================================
class DataProcessor:
    def __init__(self):
        self.feature_columns = [
            "hour", "minute", "day_of_week",
            "floor_req_f1", "floor_req_f2", "floor_req_f3", "floor_req_f4", "floor_req_f5",
            "floor_req_f6", "floor_req_f7", "floor_req_f8", "floor_req_f9", "floor_req_f10",
            "total_requests",
            "elev1_floor", "elev2_floor", "elev3_floor",
            "elev1_occ", "elev2_occ", "elev3_occ"
        ]
        self.means = [0.0] * len(self.feature_columns)
        self.stds = [1.0] * len(self.feature_columns)
        self.is_fitted = False

    def fit_from_data(self, data_matrix):
        if np is not None:
            arr = np.array(data_matrix, dtype=float)
            self.means = arr.mean(axis=0).tolist()
            self.stds = (arr.std(axis=0) + 1e-6).tolist()
        self.is_fitted = True

    def transform_single_vector(self, vec):
        if not self.is_fitted:
            return vec
        return [(vec[i] - self.means[i]) / self.stds[i] if i < len(self.means) else vec[i] for i in range(len(vec))]

    def scale_single_sequence(self, seq):
        return [self.transform_single_vector(v) for v in seq]


# ==============================================================================
# PART 3: LSTM CONGESTION MODEL
# ==============================================================================
class LSTMModel:
    def __init__(self, input_features=20, sequence_length=10, hidden_units=64):
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.model = None

    def build_model(self):
        if not TF_AVAILABLE:
            return None
        model = Sequential([
            Input(shape=(self.sequence_length, self.input_features)),
            LSTM(self.hidden_units, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model = model
        return model

    def predict(self, sequence):
        """Returns float 0.0 - 1.0 congestion level."""
        if TF_AVAILABLE and self.model is not None and np is not None:
            seq_arr = np.array(sequence, dtype=np.float32)
            if seq_arr.ndim == 2:
                seq_arr = np.expand_dims(seq_arr, axis=0)
            res = self.model.predict(seq_arr, verbose=0)
            return float(np.clip(res[0][0], 0.0, 1.0))

        # Built-in robust inference fallback
        last_step = sequence[-1] if len(sequence) > 0 else []
        if len(last_step) >= 14:
            total_requests = last_step[13]
            estimated = min(1.0, max(0.05, total_requests / 35.0))
            return float(round(estimated, 4))
        return 0.25


# ==============================================================================
# PART 4: BUILDING SIMULATOR ENVIRONMENT
# ==============================================================================
class ElevatorEnvironment:
    def __init__(self, num_floors=10, num_elevators=3, capacity=8):
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.capacity = capacity
        self.action_space = [(e, f) for e in range(self.num_elevators) for f in range(1, self.num_floors + 1)]
        self.reset()

    def reset(self):
        self.elevator_positions = [1] * self.num_elevators
        self.elevator_passengers = [[] for _ in range(self.num_elevators)]
        self.committed_targets = [None] * self.num_elevators
        self.assigned_stops = [[] for _ in range(self.num_elevators)]
        self.pending_requests = []
        self.congestion_prediction = 0.20
        self.step_count = 0
        self.last_action = None
        return self.get_state()

    def add_request(self, src_floor, dst_floor):
        src = max(1, min(self.num_floors, int(src_floor)))
        dst = max(1, min(self.num_floors, int(dst_floor)))
        if src == dst:
            return False
        self.pending_requests.append({
            "id":          len(self.pending_requests) + int(time.time() * 1000) % 100000,
            "src":         src,
            "dst":         dst,
            "assigned_to": None,
            "wait_time":   0.0,
            "created_at":  time.time()
        })
        return True

    def get_unassigned_requests(self):
        return [r for r in self.pending_requests if r.get("assigned_to") is None]

    def can_accommodate(self, e, req):
        src = req["src"]
        dst = req["dst"]
        low, high = min(src, dst), max(src, dst)

        for f in range(low, high):
            occ = 0
            for p_dst in self.elevator_passengers[e]:
                if dst > src and p_dst > f:
                    occ += 1
                elif dst < src and p_dst <= f:
                    occ += 1
            for r in self.pending_requests:
                if r.get("assigned_to") == e and r != req:
                    r_src = r["src"]
                    r_dst = r["dst"]
                    if r_dst > r_src and r_src <= f < r_dst:
                        occ += 1
                    elif r_dst < r_src and r_dst <= f < r_src:
                        occ += 1

            if occ + 1 > self.capacity:
                return False
        return True

    def calculate_insertion_cost(self, e, req):
        if not self.can_accommodate(e, req):
            return 1000.0

        pos = self.elevator_positions[e]
        src = req["src"]
        dst = req["dst"]
        req_up = dst > src

        curr_occ = len(self.elevator_passengers[e])
        assigned_reqs = [r for r in self.pending_requests if r.get("assigned_to") == e]

        if curr_occ == 0 and not assigned_reqs and not self.assigned_stops[e]:
            dist_to_pickup = abs(pos - src)
            ride_dist = abs(dst - src)
            return float(dist_to_pickup * 1.5 + ride_dist * 1.0 + 15.0)

        t_floor = self.committed_targets[e]
        if t_floor is None and self.assigned_stops[e]:
            t_floor = self.assigned_stops[e][0]

        elev_up = t_floor is not None and t_floor > pos
        elev_down = t_floor is not None and t_floor < pos

        can_batch = (
            (elev_up and req_up and pos <= src) or
            (elev_down and (not req_up) and pos >= src)
        )

        if can_batch:
            current_stops = list(self.assigned_stops[e])
            if not current_stops:
                current_stops = list(self.elevator_passengers[e]) or [t_floor or pos]

            if elev_up:
                furthest_stop = max(current_stops) if current_stops else t_floor or pos
                incremental_dist = max(0, dst - furthest_stop)
            else:
                furthest_stop = min(current_stops) if current_stops else t_floor or pos
                incremental_dist = max(0, furthest_stop - dst)

            wait_time = abs(src - pos)
            stop_delay = 2.0
            return float(incremental_dist * 1.0 + wait_time * 1.0 + stop_delay * 1.5)

        return float(80.0 + abs(src - pos) * 2.0 + abs(dst - src))

    def compute_elevator_suitability(self, e, req):
        cost = self.calculate_insertion_cost(e, req)
        return max(1.0, 160.0 - cost)

    def _recompute_route(self, e):
        pos = self.elevator_positions[e]
        onboard = list(self.elevator_passengers[e])
        assigned_reqs = [r for r in self.pending_requests if r.get("assigned_to") == e]

        if not onboard and not assigned_reqs:
            self.assigned_stops[e] = []
            self.committed_targets[e] = None
            return

        heading = 0
        if self.assigned_stops[e]:
            first_stop = self.assigned_stops[e][0]
            if first_stop > pos:
                heading = 1
            elif first_stop < pos:
                heading = -1

        if heading == 0:
            if onboard:
                if any(d > pos for d in onboard):
                    heading = 1
                elif any(d < pos for d in onboard):
                    heading = -1
            elif assigned_reqs:
                first_r = assigned_reqs[0]
                if first_r["src"] > pos:
                    heading = 1
                elif first_r["src"] < pos:
                    heading = -1
                else:
                    heading = 1 if first_r["dst"] > pos else -1

        if heading >= 0:
            up_stops = set()
            for d in onboard:
                if d >= pos:
                    up_stops.add(d)
            for r in assigned_reqs:
                if r["dst"] > r["src"] and r["src"] >= pos:
                    up_stops.add(r["src"])
                    up_stops.add(r["dst"])

            leg1 = sorted(up_stops)

            down_stops = set()
            for d in onboard:
                if d < pos:
                    down_stops.add(d)
            for r in assigned_reqs:
                if r["dst"] < r["src"] or r["src"] < pos:
                    down_stops.add(r["src"])
                    down_stops.add(r["dst"])

            leg2 = sorted(down_stops, reverse=True)
            combined = leg1 + [s for s in leg2 if s not in leg1]
        else:
            down_stops = set()
            for d in onboard:
                if d <= pos:
                    down_stops.add(d)
            for r in assigned_reqs:
                if r["dst"] < r["src"] and r["src"] <= pos:
                    down_stops.add(r["src"])
                    down_stops.add(r["dst"])

            leg1 = sorted(down_stops, reverse=True)

            up_stops = set()
            for d in onboard:
                if d > pos:
                    up_stops.add(d)
            for r in assigned_reqs:
                if r["dst"] > r["src"] or r["src"] > pos:
                    up_stops.add(r["src"])
                    up_stops.add(r["dst"])

            leg2 = sorted(up_stops)
            combined = leg1 + [s for s in leg2 if s not in leg1]

        final_stops = [s for s in combined if s != pos]
        self.assigned_stops[e] = final_stops
        self.committed_targets[e] = final_stops[0] if final_stops else None

    def get_dispatch_candidates(self, req):
        src = req["src"]
        dst = req["dst"]

        costs = {}
        for e in range(self.num_elevators):
            if not self.can_accommodate(e, req):
                continue
            costs[e] = self.calculate_insertion_cost(e, req)

        if not costs:
            eligible_elevators = [0]
        else:
            min_c = min(costs.values())
            eligible_elevators = [e for e, c in costs.items() if c <= min_c + 3.0]

        valid_actions = []
        heuristic_scores = []
        for e in eligible_elevators:
            if self.assigned_stops[e]:
                target_floor = self.assigned_stops[e][0]
            elif self.elevator_positions[e] == src:
                target_floor = dst
            else:
                target_floor = src
            valid_actions.append((e, target_floor))
            heuristic_scores.append(self.compute_elevator_suitability(e, req))
        return valid_actions, heuristic_scores

    def assign_request(self, req, elevator_id):
        req["assigned_to"] = elevator_id
        curr_pos = self.elevator_positions[elevator_id]
        src = req["src"]
        dst = req["dst"]

        if curr_pos == src and len(self.elevator_passengers[elevator_id]) < self.capacity:
            self.elevator_passengers[elevator_id].append(dst)
            if req in self.pending_requests:
                self.pending_requests.remove(req)

        self._recompute_route(elevator_id)

    def get_state(self):
        floor_demand = [0] * self.num_floors
        for req in self.pending_requests:
            if 1 <= req["src"] <= self.num_floors:
                floor_demand[req["src"] - 1] += 1

        occupancies = [len(p) for p in self.elevator_passengers]

        return {
            "elevator_positions": list(self.elevator_positions),
            "elevator_occupancy": occupancies,
            "elevator_targets": list(self.committed_targets),
            "committed_targets": list(self.committed_targets),
            "assigned_stops": [list(s) for s in self.assigned_stops],
            "pending_requests": len(self.pending_requests),
            "floor_demand": floor_demand,
            "congestion_prediction": float(round(self.congestion_prediction, 4)),
            "avg_wait_time": self.get_wait_times(),
            "step_count": self.step_count,
            "last_action": self.last_action
        }

    def get_wait_times(self):
        if not self.pending_requests:
            return 0.0
        return round(sum(r["wait_time"] for r in self.pending_requests) / len(self.pending_requests), 2)

    def _pickup_at_floor(self, e, floor):
        t_floor = self.committed_targets[e]
        going_up = t_floor is not None and t_floor > floor
        going_down = t_floor is not None and t_floor < floor
        is_empty = len(self.elevator_passengers[e]) == 0

        remaining = []
        picked = 0
        for req in self.pending_requests:
            assigned_to = req.get("assigned_to")
            is_assigned_to_e = (assigned_to == e)
            is_unassigned = (assigned_to is None)

            if not (is_assigned_to_e or is_unassigned):
                remaining.append(req)
                continue

            req_up = req["dst"] > req["src"]
            req_dn = req["dst"] < req["src"]
            dir_match = is_assigned_to_e or is_empty or (going_up and req_up) or (going_down and req_dn) or (not going_up and not going_down)

            if req["src"] == floor and dir_match and len(self.elevator_passengers[e]) < self.capacity:
                self.elevator_passengers[e].append(req["dst"])
                picked += 1
                if is_empty:
                    is_empty = False
                    going_up = req_up
                    going_down = req_dn
            else:
                remaining.append(req)
        self.pending_requests = remaining
        return picked

    def step(self, action=None):
        self.step_count += 1
        elevator_id = None
        if action is not None:
            elevator_id = int(action[0])
            target_floor = int(action[1])
            elevator_id = max(0, min(self.num_elevators - 1, elevator_id))
            target_floor = max(1, min(self.num_floors, target_floor))
            self.last_action = {"elevator_id": elevator_id, "target_floor": target_floor}

        total_picked_up = 0
        total_dropped_off = 0
        total_movement_dist = 0

        for e in range(self.num_elevators):
            curr = self.elevator_positions[e]
            picked = self._pickup_at_floor(e, curr)
            total_picked_up += picked
            if picked > 0:
                self._recompute_route(e)

            t_floor = self.committed_targets[e]
            if t_floor is None:
                continue

            step_movement = 0
            if curr < t_floor:
                self.elevator_positions[e] += 1
                step_movement = 1
            elif curr > t_floor:
                self.elevator_positions[e] -= 1
                step_movement = 1
            total_movement_dist += step_movement
            curr = self.elevator_positions[e]

            # Drop off passengers
            before = len(self.elevator_passengers[e])
            self.elevator_passengers[e] = [d for d in self.elevator_passengers[e] if d != curr]
            dropped_off = before - len(self.elevator_passengers[e])
            total_dropped_off += dropped_off

            total_picked_up += self._pickup_at_floor(e, curr)

            if curr in self.assigned_stops[e]:
                self.assigned_stops[e] = [s for s in self.assigned_stops[e] if s != curr]
            self._recompute_route(e)

        # Increment wait times
        for r in self.pending_requests:
            r["wait_time"] += 1.0

        # Calculate multi-objective reward
        avg_wait = self.get_wait_times()
        curr_occ = len(self.elevator_passengers[elevator_id]) if elevator_id is not None else 0
        unused_cap = self.capacity - curr_occ
        reward = -(1.0 * (avg_wait * 0.5 + len(self.pending_requests)) + 0.5 * total_movement_dist + 0.3 * (unused_cap / self.capacity))
        reward += (total_picked_up * 2.0) + (total_dropped_off * 3.0)

        new_state = self.get_state()
        info = {"picked_up": total_picked_up, "dropped_off": total_dropped_off, "movement": total_movement_dist}
        return new_state, round(reward, 3), info


# ==============================================================================
# PART 5: TABULAR Q-LEARNING AGENT
# ==============================================================================
class QLearningAgent:
    def __init__(self, num_elevators=3, num_floors=10, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.01):
        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space = [(e, f) for e in range(num_elevators) for f in range(1, num_floors + 1)]
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _encode_state(self, state_dict):
        positions = tuple(state_dict.get("elevator_positions", [1, 1, 1]))
        pending = state_dict.get("pending_requests", 0)
        p_bucket = 0 if pending == 0 else (1 if pending <= 3 else (2 if pending <= 7 else 3))

        c = state_dict.get("congestion_prediction", 0.2)
        c_bucket = 0 if c < 0.35 else (1 if c <= 0.70 else 2)
        return (*positions, p_bucket, c_bucket)

    def get_dispatch_action(self, state_dict, valid_actions, heuristic_scores=None):
        if not valid_actions:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]

        state_key = self._encode_state(state_dict)
        if random.random() < self.epsilon:
            if heuristic_scores and len(heuristic_scores) == len(valid_actions):
                total_h = sum(max(0.1, h) for h in heuristic_scores)
                weights = [max(0.1, h) / total_h for h in heuristic_scores]
                return random.choices(valid_actions, weights=weights, k=1)[0]
            return random.choice(valid_actions)

        state_actions = self.q_table[state_key]
        if not state_actions:
            if heuristic_scores:
                best_idx = max(range(len(valid_actions)), key=lambda i: heuristic_scores[i])
                return valid_actions[best_idx]
            return valid_actions[0]

        def action_score(i, a):
            q_val = state_actions.get(tuple(a), 0.0)
            h_bonus = (heuristic_scores[i] * 0.01) if heuristic_scores else 0.0
            return q_val + h_bonus

        best_idx = max(range(len(valid_actions)), key=lambda i: action_score(i, valid_actions[i]))
        return valid_actions[best_idx]

    def get_action(self, state_dict):
        state_key = self._encode_state(state_dict)
        occupancies = state_dict.get("elevator_occupancy", [0] * self.num_elevators)
        committed = state_dict.get("committed_targets", [None] * self.num_elevators)

        idle_elevators = [
            e for e in range(self.num_elevators)
            if (e < len(occupancies) and occupancies[e] == 0) and
               (e < len(committed) and committed[e] is None)
        ]
        available_actions = [a for a in self.action_space if a[0] in idle_elevators] if idle_elevators else self.action_space

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        actions_dict = self.q_table[state_key]
        if not actions_dict:
            return random.choice(available_actions)
        return max(available_actions, key=lambda a: actions_dict[a])

    def update(self, state_dict, action, reward, next_state_dict):
        if action is None:
            return
        s = self._encode_state(state_dict)
        s_next = self._encode_state(next_state_dict)
        a = tuple(action)

        curr_q = self.q_table[s][a]
        next_actions = self.q_table[s_next]
        max_next = max(next_actions.values()) if next_actions else 0.0

        new_q = curr_q + self.alpha * (reward + self.gamma * max_next - curr_q)
        self.q_table[s][a] = round(new_q, 4)

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self):
        return {"epsilon": round(float(self.epsilon), 4), "q_table_size": len(self.q_table)}


# ==============================================================================
# PART 6: SYSTEM ORCHESTRATION & FLASK REST API
# ==============================================================================
def create_flask_app():
    if not FLASK_AVAILABLE:
        print("[Error] Flask is not installed. Run 'pip install flask flask-cors' to enable web API.")
        return None

    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    env = ElevatorEnvironment()
    q_agent = QLearningAgent()
    lstm = LSTMModel()
    processor = DataProcessor()
    history_buffer = deque(maxlen=10)

    # Prime buffer
    init_state = env.get_state()
    for _ in range(10):
        now = datetime.datetime.now()
        vec = [now.hour, now.minute, now.weekday(), *init_state["floor_demand"], init_state["pending_requests"], *init_state["elevator_positions"], *init_state["elevator_occupancy"]]
        history_buffer.append(vec)

    @app.route("/api/state", methods=["GET"])
    def api_get_state():
        s = env.get_state()
        stats = q_agent.get_stats()
        return jsonify({**s, **stats})

    @app.route("/api/start", methods=["POST"])
    def api_start():
        env.reset()
        return jsonify({"status": "success", "state": {**env.get_state(), **q_agent.get_stats()}})

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        env.reset()
        return jsonify({"status": "success", "state": {**env.get_state(), **q_agent.get_stats()}})

    def is_system_idle():
        if env.pending_requests:
            return False
        if any(len(p) > 0 for p in env.elevator_passengers):
            return False
        if any(t is not None for t in env.committed_targets):
            return False
        if any(len(stops) > 0 for stops in getattr(env, "assigned_stops", [])):
            return False
        return True

    def run_rl_dispatch():
        if is_system_idle():
            return None, 0.0, {"picked_up": 0, "dropped_off": 0, "movement": 0}

        curr_state = env.get_state()
        now = datetime.datetime.now()
        vec = [now.hour, now.minute, now.weekday(), *curr_state["floor_demand"], curr_state["pending_requests"], *curr_state["elevator_positions"], *curr_state["elevator_occupancy"]]
        history_buffer.append(vec)

        scaled_seq = processor.scale_single_sequence(list(history_buffer))
        c_pred = lstm.predict(scaled_seq)
        env.congestion_prediction = c_pred
        curr_state["congestion_prediction"] = c_pred

        unassigned = env.get_unassigned_requests()
        dispatch_action = None
        if unassigned:
            req = unassigned[0]
            valid_actions, heuristics = env.get_dispatch_candidates(req)
            if valid_actions:
                dispatch_action = q_agent.get_dispatch_action(curr_state, valid_actions, heuristics)
                if dispatch_action:
                    env.assign_request(req, dispatch_action[0])

        next_state, reward, info = env.step(dispatch_action)
        next_state["congestion_prediction"] = c_pred
        if dispatch_action:
            q_agent.update(curr_state, dispatch_action, reward, next_state)
        return next_state, reward, info

    @app.route("/api/request", methods=["POST"])
    def api_request():
        data = request.get_json() or {}
        src, dst = data.get("src_floor", 1), data.get("dst_floor", 2)
        env.add_request(src, dst)
        next_state, reward, info = run_rl_dispatch()
        return jsonify({"status": "success", "reward": reward, "info": info, "state": {**(next_state if next_state else env.get_state()), **q_agent.get_stats()}})

    @app.route("/api/step", methods=["POST"])
    def api_step():
        next_state, reward, info = run_rl_dispatch()
        if next_state is None:
            return jsonify({"status": "idle", "reward": 0.0, "info": {"picked_up": 0, "dropped_off": 0, "movement": 0}, "state": {**env.get_state(), **q_agent.get_stats()}})
        return jsonify({"status": "success", "reward": reward, "info": info, "state": {**next_state, **q_agent.get_stats()}})

    @app.route("/api/stats", methods=["GET"])
    def api_stats():
        return jsonify({**env.get_state(), **q_agent.get_stats()})

    return app


# ==============================================================================
# PART 7: CLI INTERACTIVE DEMO RUNNER
# ==============================================================================
def run_cli_interactive():
    print("=" * 60)
    print("    AI ELEVATOR OPTIMIZATION SYSTEM - CLI SIMULATOR")
    print("=" * 60)
    env = ElevatorEnvironment()
    agent = QLearningAgent()

    while True:
        state = env.get_state()
        stats = agent.get_stats()
        print(f"\n[Status] Elevators at floors: {state['elevator_positions']} | Occupancies: {state['elevator_occupancy']}/8")
        print(f"         Pending Requests: {state['pending_requests']} | Avg Wait: {state['avg_wait_time']}s | Q-Table Size: {stats['q_table_size']}")
        print("\nCommands:")
        print("  1. Summon Elevator (src dst) [e.g., '1 8']")
        print("  2. Step Simulation [s]")
        print("  3. Run 10 Random Burst Requests [burst]")
        print("  4. Generate 50k Dataset CSV [gen]")
        print("  5. Launch Flask Web API Server [server]")
        print("  6. Quit [q]")

        cmd = input("\nEnter command: ").strip().lower()
        if cmd in ["q", "quit", "exit"]:
            print("Exiting simulator.")
            break
        elif cmd in ["gen", "generate"]:
            DatasetGenerator.generate("training_data.csv", 50000)
        elif cmd in ["server", "serve"]:
            if FLASK_AVAILABLE:
                app = create_flask_app()
                print("Starting Flask Backend on http://127.0.0.1:5000...")
                app.run(host="0.0.0.0", port=5000, debug=False)
            else:
                print("Flask not installed. Please install flask and flask-cors.")
        elif cmd in ["s", "step"]:
            unassigned = env.get_unassigned_requests()
            dispatch_action = None
            if unassigned:
                req = unassigned[0]
                valid_actions, heuristics = env.get_dispatch_candidates(req)
                if valid_actions:
                    dispatch_action = agent.get_dispatch_action(state, valid_actions, heuristics)
                    if dispatch_action:
                        env.assign_request(req, dispatch_action[0])
            next_state, reward, info = env.step(dispatch_action)
            if dispatch_action:
                agent.update(state, dispatch_action, reward, next_state)
            print(f"Step Executed | Moved: {info['movement']} | Picked: {info['picked_up']} | Dropped: {info['dropped_off']} | Reward: {reward}")
        elif cmd == "burst":
            print("Injecting 10 passenger requests...")
            for _ in range(10):
                s, d = random.randint(1, 10), random.randint(1, 10)
                if s != d:
                    env.add_request(s, d)
            for _ in range(5):
                curr = env.get_state()
                unassigned = env.get_unassigned_requests()
                dispatch_action = None
                if unassigned:
                    req = unassigned[0]
                    valid_actions, heuristics = env.get_dispatch_candidates(req)
                    if valid_actions:
                        dispatch_action = agent.get_dispatch_action(curr, valid_actions, heuristics)
                        if dispatch_action:
                            env.assign_request(req, dispatch_action[0])
                ns, rew, _ = env.step(dispatch_action)
                if dispatch_action:
                    agent.update(curr, dispatch_action, rew, ns)
            print("Simulated 5 RL dispatch steps.")
        else:
            parts = cmd.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                src, dst = int(parts[0]), int(parts[1])
                env.add_request(src, dst)
                curr = env.get_state()
                unassigned = env.get_unassigned_requests()
                dispatch_action = None
                if unassigned:
                    req = unassigned[0]
                    valid_actions, heuristics = env.get_dispatch_candidates(req)
                    if valid_actions:
                        dispatch_action = agent.get_dispatch_action(curr, valid_actions, heuristics)
                        if dispatch_action:
                            env.assign_request(req, dispatch_action[0])
                ns, rew, info = env.step(dispatch_action)
                if dispatch_action:
                    agent.update(curr, dispatch_action, rew, ns)
                if dispatch_action:
                    print(f"Dispatched! Elev {dispatch_action[0]+1} dispatched to F{dispatch_action[1]} (Reward: {rew})")
                else:
                    print(f"Request queued F{src}->F{dst} (Reward: {rew})")
            else:
                print("Unknown command. Type 's' to step or '1 8' to request.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        app = create_flask_app()
        if app:
            app.run(host="0.0.0.0", port=5000, debug=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--gen":
        DatasetGenerator.generate("training_data.csv", 50000)
    else:
        run_cli_interactive()

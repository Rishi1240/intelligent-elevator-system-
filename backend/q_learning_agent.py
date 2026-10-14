"""
Tabular Q-Learning Agent for Elevator Dispatch Optimization.
Implements Bellman equation state-action value updates, state space bucketing,
and epsilon-greedy exploration/exploitation.
"""

import os
import pickle
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(
        self,
        num_elevators=3,
        num_floors=10,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.9,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ):
        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 30 possible discrete actions: (elevator_id: 0..2, target_floor: 1..10)
        self.action_space = [
            (e, f) for e in range(self.num_elevators) for f in range(1, self.num_floors + 1)
        ]

        # Q-table: mapping state_tuple -> dict(action_tuple -> Q_value)
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.last_decision_source = "none"

    def _encode_state(self, state_dict):
        """
        Converts building state dict to a discrete, hashable state tuple key:
        (elev1_pos, elev2_pos, elev3_pos, pending_requests_bucket, congestion_bucket)
        """
        elev_positions = tuple(state_dict.get("elevator_positions", [1, 1, 1]))

        # Bucket pending requests: 0 -> 0, 1-3 -> 1, 4-7 -> 2, 8+ -> 3
        pending = state_dict.get("pending_requests", 0)
        if pending == 0:
            pending_bucket = 0
        elif 1 <= pending <= 3:
            pending_bucket = 1
        elif 4 <= pending <= 7:
            pending_bucket = 2
        else:
            pending_bucket = 3

        # Bucket congestion: low (<0.35) -> 0, medium (0.35-0.70) -> 1, high (>0.70) -> 2
        congestion = state_dict.get("congestion_prediction", 0.2)
        if congestion < 0.35:
            congestion_bucket = 0
        elif congestion <= 0.70:
            congestion_bucket = 1
        else:
            congestion_bucket = 2

        return (*elev_positions, pending_bucket, congestion_bucket)

    def get_dispatch_action(self, state_dict, valid_actions, heuristic_scores=None):
        """
        Ranks and selects the best dispatch action among valid_actions using Q-learning.
        Invalid actions are completely masked out.
        If heuristic_scores are provided, they guide exploration/tie-breaking.
        """
        if not valid_actions:
            return None

        if len(valid_actions) == 1:
            return valid_actions[0]

        state_key = self._encode_state(state_dict)

        if random.random() < self.epsilon:
            # Exploration strictly among valid actions, biased by heuristic suitability if provided
            self.last_decision_source = "heuristic"
            if heuristic_scores and len(heuristic_scores) == len(valid_actions):
                total_h = sum(max(0.1, h) for h in heuristic_scores)
                weights = [max(0.1, h) / total_h for h in heuristic_scores]
                return random.choices(valid_actions, weights=weights, k=1)[0]
            return random.choice(valid_actions)

        # Exploitation among valid actions
        state_actions = self.q_table[state_key]

        if not state_actions:
            self.last_decision_source = "heuristic"
            if heuristic_scores:
                best_idx = max(range(len(valid_actions)), key=lambda i: heuristic_scores[i])
                return valid_actions[best_idx]
            return valid_actions[0]

        def action_score(i, a):
            q_val = state_actions.get(tuple(a), 0.0)
            h_bonus = (heuristic_scores[i] * 0.01) if heuristic_scores else 0.0
            return q_val + h_bonus

        best_idx = max(range(len(valid_actions)), key=lambda i: action_score(i, valid_actions[i]))
        self.last_decision_source = "q_learning"
        return valid_actions[best_idx]

    def get_action(self, state_dict):
        """
        Selects action using epsilon-greedy strategy.
        Prioritizes available (idle / non-occupied) elevators for dispatching pending requests.
        """
        state_key = self._encode_state(state_dict)

        # Identify elevators that are free (occupancy == 0) and not committed
        occupancies = state_dict.get("elevator_occupancy", [0] * self.num_elevators)
        committed = state_dict.get("committed_targets", [None] * self.num_elevators)

        idle_elevators = [
            e for e in range(self.num_elevators)
            if (e < len(occupancies) and occupancies[e] == 0) and
               (e < len(committed) and committed[e] is None)
        ]

        # Prioritize actions on idle elevators if any are available
        if idle_elevators:
            available_actions = [a for a in self.action_space if a[0] in idle_elevators]
        else:
            available_actions = self.action_space

        if random.random() < self.epsilon:
            # Exploration
            return random.choice(available_actions)

        # Exploitation: argmax over Q-values for current state among available actions
        state_actions = self.q_table[state_key]
        if not state_actions:
            return random.choice(available_actions)

        best_action = max(available_actions, key=lambda a: state_actions[a])
        return best_action

    def update(self, state_dict, action, reward, next_state_dict):
        """
        Updates Q-value via Bellman equation:
        Q[s, a] = Q[s, a] + alpha * (reward + gamma * max(Q[s']) - Q[s, a])
        Also decays epsilon.
        """
        if action is None:
            return
        state_key = self._encode_state(state_dict)
        next_state_key = self._encode_state(next_state_dict)
        action_tuple = tuple(action)

        current_q = self.q_table[state_key][action_tuple]

        # max_a' Q(s', a')
        next_actions = self.q_table[next_state_key]
        max_next_q = max(next_actions.values()) if next_actions else 0.0

        # Bellman update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_tuple] = round(new_q, 4)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """Serializes Q-table to disk using pickle."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Convert defaultdict to regular dict for clean pickling
        serializable_dict = {s: dict(actions) for s, actions in self.q_table.items()}
        with open(path, "wb") as f:
            pickle.dump(serializable_dict, f)

    def load_q_table(self, path):
        """Loads serialized Q-table from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Q-table file not found at {path}")
        with open(path, "rb") as f:
            loaded_dict = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float))
        for s, actions in loaded_dict.items():
            for a, val in actions.items():
                self.q_table[s][a] = val

    def get_stats(self):
        """Returns training diagnostic metrics."""
        return {
            "epsilon": round(float(self.epsilon), 4),
            "q_table_size": len(self.q_table)
        }

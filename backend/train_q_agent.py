"""
Q-Table Training Script for Elevator Dispatch Agent.

Runs NUM_EPISODES simulated episodes, each with random passenger request
sequences, and trains the tabular Q-agent via Bellman updates.

Saves:
  saved_model/q_table.pkl  – serialised Q-table dictionary

Usage:
  python backend/train_q_agent.py [--episodes N]

IMPORTANT: This trains on synthetic simulations.  Reported rewards reflect
simulated, not real-building, performance.
"""

import os
import sys
import random
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from elevator_env    import ElevatorEnvironment
from q_learning_agent import QLearningAgent

# ── hyper-parameters ──────────────────────────────────────────────────────────
NUM_FLOORS    = 10
NUM_ELEVATORS = 3
CAPACITY      = 8
NUM_EPISODES  = 200      # short enough to complete in < 60 s
STEPS_PER_EP  = 60       # simulation steps per episode
MAX_REQS_PER_EP = 20     # max passenger requests generated per episode
SAVE_PATH     = os.path.join(CURRENT_DIR, "saved_model", "q_table.pkl")


def generate_episode_requests(env: ElevatorEnvironment, n: int) -> list:
    """Creates n distinct random passenger requests."""
    reqs = []
    for _ in range(n):
        src = random.randint(1, env.num_floors)
        dst = random.randint(1, env.num_floors)
        while dst == src:
            dst = random.randint(1, env.num_floors)
        reqs.append((src, dst))
    return reqs


def run_training(num_episodes: int = NUM_EPISODES) -> QLearningAgent:
    env   = ElevatorEnvironment(
        num_floors=NUM_FLOORS,
        num_elevators=NUM_ELEVATORS,
        capacity=CAPACITY,
    )
    agent = QLearningAgent(
        num_elevators=NUM_ELEVATORS,
        num_floors=NUM_FLOORS,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.9,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    )

    total_reward = 0.0

    for ep in range(1, num_episodes + 1):
        env.reset()

        # Scatter a random number of requests throughout the episode
        n_reqs = random.randint(1, MAX_REQS_PER_EP)
        req_list = generate_episode_requests(env, n_reqs)
        # Add initial wave (up to 5 at episode start)
        initial_wave = req_list[:5]
        remaining    = req_list[5:]
        release_at   = {}
        for i, (src, dst) in enumerate(remaining):
            step_n = random.randint(2, STEPS_PER_EP - 2)
            release_at.setdefault(step_n, []).append((src, dst))

        for (src, dst) in initial_wave:
            env.add_request(src, dst)

        ep_reward = 0.0

        for step in range(1, STEPS_PER_EP + 1):
            # Release scheduled requests
            for (src, dst) in release_at.get(step, []):
                env.add_request(src, dst)

            state = env.get_state()

            # Identify the single best unassigned request to dispatch
            unassigned = env.get_unassigned_requests()
            dispatch_action = None

            if unassigned:
                req = unassigned[0]
                raw_actions, raw_heuristics = env.get_dispatch_candidates(req)
                if raw_actions:
                    act = agent.get_dispatch_action(state, raw_actions, raw_heuristics)
                    if act is not None:
                        env.assign_request(req, act[0])
                        dispatch_action = act

            next_state, reward, _info = env.step(dispatch_action)
            ep_reward += reward

            if dispatch_action:
                agent.update(state, dispatch_action, reward, next_state)

        total_reward += ep_reward
        if ep % 50 == 0 or ep == 1:
            print(
                f"[Train] Episode {ep:3d}/{num_episodes}  "
                f"ep_reward={ep_reward:+.1f}  "
                f"epsilon={agent.epsilon:.3f}  "
                f"q_entries={len(agent.q_table)}"
            )

    avg_reward = total_reward / num_episodes
    print(f"\n[Train] Completed {num_episodes} episodes.  Avg reward={avg_reward:.2f}")
    print(f"[Train] Q-table entries: {len(agent.q_table)}")

    # Persist Q-table
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    agent.save_q_table(SAVE_PATH)
    print(f"[Train] Q-table saved to {SAVE_PATH}")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    args = parser.parse_args()
    run_training(num_episodes=args.episodes)

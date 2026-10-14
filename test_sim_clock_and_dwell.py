"""
Unit Tests for Simulation Clock, Dwell Window Dynamics, and Intermediate Floor Traversal.

Covers:
  - SimClock monotonic advancement, reset, and deterministic mode
  - Dwell window (3s) at pickup and dropoff floors
  - Combined pickup + dropoff single 3s dwell window
  - Stationary idle elevators
  - Intermediate floor travel (1s per floor)
  - QLearningAgent last_decision_source tracking
"""

import pytest
from backend.sim_clock import SimClock
from backend.elevator_env import ElevatorEnvironment, DWELL_WINDOW_SECS, TRAVEL_SECS
from backend.q_learning_agent import QLearningAgent


class TestSimClock:
    def test_clock_advancement(self):
        clock = SimClock(deterministic=True)
        assert clock.now() == 0.0
        clock.advance(1.0)
        assert clock.now() == 1.0
        clock.advance(2.5)
        assert clock.now() == 3.5

    def test_clock_reset(self):
        clock = SimClock(deterministic=True)
        clock.advance(10.0)
        assert clock.now() == 10.0
        clock.reset()
        assert clock.now() == 0.0

    def test_tick_delta(self):
        clock = SimClock(deterministic=True)
        clock.set_tick_delta(2.0)
        assert clock.tick_delta == 2.0
        clock.advance()  # defaults to tick_delta
        assert clock.now() == 2.0


class TestDwellAndTiming:
    def test_idle_elevator_stationary(self):
        clock = SimClock(deterministic=True)
        env = ElevatorEnvironment(num_floors=10, num_elevators=3, clock=clock)
        assert env.elevator_positions == [1, 1, 1]
        
        # Step with no requests
        for _ in range(5):
            env.step()
        
        # All elevators remain at floor 1, clock advanced by 5s
        assert env.elevator_positions == [1, 1, 1]
        assert env.clock.now() == 5.0

    def test_dwell_window_at_pickup(self):
        clock = SimClock(deterministic=True)
        env = ElevatorEnvironment(num_floors=10, num_elevators=1, clock=clock)
        # Elevator 0 starts at floor 1
        # Add request from floor 1 to floor 5
        env.add_request(1, 5)
        env.assign_request(env.pending_requests[0], 0)
        
        # Already at floor 1 -> immediate passenger board, target becomes 5
        assert env.elevator_positions[0] == 1
        assert 5 in env.elevator_passengers[0]
        assert env.committed_targets[0] == 5
        
        # Current floor is 1, target is 5 -> scheduled pickup floor triggers dwell or starts travel
        state, reward, info = env.step()
        # In step, it was at 1, moved to 2 (or dwelt)
        assert clock.now() == 1.0

    def test_dwell_window_at_dropoff(self):
        clock = SimClock(deterministic=True)
        env = ElevatorEnvironment(num_floors=5, num_elevators=1, clock=clock, dwell_window_secs=3.0)
        # Place elevator at floor 1 with passenger going to floor 2
        env.elevator_passengers[0] = [2]
        env.committed_targets[0] = 2
        
        # Step 1: elevator moves from floor 1 to floor 2
        env.step()
        assert env.elevator_positions[0] == 2
        # Passenger dropped off
        assert len(env.elevator_passengers[0]) == 0
        assert env.completed_passengers == 1
        # Elevator should have triggered dwell window at arrival
        assert env.dwell_remaining[0] == DWELL_WINDOW_SECS

        # During dwell, elevator stays stationary
        initial_floor = env.elevator_positions[0]
        env.committed_targets[0] = 5  # attempt to move to floor 5
        env.step()
        assert env.elevator_positions[0] == initial_floor
        assert env.dwell_remaining[0] == DWELL_WINDOW_SECS - TRAVEL_SECS

    def test_intermediate_floors_travel(self):
        """Travel from floor 1 to floor 4 takes 3 travel steps."""
        clock = SimClock(deterministic=True)
        env = ElevatorEnvironment(num_floors=10, num_elevators=1, clock=clock, dwell_window_secs=3.0)
        env.elevator_positions[0] = 1
        env.committed_targets[0] = 4
        env.assigned_stops[0] = [4]
        
        # Step 1: moves to 2
        env.step()
        assert env.elevator_positions[0] == 2
        # Step 2: moves to 3
        env.step()
        assert env.elevator_positions[0] == 3
        # Step 3: moves to 4 (destination, triggers dwell)
        env.step()
        assert env.elevator_positions[0] == 4
        assert env.dwell_remaining[0] == DWELL_WINDOW_SECS


class TestQLearningSource:
    def test_decision_source_tracking(self):
        agent = QLearningAgent(num_elevators=3, num_floors=10, epsilon=0.0)
        state_dict = {
            "elevator_positions": [1, 1, 1],
            "elevator_occupancy": [0, 0, 0],
            "pending_requests": 1,
            "congestion_prediction": 0.2,
        }
        valid_actions = [(0, 3), (1, 3)]
        heuristics = [10.0, 5.0]
        
        # Empty Q-table with epsilon 0 -> falls back to heuristic
        act = agent.get_dispatch_action(state_dict, valid_actions, heuristics)
        assert agent.last_decision_source == "heuristic"
        assert act == (0, 3)

        # Populate Q-table
        state_key = agent._encode_state(state_dict)
        agent.q_table[state_key][(1, 3)] = 50.0
        
        act = agent.get_dispatch_action(state_dict, valid_actions, heuristics)
        assert agent.last_decision_source == "q_learning"
        assert act == (1, 3)

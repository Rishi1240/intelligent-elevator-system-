"""
Elevator Simulation Environment.
Simulates a configurable building (2–20 floors, 1–5 elevators) with passenger
boarding/alighting, queue dynamics, wait time calculations, and multi-objective
RL reward formulation.

Simulation Timing (see sim_clock.py):
  - 1 simulated second per floor-to-floor movement.
  - No pause at intermediate floors (1s travel only).
  - 3 simulated seconds request-collection window at pickup floors only.
  - Drop-off floors: passengers exit immediately, no collection window.
  - Combined pickup+drop-off floors: single 3-second window.

Caps:
  - num_floors  : 2 ≤ n ≤ 20  (hard cap: 20)
  - num_elevators: 1 ≤ n ≤ 5   (hard cap:  5)
"""

import time
from typing import Any

from backend.sim_clock import SimClock


# ── absolute limits ──────────────────────────────────────────────────────────
MAX_FLOORS     = 20
MAX_ELEVATORS  = 5
MIN_FLOORS     = 2
MIN_ELEVATORS  = 1

# ── default values ───────────────────────────────────────────────────────────
DEFAULT_FLOORS     = 10
DEFAULT_ELEVATORS  = 3
DEFAULT_CAPACITY   = 8

# ── dwell / collection window durations (simulated seconds) ──────────────────
DWELL_WINDOW_SECS = 3.0     # collection window at pickup/drop-off floor
TRAVEL_SECS       = 1.0     # travel time between adjacent floors


class ElevatorEnvironment:
    def __init__(
        self,
        num_floors:    int = DEFAULT_FLOORS,
        num_elevators: int = DEFAULT_ELEVATORS,
        capacity:      int = DEFAULT_CAPACITY,
        clock:         SimClock | None = None,
        dwell_window_secs: float = 0.0,
    ):
        # ── clamp to allowed ranges ──────────────────────────────────────────
        self.num_floors    = max(MIN_FLOORS,    min(MAX_FLOORS,    int(num_floors)))
        self.num_elevators = max(MIN_ELEVATORS, min(MAX_ELEVATORS, int(num_elevators)))
        self.capacity      = max(1, int(capacity))
        self.dwell_window_secs = max(0.0, float(dwell_window_secs))

        # Action space: (elevator_id: 0..num_elevators-1, target_floor: 1..num_floors)
        self.action_space = [
            (e, f)
            for e in range(self.num_elevators)
            for f in range(1, self.num_floors + 1)
        ]

        self.clock = clock or SimClock()
        self.emergency_dispatcher: Any = None
        self.reset()


    # ── helpers ──────────────────────────────────────────────────────────────

    def reconfigure(
        self,
        num_floors:    int | None = None,
        num_elevators: int | None = None,
        capacity:      int | None = None,
    ) -> dict:
        """
        Hot-reconfigure the building without losing Q-table knowledge.
        Resets elevator positions and clears pending requests.
        Returns the new empty state.
        """
        if num_floors is not None:
            self.num_floors = max(MIN_FLOORS, min(MAX_FLOORS, int(num_floors)))
        if num_elevators is not None:
            self.num_elevators = max(MIN_ELEVATORS, min(MAX_ELEVATORS, int(num_elevators)))
        if capacity is not None:
            self.capacity = max(1, int(capacity))

        # Rebuild action space for the new dimensions
        self.action_space = [
            (e, f)
            for e in range(self.num_elevators)
            for f in range(1, self.num_floors + 1)
        ]

        return self.reset()

    def reset(self) -> dict:
        """Resets all elevators to floor 1, clears passengers and pending requests."""
        self.elevator_positions  = [1] * self.num_elevators
        self.elevator_passengers = [[] for _ in range(self.num_elevators)]
        # committed_targets[e] = destination floor the elevator must reach.
        # Once passengers board, this destination cannot be overridden by new Q-actions.
        self.committed_targets: list[int | None] = [None] * self.num_elevators

        self.assigned_stops      = [[] for _ in range(self.num_elevators)]
        self.pending_requests    = []
        self._next_request_id    = 1
        self.congestion_prediction = 0.20
        self.step_count  = 0
        self.last_action = None

        # ── dwell state per elevator ─────────────────────────────────────────
        # When an elevator arrives at a scheduled stop, it enters a dwell
        # (collection window).  dwell_remaining[e] counts how many simulated
        # seconds of dwell are left.  While > 0 the elevator stays put and new
        # passengers may board.
        self.dwell_remaining: list[float] = [0.0] * self.num_elevators

        # ── completed passenger statistics ───────────────────────────────────
        self.completed_passengers = 0
        self.total_wait_time_completed = 0.0
        self.empty_movement_floors = 0
        self.passenger_movement_floors = 0

        self.clock.reset()
        return self.get_state()

    def add_request(self, src_floor: int, dst_floor: int) -> bool:
        """Adds a passenger floor request to the pending queue with unassigned status."""
        src = max(1, min(self.num_floors, int(src_floor)))
        dst = max(1, min(self.num_floors, int(dst_floor)))
        if src == dst:
            return False

        if not hasattr(self, "_next_request_id"):
            self._next_request_id = 1
        req_id = self._next_request_id
        self._next_request_id += 1

        self.pending_requests.append({
            "id":          req_id,
            "src":         src,
            "dst":         dst,
            "assigned_to": None,
            "wait_time":   0.0,
            "created_at":  time.time(),
        })
        return True

    def get_unassigned_requests(self) -> list:
        """Returns all pending requests that are not yet assigned to any elevator."""
        return [r for r in self.pending_requests if r.get("assigned_to") is None]

    def simulate_projected_occupancies(self, e: int, additional_reqs: list[dict] = None) -> dict:
        """
        Simulates elevator e traversing its proposed route with currently onboard
        passengers, currently assigned requests, and any candidate additional_reqs.

        At each stop:
          1. Drops off exiting passengers.
          2. Recalculates occupancy.
          3. Boards waiting passengers that can fit (up to self.capacity).
          4. Updates individual destinations and remaining stops.

        Returns dict:
          - "feasible": bool (whether all onboard and proposed passengers fit within capacity)
          - "peak_occupancy": int (max occupancy across entire route)
          - "occupancy_at_pickup": dict[int, int] (floor -> occupancy when departing that pickup)
          - "accommodated_count": int (number of additional_reqs that can fit)
          - "stops": list[int] (projected sequence of stops)
        """
        pos = self.elevator_positions[e]
        current_onboard = list(self.elevator_passengers[e])
        assigned = [r for r in self.pending_requests if r.get("assigned_to") == e]
        cand_reqs = list(additional_reqs or [])
        all_reqs = assigned + cand_reqs

        if not current_onboard and not all_reqs:
            return {
                "feasible": True,
                "peak_occupancy": 0,
                "occupancy_at_pickup": {},
                "accommodated_count": len(cand_reqs),
                "stops": [],
            }

        # Heading determination
        heading = 0
        if self.assigned_stops[e]:
            first_stop = self.assigned_stops[e][0]
            heading = 1 if first_stop > pos else (-1 if first_stop < pos else 0)
        elif self.committed_targets[e] is not None:
            t = self.committed_targets[e]
            heading = 1 if t > pos else (-1 if t < pos else 0)

        if heading == 0:
            if current_onboard:
                heading = 1 if any(d > pos for d in current_onboard) else (-1 if any(d < pos for d in current_onboard) else 0)
            elif all_reqs:
                first_r = all_reqs[0]
                heading = 1 if first_r["src"] > pos else (-1 if first_r["src"] < pos else (1 if first_r["dst"] > pos else -1))

        # Active emergency stops for elevator e
        emg_stops = set()
        if hasattr(self, "emergency_dispatcher") and self.emergency_dispatcher:
            for emg in self.emergency_dispatcher.get_all_active():
                if emg["status"] in ("en_route", "dispatched", "assigned") and emg["elevator_id"] == e:
                    emg_stops.add(emg["floor"])

        if heading >= 0:
            up_stops = set()
            for d in current_onboard:
                if d >= pos:
                    up_stops.add(d)
            for r in all_reqs:
                if r["dst"] > r["src"] and r["src"] >= pos:
                    up_stops.add(r["src"])
                    up_stops.add(r["dst"])
                elif r["src"] >= pos:
                    up_stops.add(r["src"])
            for s in emg_stops:
                if s >= pos:
                    up_stops.add(s)

            leg1 = sorted(up_stops)

            down_stops = set()
            for d in current_onboard:
                if d < pos:
                    down_stops.add(d)
            for r in all_reqs:
                if r["dst"] < r["src"] or r["src"] < pos:
                    down_stops.add(r["src"])
                    down_stops.add(r["dst"])
            for s in emg_stops:
                if s < pos:
                    down_stops.add(s)

            leg2 = sorted(down_stops, reverse=True)
            combined = leg1 + [s for s in leg2 if s not in leg1]
        else:
            down_stops = set()
            for d in current_onboard:
                if d <= pos:
                    down_stops.add(d)
            for r in all_reqs:
                if r["dst"] < r["src"] and r["src"] <= pos:
                    down_stops.add(r["src"])
                    down_stops.add(r["dst"])
                elif r["src"] <= pos:
                    down_stops.add(r["src"])
            for s in emg_stops:
                if s <= pos:
                    down_stops.add(s)

            leg1 = sorted(down_stops, reverse=True)

            up_stops = set()
            for d in current_onboard:
                if d > pos:
                    up_stops.add(d)
            for r in all_reqs:
                if r["dst"] > r["src"] or r["src"] > pos:
                    up_stops.add(r["src"])
                    up_stops.add(r["dst"])
            for s in emg_stops:
                if s > pos:
                    up_stops.add(s)

            leg2 = sorted(up_stops)
            combined = leg1 + [s for s in leg2 if s not in leg1]

        stops = [s for idx, s in enumerate(combined) if not (idx == 0 and s == pos)]

        # Simulate route traversal
        onboard = list(current_onboard)
        waiting = list(all_reqs)
        peak_occ = len(onboard)
        occupancy_at_pickup = {}
        accommodated_cands = set()

        # Step 0: at current floor
        onboard = [d for d in onboard if d != pos]
        for r in list(waiting):
            if r["src"] == pos:
                if len(onboard) < self.capacity:
                    onboard.append(r["dst"])
                    waiting.remove(r)
                    if r in cand_reqs:
                        accommodated_cands.add(id(r))
        peak_occ = max(peak_occ, len(onboard))
        occupancy_at_pickup[pos] = len(onboard)

        # Steps along planned stops
        for st in stops:
            # 1. Process drop-offs
            onboard = [d for d in onboard if d != st]
            # 2. Process pickups
            for r in list(waiting):
                if r["src"] == st:
                    if len(onboard) < self.capacity:
                        onboard.append(r["dst"])
                        waiting.remove(r)
                        if r in cand_reqs:
                            accommodated_cands.add(id(r))
            peak_occ = max(peak_occ, len(onboard))
            occupancy_at_pickup[st] = len(onboard)

        accommodated_count = len(accommodated_cands)
        feasible = (accommodated_count == len(cand_reqs)) and (peak_occ <= self.capacity)

        return {
            "feasible": feasible,
            "peak_occupancy": peak_occ,
            "occupancy_at_pickup": occupancy_at_pickup,
            "accommodated_count": accommodated_count,
            "stops": stops,
        }

    def can_accommodate(self, e: int, req: dict) -> bool:
        """
        Verifies whether elevator e has sufficient capacity to serve req
        without exceeding capacity at any point along its journey.
        """
        sim = self.simulate_projected_occupancies(e, [req])
        return sim["feasible"]

    def max_accommodatable(self, e: int, reqs: list[dict]) -> int:
        """
        Returns how many requests from reqs elevator e can accommodate
        without exceeding capacity along its projected journey.
        """
        sim = self.simulate_projected_occupancies(e, reqs)
        return sim["accommodated_count"]

    def calculate_insertion_cost(self, e: int, req: dict) -> float:
        """
        Calculates incremental travel cost and passenger delay for inserting req
        into elevator e's existing route.
        Lower cost = more efficient and preferred.
        """
        if not self.can_accommodate(e, req):
            return 1000.0

        pos = self.elevator_positions[e]
        src = req["src"]
        dst = req["dst"]
        req_up = dst > src

        curr_occ = len(self.elevator_passengers[e])
        assigned_reqs = [r for r in self.pending_requests if r.get("assigned_to") == e]
        stops = list(self.assigned_stops[e])
        target = self.committed_targets[e]

        # 1. Idle elevator
        if curr_occ == 0 and not assigned_reqs and not stops and target is None:
            dist_to_pickup = abs(pos - src)
            ride_dist = abs(dst - src)
            startup_penalty = 15.0
            return float(dist_to_pickup * 1.5 + ride_dist * 1.0 + startup_penalty)

        # 2. Moving / Active elevator
        all_planned = [pos]
        if target is not None:
            all_planned.append(target)
        all_planned.extend(stops)
        for r in assigned_reqs:
            all_planned.extend([r["src"], r["dst"]])
        max_stop = max(all_planned)
        min_stop = min(all_planned)

        can_serve_in_flight = False

        if not req_up:
            # DOWN request (e.g. F10 -> F1)
            # If the elevator is already carrying passengers heading away, don't reverse it
            if curr_occ > 0 and pos > src and target is not None and target > pos:
                can_serve_in_flight = False
            else:
                heading_up_to_pickup = (pos <= src and ((target is not None and target >= src) or max_stop >= src))
                heading_down_from_pickup = (pos >= src and (target is None or target <= pos))
                if heading_up_to_pickup or heading_down_from_pickup:
                    can_serve_in_flight = True
        else:
            # UP request (e.g. F1 -> F9 or F4 -> F8)
            if curr_occ > 0 and pos < src and target is not None and target < pos:
                can_serve_in_flight = False
            else:
                heading_down_to_pickup = (pos >= src and ((target is not None and target <= src) or min_stop <= src))
                heading_up_from_pickup = (pos <= src and (target is None or target >= pos))
                if heading_down_to_pickup or heading_up_from_pickup:
                    can_serve_in_flight = True

        if can_serve_in_flight:
            if not req_up:
                extra_pickup = max(0, src - max_stop)
                extra_dropoff = max(0, min_stop - dst)
            else:
                extra_pickup = max(0, min_stop - src)
                extra_dropoff = max(0, dst - max_stop)

            stop_delay = 0.0
            if src not in stops and src != target:
                stop_delay += 1.5
            if dst not in stops and dst != target:
                stop_delay += 1.5

            wait_dist = abs(src - pos) * 0.5
            return float((extra_pickup + extra_dropoff) * 1.0 + stop_delay + wait_dist + 2.0)

        # 3. Elevator requires opposite-direction reversal with passengers onboard or is moving away
        return float(80.0 + abs(src - pos) * 2.0 + abs(dst - src))

    def compute_elevator_suitability(self, e: int, req: dict) -> float:
        """
        Computes heuristic suitability score for elevator e serving req.
        Higher score = more suitable. Inverted from insertion cost.
        """
        cost = self.calculate_insertion_cost(e, req)
        return max(1.0, 160.0 - cost)

    def _recompute_route(self, e: int):
        """
        Recomputes the ordered list of assigned stops for elevator e.
        Maintains directional runs (SCAN):
        - If elevator is moving UP: serves upward stops in ascending order, then downward stops.
        - If elevator is moving DOWN: serves downward stops in descending order, then upward stops.
        - If elevator is idle: heads toward pickup, then toward destination.
        Updates self.assigned_stops[e] and sets self.committed_targets[e] to the next stop.
        """
        pos = self.elevator_positions[e]
        onboard = list(self.elevator_passengers[e])
        assigned_reqs = [r for r in self.pending_requests if r.get("assigned_to") == e]

        # Active emergency stops for elevator e
        emg_stops = set()
        if hasattr(self, "emergency_dispatcher") and self.emergency_dispatcher:
            for emg in self.emergency_dispatcher.get_all_active():
                if emg["status"] in ("en_route", "dispatched", "assigned") and emg["elevator_id"] == e:
                    emg_stops.add(emg["floor"])

        if not onboard and not assigned_reqs and not emg_stops:
            if self.assigned_stops[e]:
                self.assigned_stops[e] = [s for s in self.assigned_stops[e] if s != pos]
                self.committed_targets[e] = self.assigned_stops[e][0] if self.assigned_stops[e] else None
            else:
                self.assigned_stops[e] = []
                self.committed_targets[e] = None
            return

        # Determine current heading
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
            elif emg_stops:
                first_s = list(emg_stops)[0]
                heading = 1 if first_s >= pos else -1

        if heading >= 0:
            # UP heading
            up_stops = set()
            for d in onboard:
                if d >= pos:
                    up_stops.add(d)
            for r in assigned_reqs:
                if r["dst"] > r["src"] and r["src"] >= pos:
                    up_stops.add(r["src"])
                    up_stops.add(r["dst"])
                elif r["src"] >= pos:
                    up_stops.add(r["src"])
            for s in emg_stops:
                if s >= pos:
                    up_stops.add(s)

            leg1 = sorted(up_stops)

            down_stops = set()
            for d in onboard:
                if d < pos:
                    down_stops.add(d)
            for r in assigned_reqs:
                if r["dst"] < r["src"] or r["src"] < pos:
                    down_stops.add(r["src"])
                    down_stops.add(r["dst"])
            for s in emg_stops:
                if s < pos:
                    down_stops.add(s)

            leg2 = sorted(down_stops, reverse=True)
            combined = leg1 + [s for s in leg2 if s not in leg1]
        else:
            # DOWN heading
            down_stops = set()
            for d in onboard:
                if d <= pos:
                    down_stops.add(d)
            for r in assigned_reqs:
                if r["dst"] < r["src"] and r["src"] <= pos:
                    down_stops.add(r["src"])
                    down_stops.add(r["dst"])
                elif r["src"] <= pos:
                    down_stops.add(r["src"])
            for s in emg_stops:
                if s <= pos:
                    down_stops.add(s)

            leg1 = sorted(down_stops, reverse=True)

            up_stops = set()
            for d in onboard:
                if d > pos:
                    up_stops.add(d)
            for r in assigned_reqs:
                if r["dst"] > r["src"] or r["src"] > pos:
                    up_stops.add(r["src"])
                    up_stops.add(r["dst"])
            for s in emg_stops:
                if s > pos:
                    up_stops.add(s)

            leg2 = sorted(up_stops)
            combined = leg1 + [s for s in leg2 if s not in leg1]

        final_stops = [s for idx, s in enumerate(combined) if not (idx == 0 and s == pos)]
        self.assigned_stops[e] = final_stops
        self.committed_targets[e] = final_stops[0] if final_stops else None

    def get_dispatch_candidates(self, req: dict) -> tuple[list, list]:
        """
        Evaluates all eligible elevators for the given request.
        Returns (valid_actions, heuristic_scores):
          valid_actions: list of (elevator_id, target_floor)
          heuristic_scores: list of float scores matching valid_actions
        """
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

    def assign_request(self, req: dict, elevator_id: int):
        """
        Assigns an unassigned passenger request to exactly one elevator.
        If elevator is already at the pickup floor, boards immediately.
        Otherwise, inserts the request into elevator's ordered stops.
        """
        req["assigned_to"] = elevator_id
        curr_pos = self.elevator_positions[elevator_id]
        src = req["src"]
        dst = req["dst"]

        if curr_pos == src and len(self.elevator_passengers[elevator_id]) < self.capacity:
            self.elevator_passengers[elevator_id].append(dst)
            if req in self.pending_requests:
                self.pending_requests.remove(req)

        self._recompute_route(elevator_id)

    def get_state(self) -> dict:
        """Returns the current state dictionary of the building environment."""
        floor_demand = [0] * self.num_floors
        for req in self.pending_requests:
            idx = req["src"] - 1
            if 0 <= idx < self.num_floors:
                floor_demand[idx] += 1

        occupancies = [len(p) for p in self.elevator_passengers]

        return {
            "elevator_positions":   list(self.elevator_positions),
            "elevator_occupancy":   occupancies,
            "elevator_targets":     list(self.committed_targets),
            "committed_targets":    list(self.committed_targets),
            "assigned_stops":       [list(s) for s in self.assigned_stops],
            "pending_requests":     len(self.pending_requests),
            "floor_demand":         floor_demand,
            "congestion_prediction": float(round(self.congestion_prediction, 4)),
            "avg_wait_time":        self.get_wait_times(),
            "step_count":           self.step_count,
            "last_action":          self.last_action,
            # ── expose config so the frontend knows dimensions ───────────────
            "num_floors":           self.num_floors,
            "num_elevators":        self.num_elevators,
            "capacity":             self.capacity,
            # ── simulation clock ─────────────────────────────────────────────
            "sim_time":             round(self.clock.now(), 2),
            # ── dwell windows ────────────────────────────────────────────────
            "dwell_remaining":      [round(d, 1) for d in self.dwell_remaining],
            # ── completed passenger stats ────────────────────────────────────
            "completed_passengers": self.completed_passengers,
            # ── energy & movement metrics ────────────────────────────────────
            "empty_movement_floors":     self.empty_movement_floors,
            "passenger_movement_floors": self.passenger_movement_floors,
        }

    def get_action_space(self) -> list:
        """Returns list of all valid (elevator_id, target_floor) action tuples."""
        return list(self.action_space)

    def get_wait_times(self) -> float:
        if not self.pending_requests:
            return 0.0
        total_wait = sum(r["wait_time"] for r in self.pending_requests)
        return round(total_wait / len(self.pending_requests), 2)

    def _pickup_at_floor(self, e: int, floor: int) -> int:
        """
        Picks up passengers waiting at floor into elevator e.
        Only picks up requests assigned to elevator e, or unassigned requests matching heading.
        """
        t_floor = self.committed_targets[e]
        going_up = t_floor is not None and t_floor > floor
        going_down = t_floor is not None and t_floor < floor
        is_empty = len(self.elevator_passengers[e]) == 0

        remaining_pending = []
        picked = 0
        for req in self.pending_requests:
            assigned_to = req.get("assigned_to")
            is_assigned_to_e = (assigned_to == e)
            is_unassigned = (assigned_to is None)

            if not (is_assigned_to_e or is_unassigned):
                remaining_pending.append(req)
                continue

            req_up = req["dst"] > req["src"]
            req_dn = req["dst"] < req["src"]
            dir_match = is_assigned_to_e or is_empty or (going_up and req_up) or (going_down and req_dn) or (not going_up and not going_down)

            if (
                req["src"] == floor
                and dir_match
                and len(self.elevator_passengers[e]) < self.capacity
            ):
                self.elevator_passengers[e].append(req["dst"])
                picked += 1
                if is_empty:
                    is_empty = False
                    going_up = req_up
                    going_down = req_dn
            else:
                remaining_pending.append(req)

        self.pending_requests = remaining_pending
        return picked

    def _is_scheduled_stop(self, e: int, floor: int) -> bool:
        """Returns True if `floor` is a scheduled pickup, drop-off, or emergency stop for elevator e."""
        # Drop-off?
        if floor in self.elevator_passengers[e]:
            return True
        # Pickup?
        for req in self.pending_requests:
            if req.get("assigned_to") == e and req["src"] == floor:
                return True
        # Emergency stop?
        if hasattr(self, "emergency_dispatcher") and self.emergency_dispatcher:
            for emg in self.emergency_dispatcher.get_all_active():
                if emg["elevator_id"] == e and emg["floor"] == floor:
                    return True
        # In assigned_stops list?
        if floor in self.assigned_stops[e]:
            return True
        return False

    def _needs_dwell_window(self, e: int, floor: int) -> bool:
        """Returns True if floor is a scheduled stop (pickup, drop-off, or emergency) requiring dwell."""
        return self._is_scheduled_stop(e, floor)

    def step(self, action=None) -> tuple:
        """
        Executes one environment step (= 1 simulated second).

        Behaviour per elevator each step:
        1. If dwelling (collection window active), decrement dwell_remaining.
           During dwell: process pickups and drop-offs; do NOT move.
        2. If NOT dwelling:
           a. If at a scheduled stop floor, start collection window.
           b. Otherwise, move 1 floor toward committed_targets[e].
           c. After moving, process drop-offs and pickups.
           d. If the new floor was a scheduled stop, start a collection window.
           e. Intermediate floors do NOT trigger dwell (1s transit).
        3. Idle elevators (no passengers, no stops, no targets) stay stationary.
        """
        prev_state = self.get_state()
        self.step_count += 1
        if self.clock.deterministic:
            self.clock.advance(TRAVEL_SECS)

        dispatched_elevator_id = None
        if action is not None:
            dispatched_elevator_id = int(action[0])
            target_floor = int(action[1])
            dispatched_elevator_id = max(0, min(self.num_elevators - 1, dispatched_elevator_id))
            target_floor = max(1, min(self.num_floors, target_floor))
            self.last_action = {"elevator_id": dispatched_elevator_id, "target_floor": target_floor}

        total_picked_up = 0
        total_dropped_off = 0
        total_movement_dist = 0

        # Advance elevators with active tasks
        for e in range(self.num_elevators):
            curr_floor = self.elevator_positions[e]

            # ── Dwelling (collection window) ─────────────────────────────────
            if self.dwell_remaining[e] > 0:
                self.dwell_remaining[e] -= TRAVEL_SECS
                if self.dwell_remaining[e] < 0.01:
                    self.dwell_remaining[e] = 0.0

                # During dwell: process drop-offs and pickups at current floor
                passengers_before = len(self.elevator_passengers[e])
                self.elevator_passengers[e] = [
                    dst for dst in self.elevator_passengers[e] if dst != curr_floor
                ]
                dropped = passengers_before - len(self.elevator_passengers[e])
                total_dropped_off += dropped
                if dropped > 0:
                    self.completed_passengers += dropped

                picked = self._pickup_at_floor(e, curr_floor)
                total_picked_up += picked

                # Remove curr_floor from assigned_stops if present
                if curr_floor in self.assigned_stops[e]:
                    self.assigned_stops[e] = [s for s in self.assigned_stops[e] if s != curr_floor]

                # When dwell ends, recompute route for the remaining journey
                if self.dwell_remaining[e] <= 0:
                    self._recompute_route(e)
                continue

            # ── Not dwelling ─────────────────────────────────────────────────
            # Check if current floor is a scheduled stop before moving
            was_scheduled_here = self._is_scheduled_stop(e, curr_floor)

            # 1. Board any assigned passenger waiting at current floor before moving
            picked = self._pickup_at_floor(e, curr_floor)
            total_picked_up += picked
            if picked > 0:
                self._recompute_route(e)

            t_floor = self.committed_targets[e]
            if t_floor is None:
                # Idle elevator: remains strictly stationary! Movement = 0.
                continue

            step_movement = 0
            if curr_floor < t_floor:
                self.elevator_positions[e] += 1
                step_movement = 1
            elif curr_floor > t_floor:
                self.elevator_positions[e] -= 1
                step_movement = 1
            total_movement_dist += step_movement
            if step_movement > 0:
                if len(self.elevator_passengers[e]) == 0:
                    self.empty_movement_floors += 1
                else:
                    self.passenger_movement_floors += 1
            curr_floor = self.elevator_positions[e]

            # 4. Drop off passengers whose destination is curr_floor
            passengers_before = len(self.elevator_passengers[e])
            self.elevator_passengers[e] = [
                dst for dst in self.elevator_passengers[e] if dst != curr_floor
            ]
            dropped = passengers_before - len(self.elevator_passengers[e])
            total_dropped_off += dropped
            if dropped > 0:
                self.completed_passengers += dropped

            # 5. Pick up passengers waiting at arrival floor
            picked = self._pickup_at_floor(e, curr_floor)
            total_picked_up += picked

            # 6. Check if this arrival was a scheduled stop
            was_scheduled = (
                dropped > 0
                or picked > 0
                or curr_floor in self.assigned_stops[e]
                or (t_floor is not None and t_floor == curr_floor)
                or self._is_scheduled_stop(e, curr_floor)
            )

            # Remove curr_floor from assigned_stops if arrived
            if curr_floor in self.assigned_stops[e]:
                self.assigned_stops[e] = [s for s in self.assigned_stops[e] if s != curr_floor]

            # 7. Start collection window at scheduled stops (pickup, drop-off, emergency)
            #    Intermediate floors do NOT trigger dwell.
            #    Combined pickup+dropoff triggers single window.
            if was_scheduled and self.dwell_window_secs > 0:
                self.dwell_remaining[e] = self.dwell_window_secs

            # 8. Recompute route to keep assigned_stops and committed_targets synchronized
            self._recompute_route(e)

        # 6. Age remaining pending requests
        for req in self.pending_requests:
            req["wait_time"] += TRAVEL_SECS

        # 7. Calculate reward
        reward = self._calculate_reward(
            action if action is not None else (0, 1),
            prev_state,
            total_movement_dist,
            total_picked_up,
            total_dropped_off
        )

        new_state = self.get_state()
        info = {
            "picked_up":   total_picked_up,
            "dropped_off": total_dropped_off,
            "movement":    total_movement_dist,
        }
        return new_state, reward, info

    def _calculate_reward(
        self,
        action,
        prev_state,
        movement_dist: int = 1,
        picked_up:     int = 0,
        dropped_off:   int = 0,
    ) -> float:
        """
        Multi-objective RL reward:
        R = -(α·wait + β·movement + γ·unused_capacity) + service_bonuses
        """
        alpha, beta, gamma = 1.0, 0.5, 0.3

        avg_wait      = self.get_wait_times()
        pending_count = len(self.pending_requests)
        elevator_id   = int(action[0])
        curr_occ      = len(self.elevator_passengers[elevator_id])
        unused_cap    = max(0, self.capacity - curr_occ)

        wait_penalty = alpha * (avg_wait * 0.5 + pending_count)
        move_penalty = beta  * movement_dist
        cap_penalty  = gamma * (unused_cap / self.capacity)

        reward  = -(wait_penalty + move_penalty + cap_penalty)
        reward += picked_up * 2.0 + dropped_off * 3.0

        return round(float(reward), 3)

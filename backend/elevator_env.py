"""
Elevator Simulation Environment.
Simulates a configurable building (2–20 floors, 1–5 elevators) with passenger
boarding/alighting, queue dynamics, wait time calculations, and multi-objective
RL reward formulation.

Caps:
  - num_floors  : 2 ≤ n ≤ 20  (hard cap: 20)
  - num_elevators: 1 ≤ n ≤ 5   (hard cap:  5)
"""

import time

# ── absolute limits ──────────────────────────────────────────────────────────
MAX_FLOORS     = 20
MAX_ELEVATORS  = 5
MIN_FLOORS     = 2
MIN_ELEVATORS  = 1

# ── default values ───────────────────────────────────────────────────────────
DEFAULT_FLOORS     = 10
DEFAULT_ELEVATORS  = 3
DEFAULT_CAPACITY   = 8


class ElevatorEnvironment:
    def __init__(
        self,
        num_floors:    int = DEFAULT_FLOORS,
        num_elevators: int = DEFAULT_ELEVATORS,
        capacity:      int = DEFAULT_CAPACITY,
    ):
        # ── clamp to allowed ranges ──────────────────────────────────────────
        self.num_floors    = max(MIN_FLOORS,    min(MAX_FLOORS,    int(num_floors)))
        self.num_elevators = max(MIN_ELEVATORS, min(MAX_ELEVATORS, int(num_elevators)))
        self.capacity      = max(1, int(capacity))

        # Action space: (elevator_id: 0..num_elevators-1, target_floor: 1..num_floors)
        self.action_space = [
            (e, f)
            for e in range(self.num_elevators)
            for f in range(1, self.num_floors + 1)
        ]

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
        self.committed_targets   = [None] * self.num_elevators
        self.assigned_stops      = [[] for _ in range(self.num_elevators)]
        self.pending_requests    = []
        self.congestion_prediction = 0.20
        self.step_count  = 0
        self.last_action = None
        return self.get_state()

    def add_request(self, src_floor: int, dst_floor: int) -> bool:
        """Adds a passenger floor request to the pending queue with unassigned status."""
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
            "created_at":  time.time(),
        })
        return True

    def get_unassigned_requests(self) -> list:
        """Returns all pending requests that are not yet assigned to any elevator."""
        return [r for r in self.pending_requests if r.get("assigned_to") is None]

    def can_accommodate(self, e: int, req: dict) -> bool:
        """
        Verifies whether elevator e has sufficient capacity to serve req
        without exceeding capacity at any point along its journey.
        """
        src = req["src"]
        dst = req["dst"]
        low, high = min(src, dst), max(src, dst)

        # Check passenger count at every floor segment in [low, high - 1]
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

        # 1. Idle elevator
        if curr_occ == 0 and not assigned_reqs and not self.assigned_stops[e]:
            dist_to_pickup = abs(pos - src)
            ride_dist = abs(dst - src)
            # Startup penalty + physical travel distance + wait time
            return float(dist_to_pickup * 1.5 + ride_dist * 1.0 + 15.0)

        # 2. Active elevator: check directional compatibility
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
            # Batching along existing direction
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
            stop_delay = 2.0  # delay added to existing passengers for stops
            return float(incremental_dist * 1.0 + wait_time * 1.0 + stop_delay * 1.5)

        # 3. Moving elevator requiring direction reversal or pickup behind elevator
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

        if not onboard and not assigned_reqs:
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
            # DOWN heading
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

    def step(self, action=None) -> tuple:
        """
        Executes one environment step.
        Advances all elevators that have active trips (onboard passengers or assigned pickups).
        Idle elevators (no onboard passengers, no assigned pickups) remain strictly stationary.
        """
        prev_state = self.get_state()
        self.step_count += 1

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

            # 1. Board any assigned passenger waiting at current floor before moving
            picked = self._pickup_at_floor(e, curr_floor)
            total_picked_up += picked
            if picked > 0:
                self._recompute_route(e)

            t_floor = self.committed_targets[e]
            if t_floor is None:
                # Idle elevator: remains strictly stationary! Movement = 0.
                continue

            # 2. Move 1 floor toward target
            step_movement = 0
            if curr_floor < t_floor:
                self.elevator_positions[e] += 1
                step_movement = 1
            elif curr_floor > t_floor:
                self.elevator_positions[e] -= 1
                step_movement = 1
            total_movement_dist += step_movement
            curr_floor = self.elevator_positions[e]

            # 3. Drop off passengers whose destination is curr_floor
            passengers_before = len(self.elevator_passengers[e])
            self.elevator_passengers[e] = [
                dst for dst in self.elevator_passengers[e] if dst != curr_floor
            ]
            dropped = passengers_before - len(self.elevator_passengers[e])
            total_dropped_off += dropped

            # 4. Pick up passengers waiting at arrival floor
            total_picked_up += self._pickup_at_floor(e, curr_floor)

            # 5. Remove curr_floor from assigned_stops if arrived
            if curr_floor in self.assigned_stops[e]:
                self.assigned_stops[e] = [s for s in self.assigned_stops[e] if s != curr_floor]

            # 6. Recompute route to keep assigned_stops and committed_targets synchronized
            self._recompute_route(e)

        # 6. Age remaining pending requests
        for req in self.pending_requests:
            req["wait_time"] += 1.0

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

"""
Emergency Priority Dispatcher for AI Elevator System.
Handles floor-specific emergency summon requests with highest priority,
bypassing the normal 2-second batch window and selecting the most suitable elevator
without compromising safety or passenger integrity.
"""

import time
import threading


class EmergencyDispatcher:
    """
    Manages emergency requests, selects the optimal elevator based on lowest
    safe response time, prevents duplicate dispatches, implements deterministic tie-breaking,
    and protects emergency routes from normal Q-learning overrides.
    """

    def __init__(self, env):
        self.env = env
        self.lock = threading.RLock()
        self.active_emergencies = {}  # id -> dict
        self._next_id = 1
        self.event_queue = []

    def reset(self):
        """Clears all emergencies and event logs."""
        with self.lock:
            self.active_emergencies.clear()
            self.event_queue.clear()

    def _enrich_emergency(self, emg: dict) -> dict:
        """
        Dynamically enriches an emergency record with real-time elevator telemetry:
        current floor, live distance, target floor, and movement direction.
        """
        d = dict(emg)
        e = d.get("elevator_id")
        floor = d.get("floor", 1)

        if e is not None and e < self.env.num_elevators:
            curr_pos = self.env.elevator_positions[e]
            d["current_floor"] = curr_pos
            d["distance"] = abs(curr_pos - floor)
            d["target_floor"] = self.env.committed_targets[e]

            # Live status derivation
            if curr_pos == floor:
                d["direction"] = "ARRIVED"
                if d.get("status") not in ("resolved", "unassigned"):
                    d["status"] = "arrived"
            elif curr_pos < floor:
                d["direction"] = "UP"
                if d.get("status") not in ("resolved", "unassigned"):
                    d["status"] = "en_route"
            else:
                d["direction"] = "DOWN"
                if d.get("status") not in ("resolved", "unassigned"):
                    d["status"] = "en_route"
        else:
            d["current_floor"] = None
            d["distance"] = None
            d["direction"] = "NONE"
            d["target_floor"] = None

        return d

    def get_locked_elevator_ids(self) -> set:
        """
        Returns elevator indices currently committed to an active emergency.
        These elevators cannot be overridden or redirected by normal Q-learning actions.
        """
        with self.lock:
            return {
                emg["elevator_id"]
                for emg in self.active_emergencies.values()
                if emg["status"] in ("en_route", "dispatched", "arrived", "assigned")
                and emg["elevator_id"] is not None
            }

    def get_active_emergency_for_floor(self, floor: int) -> dict | None:
        """
        Returns active unresolved emergency on given floor if present.
        Used to prevent duplicate emergency requests for the same floor.
        """
        with self.lock:
            for emg in self.active_emergencies.values():
                if emg["floor"] == floor and emg["status"] in ("en_route", "dispatched", "arrived", "assigned"):
                    return self._enrich_emergency(emg)
            return None

    def get_all_active(self) -> list:
        """Returns list of all active (unresolved) emergencies with live telemetry."""
        with self.lock:
            return [
                self._enrich_emergency(emg)
                for emg in self.active_emergencies.values()
                if emg["status"] in ("en_route", "dispatched", "arrived", "assigned")
            ]

    def get_status_summary(self) -> dict:
        """Returns summary dictionary for API serialization."""
        with self.lock:
            active = [
                self._enrich_emergency(emg)
                for emg in self.active_emergencies.values()
                if emg["status"] in ("en_route", "dispatched", "arrived", "assigned")
            ]
            has_active = len(active) > 0
            primary = active[0] if has_active else None
            return {
                "has_active_emergency": has_active,
                "active_emergencies": active,
                "current_emergency": primary,
            }

    def find_best_elevator(self, floor: int) -> tuple[int | None, float, str]:
        """
        Selects elevator with lowest safe estimated response time.
        Considers position, direction, occupancy, capacity, and current routes.
        Implements deterministic multi-criteria tie-breaking.
        Returns: (elevator_id or None, response_cost, rationale)
        """
        locked = self.get_locked_elevator_ids()
        candidates = []

        for e in range(self.env.num_elevators):
            # 1. Skip if already locked to another emergency
            if e in locked:
                continue

            pos = self.env.elevator_positions[e]
            occ = len(self.env.elevator_passengers[e])
            capacity = self.env.capacity
            stops = list(self.env.assigned_stops[e])
            target = self.env.committed_targets[e]

            # 2. Safety: capacity limit check
            if occ >= capacity:
                continue

            # Case A: Elevator is already at the emergency floor
            if pos == floor:
                cost = 0.0
                rationale = f"Lift {e + 1} is already at Floor {floor}"
                candidates.append((cost, occ, len(stops), e, rationale))
                continue

            # Case B: Completely idle elevator
            is_idle = (occ == 0 and not stops and target is None)
            if is_idle:
                dist = abs(pos - floor)
                cost = dist * 1.0  # direct distance
                rationale = f"Idle Lift {e + 1} at Floor {pos} (distance: {dist})"
                candidates.append((cost, occ, len(stops), e, rationale))
                continue

            # Case C: Moving elevator heading toward emergency floor along existing route
            heading_up = (target is not None and target > pos) or (stops and stops[0] > pos)
            heading_down = (target is not None and target < pos) or (stops and stops[0] < pos)

            if heading_up and pos < floor:
                furthest = max(stops or [target or pos])
                # Only if floor is on the way or extends the current upward run
                if floor <= furthest or occ == 0:
                    dist = floor - pos
                    cost = dist * 1.0 + 0.3  # small bias preferring pure idle if equal distance
                    rationale = f"En-route upward Lift {e + 1} at Floor {pos} (distance: {dist})"
                    candidates.append((cost, occ, len(stops), e, rationale))
            elif heading_down and pos > floor:
                furthest = min(stops or [target or pos])
                # Only if floor is on the way or extends the current downward run
                if floor >= furthest or occ == 0:
                    dist = pos - floor
                    cost = dist * 1.0 + 0.3
                    rationale = f"En-route downward Lift {e + 1} at Floor {pos} (distance: {dist})"
                    candidates.append((cost, occ, len(stops), e, rationale))

        if not candidates:
            return None, float("inf"), "No suitable elevator available without capacity or route violation"

        # Deterministic sorting: (cost, occupancy, num_stops, elevator_id)
        candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
        best_candidate = candidates[0]
        best_cost, best_occ, best_num_stops, best_e, base_rationale = best_candidate

        # Check for ties at lowest cost
        tied_at_cost = [c for c in candidates if abs(c[0] - best_cost) < 1e-4]
        if len(tied_at_cost) > 1:
            tie_ids = [f"Lift {c[3] + 1}" for c in tied_at_cost]
            tie_reason = (
                f"Tie-breaker among {', '.join(tie_ids)} (all cost {best_cost:.1f}): "
                f"selected Lift {best_e + 1} (lowest occupancy: {best_occ}, stops: {best_num_stops})"
            )
            full_rationale = f"{base_rationale} [{tie_reason}]"
        else:
            full_rationale = base_rationale

        return best_e, best_cost, full_rationale

    def trigger_emergency(self, floor: int) -> dict:
        """Alias for dispatch_emergency."""
        return self.dispatch_emergency(floor)

    def dispatch_emergency(self, floor: int) -> dict:
        """
        Dispatches an emergency priority request for the given floor.
        Returns result dict with emergency metadata and status.
        """
        floor = max(1, min(self.env.num_floors, int(floor)))

        with self.lock:
            # 1. Prevent duplicate emergency requests for the same floor
            existing = self.get_active_emergency_for_floor(floor)
            if existing:
                return {
                    "status": "duplicate",
                    "message": f"Emergency on Floor {floor} is already active and handled by Lift {(existing['elevator_id'] or 0) + 1}",
                    "emergency": existing,
                }

            # 2. Find best suitable elevator
            best_e, cost, rationale = self.find_best_elevator(floor)

            emg_id = self._next_id
            self._next_id += 1
            now = time.time()

            if best_e is None:
                # No suitable lift can safely respond
                emg = {
                    "id": emg_id,
                    "floor": floor,
                    "elevator_id": None,
                    "status": "unassigned",
                    "created_at": now,
                    "arrived_at": None,
                    "resolved_at": None,
                    "rationale": "No elevator could safely respond without violating capacity or route safety",
                }
                self.active_emergencies[emg_id] = emg
                self.event_queue.append(f"[EMERGENCY] Request at Floor {floor} UNASSIGNED: no suitable lift available")
                return {
                    "status": "no_lift_available",
                    "message": f"No elevator can safely reach Floor {floor} at this time",
                    "emergency": self._enrich_emergency(emg),
                }

            # 3. Elevator found - assign and configure route
            pos = self.env.elevator_positions[best_e]
            is_already_there = (pos == floor)
            status = "arrived" if is_already_there else "en_route"
            arrived_at = now if is_already_there else None

            emg = {
                "id": emg_id,
                "floor": floor,
                "elevator_id": best_e,
                "status": status,
                "created_at": now,
                "arrived_at": arrived_at,
                "resolved_at": None,
                "rationale": rationale,
            }
            self.active_emergencies[emg_id] = emg

            if is_already_there:
                self.event_queue.append(f"[EMERGENCY] Lift {best_e + 1} is ALREADY at Emergency Floor {floor}!")
            else:
                # Insert emergency stop with top priority
                current_stops = list(self.env.assigned_stops[best_e])
                if floor not in current_stops:
                    if not current_stops:
                        self.env.assigned_stops[best_e] = [floor]
                    else:
                        # Insert in direction order
                        if floor > pos:
                            # upward run
                            before_stops = [s for s in current_stops if s < floor and s > pos]
                            after_stops = [s for s in current_stops if s >= floor or s <= pos]
                            self.env.assigned_stops[best_e] = before_stops + [floor] + [s for s in after_stops if s != floor]
                        else:
                            # downward run
                            before_stops = [s for s in current_stops if s > floor and s < pos]
                            after_stops = [s for s in current_stops if s <= floor or s >= pos]
                            self.env.assigned_stops[best_e] = before_stops + [floor] + [s for s in after_stops if s != floor]

                # Update next committed target to emergency floor if closer
                self.env.committed_targets[best_e] = self.env.assigned_stops[best_e][0]

                self.event_queue.append(f"[EMERGENCY] Floor {floor} request assigned to Lift {best_e + 1} ({rationale})")

            return {
                "status": "success",
                "message": f"Emergency priority assigned to Lift {best_e + 1}",
                "emergency": self._enrich_emergency(emg),
            }

    def resolve_emergency(self, emergency_id: int | None = None, floor: int | None = None) -> dict:
        """
        Resolves an active emergency.
        Restores normal safe route policy for the elevator and preserves pending requests.
        """
        with self.lock:
            target_emg = None
            if emergency_id is not None:
                target_emg = self.active_emergencies.get(emergency_id)
            elif floor is not None:
                for emg in self.active_emergencies.values():
                    if emg["floor"] == floor and emg["status"] in ("en_route", "dispatched", "arrived", "assigned"):
                        target_emg = emg
                        break
            else:
                # Resolve most recent active emergency
                active = [
                    emg for emg in self.active_emergencies.values()
                    if emg["status"] in ("en_route", "dispatched", "arrived", "assigned")
                ]
                if active:
                    target_emg = active[-1]

            if not target_emg or target_emg["status"] == "resolved":
                return {
                    "status": "not_found",
                    "message": "No active emergency found to resolve",
                }

            target_emg["status"] = "resolved"
            target_emg["resolved_at"] = time.time()
            e = target_emg["elevator_id"]

            if e is not None and e < self.env.num_elevators:
                # Remove emergency floor from assigned stops if present
                self.env.assigned_stops[e] = [s for s in self.env.assigned_stops[e] if s != target_emg["floor"]]
                self.env._recompute_route(e)

            self.event_queue.append(f"[RESOLVED] Emergency on Floor {target_emg['floor']} resolved. Normal dispatch restored.")

            return {
                "status": "success",
                "message": f"Emergency on Floor {target_emg['floor']} successfully resolved",
                "emergency": self._enrich_emergency(target_emg),
            }

    def tick(self) -> list:
        """
        Called during each simulator step.
        Detects arrival of assigned elevators at emergency floors and updates status.
        Returns list of new event strings for logging.
        """
        events = []
        with self.lock:
            for emg in self.active_emergencies.values():
                if emg["status"] in ("en_route", "dispatched", "assigned") and emg["elevator_id"] is not None:
                    e = emg["elevator_id"]
                    if e < self.env.num_elevators and self.env.elevator_positions[e] == emg["floor"]:
                        emg["status"] = "arrived"
                        emg["arrived_at"] = time.time()
                        event_msg = f"[EMERGENCY ARRIVED] Lift {e + 1} has ARRIVED at Emergency Floor {emg['floor']}"
                        self.event_queue.append(event_msg)
                        events.append(event_msg)

            # Drain internal event queue
            events.extend(self.event_queue)
            self.event_queue.clear()
        return events

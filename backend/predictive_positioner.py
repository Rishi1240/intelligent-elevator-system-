"""
Demand-Aware Predictive Elevator Pre-Positioning Controller.

Implements demand-aware, capacity-aware pre-positioning when 5-minute-ahead
LSTM predicted congestion reaches or exceeds 55% (>= 0.55) with two consecutive
valid confirmations.

Reevaluates positioning throughout the entire simulation lifecycle:
- When a new LSTM forecast is computed.
- When passenger requests are submitted or batch-dispatched.
- When an elevator picks up passengers or completes a destination stop.
- When an elevator finishes its committed passenger route and becomes idle.
- When predictive assignments are completed, cancelled, or released.
- When emergencies are resolved.
- When the simulation clock advances.

Evaluates every configured floor using floor-specific demand forecasts (demand_f1_t5..demand_fn_t5).
Allocates available eligible elevators according to unmet demand and service capacity:
- Prioritizes highest unmet demand floor first.
- Sends only the number of elevators actually needed (1 if sufficient, 2 or 3 only if demand justifies).
- Distributes elevators across multiple high-demand floors when appropriate.
- Preserves valid assignments without reissuing commands.
- Never interrupts passenger journeys or emergencies.
- Always keeps at least one elevator available for real requests where operationally possible.
- Hysteresis release threshold defaults to 45% (<= 0.45).
"""

from collections import deque
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple


class PredictivePositioner:
    """
    Centralized controller for demand-aware forecast-driven elevator pre-positioning.
    Separated from normal passenger dispatch and emergency dispatch.
    """

    def __init__(
        self,
        high_congestion_threshold: float = 0.55,
        release_threshold: float = 0.45,
        confirmation_cycles_required: int = 2,
        min_demand_threshold: float = 1.0,
        service_capacity_per_elevator: float = 4.0,
        target_reposition_count: int = 3,
        reposition_cooldown_secs: float = 60.0,
        target_change_confirmation_cycles: int = 2,
    ):
        # Configurable thresholds & parameters
        self.high_congestion_threshold = float(high_congestion_threshold)
        self.release_threshold = float(release_threshold)
        self.confirmation_cycles_required = int(confirmation_cycles_required)
        self.min_demand_threshold = float(min_demand_threshold)
        self.service_capacity_per_elevator = float(service_capacity_per_elevator)
        self.target_reposition_count = int(target_reposition_count)
        self.reposition_cooldown_secs = float(reposition_cooldown_secs)
        self.target_change_confirmation_cycles = int(target_change_confirmation_cycles)

        # Operational state
        self.consecutive_high_count: int = 0
        self.active_target_floor: Optional[int] = None
        self.active_target_demand: float = 0.0
        self.pending_target_floor: Optional[int] = None
        self.pending_target_count: int = 0

        # Assigned elevators: elevator_id -> target_floor (currently in transit)
        self.assigned_elevators: Dict[int, int] = {}
        # Assignment reasons: elevator_id -> reason_str
        self.assigned_reasons: Dict[int, str] = {}
        # Stationed elevators: elevator_id -> floor where it completed predictive positioning
        self.stationed_elevators: Dict[int, int] = {}
        # Cooldown per elevator: elevator_id -> monotonic expiry timestamp for repositioning away
        self.cooldown_expiry: Dict[int, float] = {}

        # Telemetry, evaluation snapshots & diagnostic log ring buffer
        self.status: str = "inactive"
        self.status_reason: str = f"Congestion below activation threshold (< {int(self.high_congestion_threshold * 100)}%)"
        self.last_decision_log: str = ""
        self.last_eligible_count: int = 0
        self.last_floor_evaluations: List[dict] = []
        self.evaluation_diagnostics: deque = deque(maxlen=25)

    def reset(self):
        """Resets all internal positioning state."""
        self.consecutive_high_count = 0
        self.active_target_floor = None
        self.active_target_demand = 0.0
        self.pending_target_floor = None
        self.pending_target_count = 0
        self.assigned_elevators.clear()
        self.assigned_reasons.clear()
        self.stationed_elevators.clear()
        self.cooldown_expiry.clear()
        self.status = "inactive"
        self.status_reason = "System reset"
        self.last_decision_log = ""
        self.last_eligible_count = 0
        self.last_floor_evaluations.clear()
        self.evaluation_diagnostics.clear()

    def cancel_assignment(self, elevator_id: int):
        """
        Cancels any predictive positioning assignment for a specific elevator.
        Called when an elevator is assigned a real passenger request or emergency.
        Clears cooldown so the elevator is a freshly available resource when it
        completes its real passenger mission.
        """
        if elevator_id in self.assigned_elevators:
            del self.assigned_elevators[elevator_id]
        if elevator_id in self.assigned_reasons:
            del self.assigned_reasons[elevator_id]
        if elevator_id in self.stationed_elevators:
            del self.stationed_elevators[elevator_id]
        # Clear cooldown so previous predictive repositioning does not block future availability
        self.cooldown_expiry.pop(elevator_id, None)

    def has_active_movement(self) -> bool:
        """Returns True if any elevator is actively assigned and travelling to reposition."""
        return len(self.assigned_elevators) > 0

    def get_assigned_elevators(self) -> List[int]:
        """Returns list of elevator IDs currently assigned to predictive positioning."""
        return sorted(list(self.assigned_elevators.keys()))

    def get_status_dict(self) -> dict:
        """Returns comprehensive telemetry dictionary for API and dashboard."""
        return {
            "status": self.status,
            "reason": self.status_reason,
            "active_target_floor": self.active_target_floor,
            "active_target_demand": round(self.active_target_demand, 2),
            "assigned_elevators": self.get_assigned_elevators(),
            "assigned_allocations": [
                {
                    "elevator_id": e,
                    "target_floor": tgt,
                    "reason": self.assigned_reasons.get(e, f"Assigned to F{tgt}"),
                }
                for e, tgt in sorted(self.assigned_elevators.items())
            ],
            "stationed_elevators": [
                {"elevator_id": e, "floor": fl}
                for e, fl in sorted(self.stationed_elevators.items())
            ],
            "eligible_elevators_count": self.last_eligible_count,
            "consecutive_high_count": self.consecutive_high_count,
            "confirmation_required": self.confirmation_cycles_required,
            "high_threshold": self.high_congestion_threshold,
            "release_threshold": self.release_threshold,
            "min_demand_threshold": self.min_demand_threshold,
            "service_capacity_per_elevator": self.service_capacity_per_elevator,
            "target_count": self.target_reposition_count,
            "last_decision_log": self.last_decision_log,
            "floor_evaluations": self.last_floor_evaluations,
            "recent_diagnostics": list(self.evaluation_diagnostics),
        }

    def evaluate(
        self,
        env: Any,
        congestion_t5: float,
        floor_demand_t5: Optional[List[float]],
        locked_elevator_ids: Set[int],
        current_time: Optional[float] = None,
        model_loaded: bool = True,
        trigger_event: str = "periodic",
    ) -> Tuple[bool, str, List[dict]]:
        """
        Evaluates current prediction against demand-aware positioning rules.
        Reconsiders allocation continuously during simulation without freezing on active state.

        Returns:
            (activated: bool, reason: str, actions: list of {elevator_id, target_floor, reason, ...})
        """
        now = time.monotonic() if current_time is None else current_time
        num_floors = getattr(env, "num_floors", 10)
        num_elevators = getattr(env, "num_elevators", 3)

        # ── 1. Model Validity & Floor Demand Check ────────────────────────────
        if not model_loaded:
            self.status = "unavailable"
            self.status_reason = "Model is not successfully loaded – predictive positioning disabled"
            self.consecutive_high_count = 0
            self.last_decision_log = self.status_reason
            return False, self.status_reason, []

        if not floor_demand_t5 or len(floor_demand_t5) == 0:
            self.status = "unavailable"
            self.status_reason = "No valid floor-specific demand predictions available"
            self.consecutive_high_count = 0
            self.last_decision_log = self.status_reason
            return False, self.status_reason, []

        for d in floor_demand_t5:
            if math.isnan(d) or math.isinf(d):
                self.status = "unavailable"
                self.status_reason = "Floor demand predictions contain NaN/Inf"
                self.consecutive_high_count = 0
                self.last_decision_log = self.status_reason
                return False, self.status_reason, []

        # ── 2. Elevator Arrival & Stationed Transition Management ────────────
        for eid in list(self.assigned_elevators.keys()):
            target = self.assigned_elevators[eid]
            if env.elevator_positions[eid] == target:
                # Elevator has completed positioning transit at target floor
                env.committed_targets[eid] = None
                if env.assigned_stops[eid] == [target]:
                    env.assigned_stops[eid] = []
                self.stationed_elevators[eid] = target
                # Repositioning cooldown applies only to dispatching away from this target
                self.cooldown_expiry[eid] = now + self.reposition_cooldown_secs
                del self.assigned_elevators[eid]
                self.assigned_reasons.pop(eid, None)

        # Clean up stationed elevators if they moved away (e.g. took a passenger request)
        for eid in list(self.stationed_elevators.keys()):
            st_fl = self.stationed_elevators[eid]
            if env.elevator_positions[eid] != st_fl or len(env.elevator_passengers[eid]) > 0:
                del self.stationed_elevators[eid]

        # ── 3. Congestion Threshold & Confirmation / Hysteresis Tracking ───────
        is_high = congestion_t5 >= self.high_congestion_threshold
        # Identify intra-step events that occur inside an ongoing dispatch step
        is_intra_step = trigger_event in ("elevator_idle_or_arrival", "batch_dispatched")

        if not is_high:
            if congestion_t5 <= self.release_threshold:
                # Congestion dropped below release threshold (<= 45%): Safe deactivation
                self.consecutive_high_count = 0
                self.pending_target_floor = None
                self.pending_target_count = 0
                # Safely clear assignments for idle cars without abrupt reversal
                for eid in list(self.assigned_elevators.keys()):
                    if env.elevator_positions[eid] == env.committed_targets[eid]:
                        env.committed_targets[eid] = None
                        if env.assigned_stops[eid] == [self.assigned_elevators[eid]]:
                            env.assigned_stops[eid] = []
                    del self.assigned_elevators[eid]
                    self.assigned_reasons.pop(eid, None)
                self.stationed_elevators.clear()
                self.active_target_floor = None
                self.active_target_demand = 0.0
                self.status = "inactive"
                self.status_reason = (
                    f"Congestion ({congestion_t5 * 100:.1f}%) <= release threshold "
                    f"({self.release_threshold * 100:.0f}%)"
                )
                self.last_decision_log = self.status_reason
                self._record_diagnostic(
                    trigger_event=trigger_event,
                    congestion_t5=congestion_t5,
                    floor_demand_t5=floor_demand_t5,
                    unmet_demands={},
                    eligible_elevators=[],
                    existing_assignments=dict(self.assigned_elevators),
                    proposed_actions=[],
                    allocation_changed=False,
                    reason=self.status_reason,
                )
                return False, self.status_reason, []
            else:
                # In hysteresis band (release_threshold < congestion < high_threshold)
                if self.status != "active":
                    # Was not active, do not activate without meeting >= 55%
                    self.consecutive_high_count = 0
                    self.status = "inactive"
                    self.status_reason = (
                        f"Congestion ({congestion_t5 * 100:.1f}%) in hysteresis band "
                        f"< high threshold ({self.high_congestion_threshold * 100:.0f}%)"
                    )
                    self.last_decision_log = self.status_reason
                    return False, self.status_reason, []
                else:
                    # Was already active: maintain active mode and continue evaluating allocations!
                    pass
        else:
            # Greater than or equal to 55%
            if not is_intra_step:
                self.consecutive_high_count += 1
            if self.consecutive_high_count < self.confirmation_cycles_required:
                self.status = "awaiting_confirmation"
                self.status_reason = (
                    f"Congestion ({congestion_t5 * 100:.1f}%) >= {self.high_congestion_threshold * 100:.0f}% "
                    f"(Confirmation {self.consecutive_high_count}/{self.confirmation_cycles_required})"
                )
                self.last_decision_log = self.status_reason
                self._record_diagnostic(
                    trigger_event=trigger_event,
                    congestion_t5=congestion_t5,
                    floor_demand_t5=floor_demand_t5,
                    unmet_demands={},
                    eligible_elevators=[],
                    existing_assignments=dict(self.assigned_elevators),
                    proposed_actions=[],
                    allocation_changed=False,
                    reason=self.status_reason,
                )
                return False, self.status_reason, []

        # At this point, high congestion is active & confirmed
        self.status = "active"

        # ── 4. Track Peak Demand Floor with Target-Change Confirmation ────────
        max_demand = max(floor_demand_t5)
        candidate_floor = floor_demand_t5.index(max_demand) + 1

        if self.active_target_floor is None:
            self.active_target_floor = candidate_floor
            self.active_target_demand = float(max_demand)
            self.pending_target_floor = None
            self.pending_target_count = 0
        elif candidate_floor != self.active_target_floor:
            # Peak floor shifted
            if not is_intra_step:
                if self.pending_target_floor == candidate_floor:
                    self.pending_target_count += 1
                else:
                    self.pending_target_floor = candidate_floor
                    self.pending_target_count = 1

            if self.pending_target_count >= self.target_change_confirmation_cycles:
                # Target change confirmed for 2 cycles
                current_target_demand = 0.0
                if 1 <= self.active_target_floor <= len(floor_demand_t5):
                    current_target_demand = floor_demand_t5[self.active_target_floor - 1]

                if max_demand >= current_target_demand:
                    self.active_target_floor = candidate_floor
                    self.active_target_demand = float(max_demand)
                    self.pending_target_floor = None
                    self.pending_target_count = 0
        else:
            self.active_target_demand = float(max_demand)
            self.pending_target_floor = None
            self.pending_target_count = 0

        # ── 5. Evaluate Elevator Availability & Eligibility ───────────────────
        # An elevator is eligible for consideration if:
        # 1. Not locked to an active emergency
        # 2. No passengers on board
        # 3. No real passenger requests assigned
        # 4. Not committed to a real passenger route
        eligible_elevators: List[int] = []
        for e in range(num_elevators):
            if e in locked_elevator_ids:
                continue
            if len(env.elevator_passengers[e]) > 0:
                continue
            if any(r.get("assigned_to") == e for r in env.pending_requests):
                continue
            # If it has a committed target that is NOT its predictive assignment, it's busy
            if env.committed_targets[e] is not None and e not in self.assigned_elevators:
                continue
            eligible_elevators.append(e)

        self.last_eligible_count = len(eligible_elevators)

        # ── 6. Floor-by-Floor Unmet Demand & Coverage Calculation ─────────────
        floor_evals: List[dict] = []
        unmet_demands: Dict[int, float] = {}
        floors_count = min(num_floors, len(floor_demand_t5))

        for f in range(1, floors_count + 1):
            dem = max(0.0, float(floor_demand_t5[f - 1]))

            # Lifts stationed at floor f (idle and at floor f)
            stationed = [
                e for e in eligible_elevators
                if env.elevator_positions[e] == f and e not in self.assigned_elevators
            ]
            # Lifts en route to floor f under active predictive assignment
            en_route = [
                e for e, tgt in self.assigned_elevators.items()
                if tgt == f and e in eligible_elevators
            ]

            covered_capacity = len(stationed + en_route) * self.service_capacity_per_elevator
            unmet = max(0.0, dem - covered_capacity)
            unmet_demands[f] = unmet

            floor_evals.append({
                "floor": f,
                "demand": round(dem, 2),
                "stationed": [e + 1 for e in stationed],
                "en_route": [e + 1 for e in en_route],
                "covered_capacity": round(covered_capacity, 2),
                "unmet_demand": round(unmet, 2),
            })

        self.last_floor_evaluations = floor_evals

        if not eligible_elevators:
            self.status_reason = "Active: All elevators busy serving passenger requests or emergency"
            self.last_decision_log = self.status_reason
            self._record_diagnostic(
                trigger_event=trigger_event,
                congestion_t5=congestion_t5,
                floor_demand_t5=floor_demand_t5,
                unmet_demands=unmet_demands,
                eligible_elevators=[],
                existing_assignments=dict(self.assigned_elevators),
                proposed_actions=[],
                allocation_changed=False,
                reason=self.status_reason,
            )
            return True, self.status_reason, []

        # ── 7. Preserve Existing Valid In-Flight Assignments ───────────────────
        # Check if existing en-route predictive assignments remain beneficial.
        # Preserve them so elevators are not redirected unnecessarily.
        preserved_elevators = set()
        for e, tgt in list(self.assigned_elevators.items()):
            if e in eligible_elevators:
                # If target still has demand or is the active target, preserve
                tgt_dem = floor_demand_t5[tgt - 1] if 1 <= tgt <= len(floor_demand_t5) else 0.0
                if tgt_dem >= self.min_demand_threshold or tgt == self.active_target_floor:
                    preserved_elevators.add(e)
                else:
                    # Target floor demand vanished: safely release assignment
                    env.committed_targets[e] = None
                    if env.assigned_stops[e] == [tgt]:
                        env.assigned_stops[e] = []
                    del self.assigned_elevators[e]
                    self.assigned_reasons.pop(e, None)

        # ── 8. Candidate Selection for New Dispatches ─────────────────────────
        # An eligible elevator is available for a new predictive dispatch if:
        # - It is not already en-route to a preserved predictive target
        # - It is not in repositioning cooldown (unless target change is confirmed and advantageous)
        # - If stationed at floor P, it is NOT needed to cover floor P's own demand
        candidates: List[int] = []
        for e in eligible_elevators:
            if e in preserved_elevators:
                continue

            pos = env.elevator_positions[e]
            pos_demand = floor_demand_t5[pos - 1] if 1 <= pos <= len(floor_demand_t5) else 0.0

            # Check if this elevator is needed at its current floor
            at_pos = [
                x for x in eligible_elevators
                if env.elevator_positions[x] == pos and x not in self.assigned_elevators
            ]
            needed_at_pos = math.ceil(pos_demand / self.service_capacity_per_elevator) if pos_demand >= self.min_demand_threshold else 0

            if len(at_pos) > needed_at_pos or pos_demand < self.min_demand_threshold:
                # Elevator is not strictly needed where it is stationed
                # Cooldown check: if cooldown is active, only move if demand elsewhere is materially greater
                if now < self.cooldown_expiry.get(e, 0.0):
                    # Check if there is high unmet demand elsewhere that justifies breaking cooldown
                    max_unmet = max(unmet_demands.values()) if unmet_demands else 0.0
                    if max_unmet < self.service_capacity_per_elevator * 1.5:
                        continue  # Respect cooldown for moderate demand
                candidates.append(e)

        # ── 9. Operational Availability Constraint ────────────────────────────
        # "Keep at least one suitable elevator available for actual passenger requests where operationally possible."
        # If all elevators in the system are currently idle/eligible, limit total predictive assignments
        # to at most (num_elevators - 1), so at least 1 remains completely free.
        max_total_predictive = max(1, num_elevators - 1)
        current_assigned_count = len(self.assigned_elevators)
        allowed_new_dispatches = max(0, max_total_predictive - current_assigned_count)

        if not candidates or allowed_new_dispatches <= 0:
            if any(self.assigned_elevators):
                en_route_str = ", ".join(f"Lift {e + 1} -> F{tgt}" for e, tgt in sorted(self.assigned_elevators.items()))
                self.status_reason = f"Active: Predictive repositioning in progress ({en_route_str})"
            else:
                self.status_reason = "Active: All predicted demand adequately covered by stationed elevators"
            self.last_decision_log = self.status_reason
            self._record_diagnostic(
                trigger_event=trigger_event,
                congestion_t5=congestion_t5,
                floor_demand_t5=floor_demand_t5,
                unmet_demands=unmet_demands,
                eligible_elevators=eligible_elevators,
                existing_assignments=dict(self.assigned_elevators),
                proposed_actions=[],
                allocation_changed=False,
                reason=self.status_reason,
            )
            return True, self.status_reason, []

        # ── 10. Demand-Aware Allocation Loop ──────────────────────────────────
        dispatched_actions: List[dict] = []
        available_candidates = list(candidates)

        while available_candidates and len(dispatched_actions) < allowed_new_dispatches:
            # Find floor with the highest unmet demand
            best_floor = None
            highest_unmet = -1.0
            for f in range(1, floors_count + 1):
                u = unmet_demands[f]
                if u > highest_unmet:
                    highest_unmet = u
                    best_floor = f

            # Stop if highest unmet demand is below threshold (no movement needed)
            if best_floor is None or highest_unmet < self.min_demand_threshold:
                break

            # Pick closest candidate elevator
            best_e = min(
                available_candidates,
                key=lambda e: (abs(env.elevator_positions[e] - best_floor), e),
            )
            available_candidates.remove(best_e)

            # If already at best_floor, it satisfies demand without movement
            if env.elevator_positions[best_e] == best_floor:
                unmet_demands[best_floor] = max(
                    0.0, unmet_demands[best_floor] - self.service_capacity_per_elevator
                )
                self.stationed_elevators[best_e] = best_floor
                continue

            # Dispatch best_e to best_floor
            self.assigned_elevators[best_e] = best_floor
            env.assigned_stops[best_e] = [best_floor]
            env.committed_targets[best_e] = best_floor
            reason = (
                f"Predictive positioning: F{best_floor} highest unmet demand "
                f"({highest_unmet:.2f} reqs, forecast={floor_demand_t5[best_floor - 1]:.2f}) [Trigger: {trigger_event}]"
            )
            self.assigned_reasons[best_e] = reason

            action_dict = {
                "elevator_id": best_e,
                "target_floor": best_floor,
                "reason": reason,
                "pred_demand": floor_demand_t5[best_floor - 1],
                "unmet_demand": highest_unmet,
                "trigger_event": trigger_event,
            }
            env.last_action = action_dict
            dispatched_actions.append(action_dict)

            # Update unmet demand for best_floor
            unmet_demands[best_floor] = max(
                0.0, unmet_demands[best_floor] - self.service_capacity_per_elevator
            )

        # ── 11. Final Status & Diagnostic Logging ────────────────────────────
        if dispatched_actions:
            details = "; ".join(
                f"Lift {a['elevator_id'] + 1} -> F{a['target_floor']} (demand={a['pred_demand']:.1f})"
                for a in dispatched_actions
            )
            self.status_reason = f"Active: Dispatched {details}"
        elif any(self.assigned_elevators):
            en_route_str = ", ".join(
                f"Lift {e + 1} -> F{tgt}"
                for e, tgt in sorted(self.assigned_elevators.items())
            )
            self.status_reason = f"Active: In transit ({en_route_str})"
        else:
            self.status_reason = "Active: All predicted demand adequately covered by stationed elevators"

        self.last_decision_log = self.status_reason
        self._record_diagnostic(
            trigger_event=trigger_event,
            congestion_t5=congestion_t5,
            floor_demand_t5=floor_demand_t5,
            unmet_demands=unmet_demands,
            eligible_elevators=eligible_elevators,
            existing_assignments=dict(self.assigned_elevators),
            proposed_actions=dispatched_actions,
            allocation_changed=bool(dispatched_actions),
            reason=self.status_reason,
        )

        return True, self.status_reason, dispatched_actions

    def _record_diagnostic(
        self,
        trigger_event: str,
        congestion_t5: float,
        floor_demand_t5: Optional[List[float]],
        unmet_demands: Dict[int, float],
        eligible_elevators: List[int],
        existing_assignments: Dict[int, int],
        proposed_actions: List[dict],
        allocation_changed: bool,
        reason: str,
    ):
        """Appends a structured diagnostic record to the bounded ring buffer."""
        record = {
            "timestamp": time.time(),
            "trigger_event": trigger_event,
            "congestion_t5": round(float(congestion_t5), 3),
            "floor_demands": [round(float(d), 2) for d in (floor_demand_t5 or [])],
            "unmet_demands": {f: round(float(u), 2) for f, u in unmet_demands.items()},
            "eligible_elevators": [e + 1 for e in eligible_elevators],
            "existing_assignments": {e + 1: tgt for e, tgt in existing_assignments.items()},
            "proposed_actions": [
                {"lift": a["elevator_id"] + 1, "target": a["target_floor"], "reason": a["reason"]}
                for a in proposed_actions
            ],
            "allocation_changed": allocation_changed,
            "decision_log": reason,
        }
        self.evaluation_diagnostics.append(record)

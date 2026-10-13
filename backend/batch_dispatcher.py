"""
RequestBatchDispatcher
======================
Implements a 2-second request batching window for the AI Elevator System.

Design goals
------------
* A batching window of exactly BATCH_WINDOW_SECONDS (2 s) is started by the
  FIRST unassigned request that arrives.  Subsequent requests arriving during
  that window are queued; they do NOT restart or extend the window.
* When the window expires, all collected requests are processed together and
  dispatched to the smallest number of lifts that can serve them safely.
* Requests that cannot be batched (e.g. incompatible direction, capacity full)
  are assigned their own lift.
* The dispatcher is *stateless between windows* – it never stores passenger
  identities or merges destinations; individual src/dst pairs are always kept
  separate.
* Thread safety: a threading.Lock guards every mutation so the Flask dev-server
  (which uses threads) cannot race on the window state.

Public API
----------
    d = RequestBatchDispatcher(env, q_agent, logger=None)
    d.enqueue(req)          -> None          (call from /api/request)
    d.tick()                -> list[dict]    (call from /api/step; returns dispatched actions)
    d.get_status()          -> dict          (batch_active, countdown_ms, queued_count)
    d.flush()               -> list[dict]    (force-dispatch; used in tests)
"""

import time
import threading
import logging

BATCH_WINDOW_SECONDS: float = 2.0   # Width of batching window (never restarted)


class RequestBatchDispatcher:
    """
    Wraps an ElevatorEnvironment + QLearningAgent with a 2-second batching
    window.  All public methods are thread-safe.
    """

    def __init__(self, env, q_agent, logger=None):
        self._env      = env
        self._q_agent  = q_agent
        self._log      = logger or logging.getLogger(__name__)
        self._lock     = threading.Lock()
        self._emergency_dispatcher = None

        self._batch: list        = []           # pending, not-yet-dispatched requests
        self._window_start       = None         # wall-clock time window opened
        self._dispatched_ids     = set()        # req ids already assigned (dedup guard)

    def set_emergency_dispatcher(self, emergency_dispatcher):
        """Sets the EmergencyDispatcher reference to respect locked elevators."""
        self._emergency_dispatcher = emergency_dispatcher

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def enqueue(self, req: dict) -> None:
        """
        Accept a newly-created pending request into the batch queue.
        Opens a new window if none is active.
        """
        with self._lock:
            if req["id"] in self._dispatched_ids:
                return

            self._batch.append(req)
            self._log.info(
                "[Batch] Queued req %s F%s->F%s  (batch size=%d)",
                req["id"], req["src"], req["dst"], len(self._batch),
            )

            if self._window_start is None:
                self._window_start = time.monotonic()
                self._log.info(
                    "[Batch] Window OPENED – will dispatch in %.1fs",
                    BATCH_WINDOW_SECONDS,
                )

    def tick(self) -> list:
        """
        Called on every simulation step.  Checks whether the batching window
        has expired.  Returns a list of dispatch-action dicts (may be empty).
        """
        with self._lock:
            if not self._batch or self._window_start is None:
                return []
            elapsed = time.monotonic() - self._window_start
            if elapsed < BATCH_WINDOW_SECONDS:
                return []
            return self._flush_locked()

    def flush(self) -> list:
        """Force-dispatch all queued requests immediately (for tests / reset)."""
        with self._lock:
            if not self._batch:
                return []
            return self._flush_locked()

    def get_status(self) -> dict:
        """Return information the frontend can display."""
        with self._lock:
            if self._window_start is None or not self._batch:
                return {"batch_active": False, "countdown_ms": 0, "queued_count": 0}
            elapsed   = time.monotonic() - self._window_start
            remaining = max(0.0, BATCH_WINDOW_SECONDS - elapsed)
            return {
                "batch_active":  True,
                "countdown_ms":  int(remaining * 1000),
                "queued_count":  len(self._batch),
            }

    def reset(self) -> None:
        """Clear all batch state (called on /api/reset)."""
        with self._lock:
            self._batch.clear()
            self._window_start = None
            self._dispatched_ids.clear()

    def get_batched_ids(self) -> set:
        """
        Returns the set of request IDs currently held in the open batch window.
        Used by the caller to skip these requests in the immediate fallback
        dispatch loop (they will be dispatched when the window expires).
        """
        with self._lock:
            return {r["id"] for r in self._batch}

    # -------------------------------------------------------------------------
    # Internal helpers (must be called while _lock is held)
    # -------------------------------------------------------------------------

    def _flush_locked(self) -> list:
        """
        Core batching logic: groups requests, assigns lifts, returns actions.
        Must be called with self._lock already held.
        """
        requests = [r for r in self._batch if r["id"] not in self._dispatched_ids]
        self._batch.clear()
        self._window_start = None

        if not requests:
            return []

        self._log.info("[Batch] Window CLOSED – dispatching %d request(s)", len(requests))

        groups  = self._group_requests(requests)
        actions = []
        for group in groups:
            actions.extend(self._dispatch_group(group))
        return actions

    # -- grouping -------------------------------------------------------------

    def _group_requests(self, requests: list) -> list:
        """
        Partition requests into groups that a single lift *could* serve in one
        directional run.

        Two requests are compatible when:
          1. They share the same direction (both up or both down).
          2. All pickups AND destinations fall in the same monotone interval.
          3. The combined load never exceeds lift capacity at any segment.

        Greedy algorithm: sort by src, extend the current group if compatible,
        otherwise start a new group.
        """
        up_reqs   = [r for r in requests if r["dst"] > r["src"]]
        down_reqs = [r for r in requests if r["dst"] < r["src"]]
        groups    = []

        def _greedy(reqs, going_up):
            if going_up:
                sorted_r = sorted(reqs, key=lambda r: (r["src"], r["dst"]))
            else:
                sorted_r = sorted(reqs, key=lambda r: (-r["src"], -r["dst"]))
            current = []
            for req in sorted_r:
                if not current or self._compatible(req, current, going_up):
                    current.append(req)
                else:
                    groups.append(current)
                    current = [req]
            if current:
                groups.append(current)

        _greedy(up_reqs,   going_up=True)
        _greedy(down_reqs, going_up=False)

        self._log.info(
            "[Batch] Grouped %d request(s) -> %d group(s)",
            len(requests), len(groups),
        )
        return groups

    def _compatible(self, req: dict, group: list, going_up: bool) -> bool:
        """
        Returns True if req can join the group for a single-lift run.
        Checks monotone ordering and combined peak-load vs. capacity.
        """
        env = self._env
        if going_up:
            min_src = min(r["src"] for r in group)
            # Pickup of new request must not be before the group's start
            if req["src"] < min_src:
                return False
        else:
            max_src = max(r["src"] for r in group)
            if req["src"] > max_src:
                return False

        # Capacity check: peak load across the merged group
        if self._estimate_peak_load(group + [req], going_up) > env.capacity:
            return False
        return True

    def _estimate_peak_load(self, reqs: list, going_up: bool) -> int:
        """
        Floor-by-floor sweep to find the maximum simultaneous occupancy.
        """
        if not reqs:
            return 0
        if going_up:
            lo   = min(r["src"] for r in reqs)
            hi   = max(r["dst"] for r in reqs)
            peak = 0
            for f in range(lo, hi):
                onboard = sum(1 for r in reqs if r["src"] <= f < r["dst"])
                peak = max(peak, onboard)
        else:
            hi   = max(r["src"] for r in reqs)
            lo   = min(r["dst"] for r in reqs)
            peak = 0
            for f in range(lo, hi):
                onboard = sum(1 for r in reqs if r["dst"] <= f < r["src"])
                peak = max(peak, onboard)
        return peak

    # -- lift selection -------------------------------------------------------

    def _dispatch_group(self, group: list) -> list:
        """
        Find the best single lift for the group.  Falls back to per-request
        Q-agent dispatch when no single lift can accommodate the full group.
        Returns list of {"elevator_id": int, "req": dict} records.
        """
        env     = self._env
        q_agent = self._q_agent

        best_lift = self._find_best_lift_for_group(group)
        actions   = []

        if best_lift is not None:
            for req in group:
                if req["id"] in self._dispatched_ids:
                    continue
                env.assign_request(req, best_lift)
                self._dispatched_ids.add(req["id"])
                self._log.info(
                    "[Batch] Assigned req %s F%s->F%s -> Lift %d",
                    req["id"], req["src"], req["dst"], best_lift,
                )
                actions.append({"elevator_id": best_lift, "req": req})
            return actions

        # No single lift can serve all – dispatch each individually
        locked_lifts = set()
        if self._emergency_dispatcher:
            locked_lifts = self._emergency_dispatcher.get_locked_elevator_ids()

        for req in group:
            if req["id"] in self._dispatched_ids:
                continue
            raw_actions, raw_heuristics = env.get_dispatch_candidates(req)
            valid_actions = []
            heuristics = []
            for act, h in zip(raw_actions, raw_heuristics):
                if act[0] not in locked_lifts:
                    valid_actions.append(act)
                    heuristics.append(h)

            if not valid_actions:
                self._log.warning(
                    "[Batch] No valid lift for req %s F%s->F%s – left pending",
                    req["id"], req["src"], req["dst"],
                )
                continue
            current_state = env.get_state()
            act = q_agent.get_dispatch_action(current_state, valid_actions, heuristics)
            if act and act[0] not in locked_lifts:
                env.assign_request(req, act[0])
                self._dispatched_ids.add(req["id"])
                self._log.info(
                    "[Batch] Individually assigned req %s F%s->F%s -> Lift %d",
                    req["id"], req["src"], req["dst"], act[0],
                )
                actions.append({"elevator_id": act[0], "req": req})
        return actions

    def _find_best_lift_for_group(self, group: list):
        """
        Returns the elevator_id whose total insertion cost across all group
        requests is lowest (and can_accommodate all of them), else None.
        """
        env       = self._env
        best_lift = None
        best_cost = float("inf")

        locked_lifts = set()
        if self._emergency_dispatcher:
            locked_lifts = self._emergency_dispatcher.get_locked_elevator_ids()

        for e in range(env.num_elevators):
            if e in locked_lifts:
                continue
            total_cost = 0.0
            feasible   = True
            for req in group:
                if not env.can_accommodate(e, req):
                    feasible = False
                    break
                total_cost += env.calculate_insertion_cost(e, req)
            if feasible and total_cost < best_cost:
                best_cost = total_cost
                best_lift = e

        return best_lift

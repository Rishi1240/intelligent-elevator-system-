/**
 * Node.js Native Backend Server for AI Elevator Optimization System.
 * Supports configurable floors (2–20) and elevators (1–5).
 */

const http = require('http');
const url  = require('url');

// ── hard caps & defaults ─────────────────────────────────────────────────────
const MAX_FLOORS    = 20;
const MAX_ELEVATORS = 5;
const MIN_FLOORS    = 2;
const MIN_ELEVATORS = 1;

const DEFAULT_FLOORS     = 10;
const DEFAULT_ELEVATORS  = 3;
const DEFAULT_CAPACITY   = 8;

const PORT = process.env.PORT || 5001;

// ── 1. ELEVATOR ENVIRONMENT ───────────────────────────────────────────────────
class ElevatorEnvironment {
  constructor(numFloors = DEFAULT_FLOORS, numElevators = DEFAULT_ELEVATORS, capacity = DEFAULT_CAPACITY) {
    this.numFloors    = Math.max(MIN_FLOORS,    Math.min(MAX_FLOORS,    parseInt(numFloors, 10)));
    this.numElevators = Math.max(MIN_ELEVATORS, Math.min(MAX_ELEVATORS, parseInt(numElevators, 10)));
    this.capacity     = Math.max(1, parseInt(capacity, 10));
    this._buildActionSpace();
    this.reset();
  }

  _buildActionSpace() {
    this.actionSpace = [];
    for (let e = 0; e < this.numElevators; e++) {
      for (let f = 1; f <= this.numFloors; f++) {
        this.actionSpace.push({ elevator_id: e, target_floor: f });
      }
    }
  }

  /** Hot-reconfigure building dimensions and reset. */
  reconfigure({ numFloors, numElevators, capacity } = {}) {
    if (numFloors    != null) this.numFloors    = Math.max(MIN_FLOORS,    Math.min(MAX_FLOORS,    parseInt(numFloors, 10)));
    if (numElevators != null) this.numElevators = Math.max(MIN_ELEVATORS, Math.min(MAX_ELEVATORS, parseInt(numElevators, 10)));
    if (capacity     != null) this.capacity     = Math.max(1, parseInt(capacity, 10));
    this._buildActionSpace();
    return this.reset();
  }

  reset() {
    this.elevatorPositions  = new Array(this.numElevators).fill(1);
    this.elevatorPassengers = Array.from({ length: this.numElevators }, () => []);
    // committedTarget[e] = floor the elevator MUST reach before taking a new Q-action.
    // null means the elevator is free to be assigned by the Q-agent.
    this.committedTarget    = new Array(this.numElevators).fill(null);
    this.assignedStops      = Array.from({ length: this.numElevators }, () => []);
    this.pendingRequests    = [];
    this.congestionPrediction = 0.20;
    this.stepCount  = 0;
    this.lastAction = null;
    return this.getState();
  }

  addRequest(srcFloor, dstFloor) {
    const src = Math.max(1, Math.min(this.numFloors, parseInt(srcFloor, 10)));
    const dst = Math.max(1, Math.min(this.numFloors, parseInt(dstFloor, 10)));
    if (src === dst || isNaN(src) || isNaN(dst)) return false;
    this.pendingRequests.push({
      id:          this.pendingRequests.length + (Date.now() % 100000),
      src,
      dst,
      assigned_to: null,
      wait_time:   0.0,
      created_at:  Date.now()
    });
    return true;
  }

  getUnassignedRequests() {
    return this.pendingRequests.filter(r => r.assigned_to === null || r.assigned_to === undefined);
  }

  canAccommodate(e, req) {
    const src = req.src;
    const dst = req.dst;
    const low = Math.min(src, dst);
    const high = Math.max(src, dst);

    for (let f = low; f < high; f++) {
      let occ = 0;
      for (const pDst of this.elevatorPassengers[e]) {
        if (dst > src && pDst > f) occ++;
        else if (dst < src && pDst <= f) occ++;
      }
      for (const r of this.pendingRequests) {
        if (r.assigned_to === e && r !== req) {
          if (r.dst > r.src && r.src <= f && f < r.dst) occ++;
          else if (r.dst < r.src && r.dst <= f && f < r.src) occ++;
        }
      }
      if (occ + 1 > this.capacity) return false;
    }
    return true;
  }

  calculateInsertionCost(e, req) {
    if (!this.canAccommodate(e, req)) return 1000.0;

    const pos = this.elevatorPositions[e];
    const src = req.src;
    const dst = req.dst;
    const reqUp = dst > src;

    const currOcc = this.elevatorPassengers[e].length;
    const assignedReqs = this.pendingRequests.filter(r => r.assigned_to === e);

    // 1. Idle elevator
    if (currOcc === 0 && assignedReqs.length === 0 && (!this.assignedStops[e] || this.assignedStops[e].length === 0)) {
      const distToPickup = Math.abs(pos - src);
      const rideDist = Math.abs(dst - src);
      return distToPickup * 1.5 + rideDist * 1.0 + 15.0;
    }

    // 2. Active elevator: check directional compatibility
    let tFloor = this.committedTarget[e];
    if (tFloor === null && this.assignedStops[e] && this.assignedStops[e].length > 0) {
      tFloor = this.assignedStops[e][0];
    }

    const elevUp = tFloor !== null && tFloor > pos;
    const elevDown = tFloor !== null && tFloor < pos;

    const canBatch = (
      (elevUp && reqUp && pos <= src) ||
      (elevDown && !reqUp && pos >= src)
    );

    if (canBatch) {
      const currentStops = this.assignedStops[e] && this.assignedStops[e].length > 0
        ? [...this.assignedStops[e]]
        : [...this.elevatorPassengers[e]];
      if (currentStops.length === 0) currentStops.push(tFloor || pos);

      let incrementalDist = 0;
      if (elevUp) {
        const furthest = Math.max(...currentStops);
        incrementalDist = Math.max(0, dst - furthest);
      } else {
        const furthest = Math.min(...currentStops);
        incrementalDist = Math.max(0, furthest - dst);
      }

      const waitTime = Math.abs(src - pos);
      const stopDelay = 2.0;
      return incrementalDist * 1.0 + waitTime * 1.0 + stopDelay * 1.5;
    }

    return 80.0 + Math.abs(src - pos) * 2.0 + Math.abs(dst - src);
  }

  computeElevatorSuitability(e, req) {
    const cost = this.calculateInsertionCost(e, req);
    return Math.max(1.0, 160.0 - cost);
  }

  _recomputeRoute(e) {
    const pos = this.elevatorPositions[e];
    const onboard = [...this.elevatorPassengers[e]];
    const assignedReqs = this.pendingRequests.filter(r => r.assigned_to === e);

    if (onboard.length === 0 && assignedReqs.length === 0) {
      this.assignedStops[e] = [];
      this.committedTarget[e] = null;
      return;
    }

    let heading = 0;
    if (this.assignedStops[e] && this.assignedStops[e].length > 0) {
      const firstStop = this.assignedStops[e][0];
      if (firstStop > pos) heading = 1;
      else if (firstStop < pos) heading = -1;
    }

    if (heading === 0) {
      if (onboard.length > 0) {
        if (onboard.some(d => d > pos)) heading = 1;
        else if (onboard.some(d => d < pos)) heading = -1;
      } else if (assignedReqs.length > 0) {
        const firstR = assignedReqs[0];
        if (firstR.src > pos) heading = 1;
        else if (firstR.src < pos) heading = -1;
        else heading = firstR.dst > pos ? 1 : -1;
      }
    }

    let combined = [];
    if (heading >= 0) {
      const upStops = new Set();
      for (const d of onboard) { if (d >= pos) upStops.add(d); }
      for (const r of assignedReqs) {
        if (r.dst > r.src && r.src >= pos) {
          upStops.add(r.src);
          upStops.add(r.dst);
        }
      }
      const leg1 = Array.from(upStops).sort((a, b) => a - b);

      const downStops = new Set();
      for (const d of onboard) { if (d < pos) downStops.add(d); }
      for (const r of assignedReqs) {
        if (r.dst < r.src || r.src < pos) {
          downStops.add(r.src);
          downStops.add(r.dst);
        }
      }
      const leg2 = Array.from(downStops).sort((a, b) => b - a);
      combined = [...leg1, ...leg2.filter(s => !leg1.includes(s))];
    } else {
      const downStops = new Set();
      for (const d of onboard) { if (d <= pos) downStops.add(d); }
      for (const r of assignedReqs) {
        if (r.dst < r.src && r.src <= pos) {
          downStops.add(r.src);
          downStops.add(r.dst);
        }
      }
      const leg1 = Array.from(downStops).sort((a, b) => b - a);

      const upStops = new Set();
      for (const d of onboard) { if (d > pos) upStops.add(d); }
      for (const r of assignedReqs) {
        if (r.dst > r.src || r.src > pos) {
          upStops.add(r.src);
          upStops.add(r.dst);
        }
      }
      const leg2 = Array.from(upStops).sort((a, b) => a - b);
      combined = [...leg1, ...leg2.filter(s => !leg1.includes(s))];
    }

    const finalStops = combined.filter(s => s !== pos);
    this.assignedStops[e] = finalStops;
    this.committedTarget[e] = finalStops.length > 0 ? finalStops[0] : null;
  }

  getDispatchCandidates(req) {
    const src = req.src;
    const dst = req.dst;

    const costs = {};
    for (let e = 0; e < this.numElevators; e++) {
      if (!this.canAccommodate(e, req)) continue;
      costs[e] = this.calculateInsertionCost(e, req);
    }

    let eligibleElevators = [];
    const costKeys = Object.keys(costs);
    if (costKeys.length === 0) {
      eligibleElevators = [0];
    } else {
      const minCost = Math.min(...Object.values(costs));
      eligibleElevators = costKeys
        .map(Number)
        .filter(e => costs[e] <= minCost + 3.0);
    }

    const validActions = [];
    const heuristicScores = [];
    for (const e of eligibleElevators) {
      let targetFloor;
      if (this.assignedStops[e] && this.assignedStops[e].length > 0) {
        targetFloor = this.assignedStops[e][0];
      } else if (this.elevatorPositions[e] === src) {
        targetFloor = dst;
      } else {
        targetFloor = src;
      }
      validActions.push({ elevator_id: e, target_floor: targetFloor });
      heuristicScores.push(this.computeElevatorSuitability(e, req));
    }
    return { validActions, heuristicScores };
  }

  assignRequest(req, elevatorId) {
    req.assigned_to = elevatorId;
    const currPos = this.elevatorPositions[elevatorId];
    const src = req.src;
    const dst = req.dst;

    if (currPos === src && this.elevatorPassengers[elevatorId].length < this.capacity) {
      this.elevatorPassengers[elevatorId].push(dst);
      this.pendingRequests = this.pendingRequests.filter(r => r !== req);
    }
    this._recomputeRoute(elevatorId);
  }

  getWaitTimes() {
    if (this.pendingRequests.length === 0) return 0.0;
    const total = this.pendingRequests.reduce((s, r) => s + r.wait_time, 0);
    return parseFloat((total / this.pendingRequests.length).toFixed(2));
  }

  getState() {
    const floorDemand = new Array(this.numFloors).fill(0);
    for (const req of this.pendingRequests) {
      const idx = req.src - 1;
      if (idx >= 0 && idx < this.numFloors) floorDemand[idx]++;
    }
    return {
      elevator_positions:    [...this.elevatorPositions],
      elevator_occupancy:    this.elevatorPassengers.map((p) => p.length),
      elevator_targets:      [...this.committedTarget],   // null = idle/free
      assigned_stops:        this.assignedStops ? this.assignedStops.map(s => [...s]) : [],
      pending_requests:      this.pendingRequests.length,
      floor_demand:          floorDemand,
      congestion_prediction: parseFloat(this.congestionPrediction.toFixed(4)),
      avg_wait_time:         this.getWaitTimes(),
      step_count:            this.stepCount,
      lastAction:            this.lastAction,
      last_action:           this.lastAction,
      // ── building config ──
      num_floors:    this.numFloors,
      num_elevators: this.numElevators,
      capacity:      this.capacity,
    };
  }

  _pickupAtFloor(e, floor) {
    const tFloor = this.committedTarget[e];
    const goingUp = tFloor !== null && tFloor > floor;
    const goingDn = tFloor !== null && tFloor < floor;
    let isEmpty = this.elevatorPassengers[e].length === 0;

    const remaining = [];
    let picked = 0;
    for (const req of this.pendingRequests) {
      const assignedTo = req.assigned_to;
      const isAssignedToE = (assignedTo === e);
      const isUnassigned = (assignedTo === null || assignedTo === undefined);

      if (!isAssignedToE && !isUnassigned) {
        remaining.push(req);
        continue;
      }

      const reqGoingUp = req.dst > req.src;
      const reqGoingDn = req.dst < req.src;
      const dirMatch = isAssignedToE || isEmpty
        || (goingUp && reqGoingUp)
        || (goingDn && reqGoingDn)
        || (!goingUp && !goingDn);

      if (
        req.src === floor &&
        dirMatch &&
        this.elevatorPassengers[e].length < this.capacity
      ) {
        this.elevatorPassengers[e].push(req.dst);
        picked++;
        if (isEmpty) {
          isEmpty = false;
        }
      } else {
        remaining.push(req);
      }
    }
    this.pendingRequests = remaining;
    return picked;
  }

  step(action = null) {
    this.stepCount++;
    let elevatorId = null;
    if (action && action.elevator_id !== undefined) {
      elevatorId = Math.max(0, Math.min(this.numElevators - 1, parseInt(action.elevator_id, 10)));
      const targetFloor = Math.max(1, Math.min(this.numFloors, parseInt(action.target_floor, 10)));
      this.lastAction = { elevator_id: elevatorId, target_floor: targetFloor };
    }

    let totalPickedUp = 0;
    let totalDroppedOff = 0;
    let totalMovementDist = 0;

    // Advance only elevators that have active trips or committed targets
    for (let e = 0; e < this.numElevators; e++) {
      let currFloor = this.elevatorPositions[e];

      // 1. Pick up assigned passengers at current floor if departing
      const picked = this._pickupAtFloor(e, currFloor);
      totalPickedUp += picked;
      if (picked > 0) {
        this._recomputeRoute(e);
      }

      const tFloor = this.committedTarget[e];
      if (tFloor === null) {
        // Idle elevator: strictly stationary!
        continue;
      }

      // 2. Move one floor towards target
      let stepMovement = 0;
      if (currFloor < tFloor) {
        this.elevatorPositions[e]++;
        stepMovement = 1;
      } else if (currFloor > tFloor) {
        this.elevatorPositions[e]--;
        stepMovement = 1;
      }
      totalMovementDist += stepMovement;
      currFloor = this.elevatorPositions[e];

      // 3. Drop off passengers whose destination is currFloor
      const before = this.elevatorPassengers[e].length;
      this.elevatorPassengers[e] = this.elevatorPassengers[e].filter((d) => d !== currFloor);
      const dropped = before - this.elevatorPassengers[e].length;
      totalDroppedOff += dropped;

      // 4. Pick up passengers waiting at arrival floor
      totalPickedUp += this._pickupAtFloor(e, currFloor);

      // 5. Remove currFloor from assignedStops if reached
      if (this.assignedStops[e]) {
        this.assignedStops[e] = this.assignedStops[e].filter(s => s !== currFloor);
      }

      // 6. Recompute route to keep assignedStops and committedTarget synchronized
      this._recomputeRoute(e);
    }

    // Age remaining pending requests
    for (const req of this.pendingRequests) req.wait_time += 1.0;

    // Reward calculation
    const alpha = 1.0, beta = 0.5, gamma = 0.3;
    const avgWait  = this.getWaitTimes();
    const pending  = this.pendingRequests.length;
    const currOcc  = elevatorId !== null ? this.elevatorPassengers[elevatorId].length : 0;
    const unused   = Math.max(0, this.capacity - currOcc);
    let reward = -(alpha * (avgWait * 0.5 + pending) + beta * totalMovementDist + gamma * (unused / this.capacity));
    reward += totalPickedUp * 2.0 + totalDroppedOff * 3.0;

    return {
      newState: this.getState(),
      reward: parseFloat(reward.toFixed(3)),
      info: { picked_up: totalPickedUp, dropped_off: totalDroppedOff, movement: totalMovementDist }
    };
  }
}

// ── 2. TABULAR Q-LEARNING AGENT ───────────────────────────────────────────────
class QLearningAgent {
  constructor(numElevators = DEFAULT_ELEVATORS, numFloors = DEFAULT_FLOORS) {
    this.numElevators = numElevators;
    this.numFloors    = numFloors;
    this.alpha        = 0.1;
    this.gamma        = 0.9;
    this.epsilon      = 0.9;
    this.epsilonDecay = 0.995;
    this.epsilonMin   = 0.01;
    this.qTable       = new Map();
    this._buildActionSpace();
  }

  _buildActionSpace() {
    this.actionSpace = [];
    for (let e = 0; e < this.numElevators; e++) {
      for (let f = 1; f <= this.numFloors; f++) {
        this.actionSpace.push({ elevator_id: e, target_floor: f });
      }
    }
  }

  reinitialize(numElevators, numFloors) {
    this.numElevators = numElevators;
    this.numFloors    = numFloors;
    this.epsilon      = 0.9;
    this.qTable       = new Map();
    this._buildActionSpace();
  }

  encodeState(s) {
    const pos = (s.elevator_positions || []).join(',');
    const p   = s.pending_requests || 0;
    const pb  = p === 0 ? 0 : p <= 3 ? 1 : p <= 7 ? 2 : 3;
    const c   = s.congestion_prediction || 0.2;
    const cb  = c < 0.35 ? 0 : c <= 0.70 ? 1 : 2;
    return `${pos}:${pb}:${cb}`;
  }

  actionKey(a) { return `${a.elevator_id}:${a.target_floor}`; }

  getDispatchAction(s, validActions, heuristicScores = null) {
    if (!validActions || validActions.length === 0) return null;
    if (validActions.length === 1) return validActions[0];

    const sk = this.encodeState(s);
    if (Math.random() < this.epsilon) {
      if (heuristicScores && heuristicScores.length === validActions.length) {
        const total = heuristicScores.reduce((acc, h) => acc + Math.max(0.1, h), 0);
        let r = Math.random() * total;
        for (let i = 0; i < validActions.length; i++) {
          r -= Math.max(0.1, heuristicScores[i]);
          if (r <= 0) return validActions[i];
        }
      }
      return validActions[Math.floor(Math.random() * validActions.length)];
    }

    const sa = this.qTable.get(sk);
    if (!sa || sa.size === 0) {
      if (heuristicScores) {
        let bestIdx = 0, bestH = -Infinity;
        for (let i = 0; i < heuristicScores.length; i++) {
          if (heuristicScores[i] > bestH) { bestH = heuristicScores[i]; bestIdx = i; }
        }
        return validActions[bestIdx];
      }
      return validActions[0];
    }

    let bestAction = validActions[0], bestScore = -Infinity;
    for (let i = 0; i < validActions.length; i++) {
      const a = validActions[i];
      const qVal = sa.get(this.actionKey(a)) || 0;
      const hBonus = heuristicScores ? heuristicScores[i] * 0.01 : 0;
      const score = qVal + hBonus;
      if (score > bestScore) {
        bestScore = score;
        bestAction = a;
      }
    }
    return bestAction;
  }

  getAction(s) {
    const idleElevators = [];
    const occs = s.elevator_occupancy || [];
    const tgts = s.elevator_targets || [];
    for (let e = 0; e < this.numElevators; e++) {
      if ((occs[e] === 0 || occs[e] === undefined) && (tgts[e] === null || tgts[e] === undefined)) {
        idleElevators.push(e);
      }
    }
    const candidateActions = idleElevators.length > 0
      ? this.actionSpace.filter(a => idleElevators.includes(a.elevator_id))
      : this.actionSpace;

    if (Math.random() < this.epsilon) {
      return candidateActions[Math.floor(Math.random() * candidateActions.length)];
    }
    const sk = this.encodeState(s);
    const sa = this.qTable.get(sk);
    if (!sa || sa.size === 0) return candidateActions[Math.floor(Math.random() * candidateActions.length)];
    let best = candidateActions[0], maxQ = -Infinity;
    for (const a of candidateActions) {
      const q = sa.get(this.actionKey(a)) || 0;
      if (q > maxQ) { maxQ = q; best = a; }
    }
    return best;
  }

  update(s, a, reward, ns) {
    if (!a) return;
    const sk  = this.encodeState(s);
    const nsk = this.encodeState(ns);
    const ak  = this.actionKey(a);
    if (!this.qTable.has(sk)) this.qTable.set(sk, new Map());
    const sa  = this.qTable.get(sk);
    const cur = sa.get(ak) || 0;
    const nsa = this.qTable.get(nsk);
    const maxNQ = nsa && nsa.size > 0 ? Math.max(...nsa.values()) : 0;
    sa.set(ak, cur + this.alpha * (reward + this.gamma * maxNQ - cur));
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  getStats() {
    return { q_table_size: this.qTable.size, epsilon: parseFloat(this.epsilon.toFixed(4)) };
  }
}

// ── 3. CONGESTION ESTIMATOR ───────────────────────────────────────────────────
function computeCongestion(state) {
  const hour = new Date().getHours();
  let base = 0.2;
  if      (hour >= 8  && hour < 10) base = 0.75;
  else if (hour >= 12 && hour < 14) base = 0.55;
  else if (hour >= 17 && hour < 20) base = 0.70;
  else if (hour >= 22 || hour < 6)  base = 0.05;
  const load   = Math.min(0.4, ((state.pending_requests || 0) / 15) * 0.4);
  const jitter = (Math.random() - 0.5) * 0.06;
  return parseFloat(Math.max(0.02, Math.min(0.98, base + load + jitter)).toFixed(4));
}

// ── 4. SYSTEM RUNTIME ─────────────────────────────────────────────────────────
let env   = new ElevatorEnvironment();
let agent = new QLearningAgent(env.numElevators, env.numFloors);
let isRunning        = false;
let totalDispatched  = 0;
let simulationInterval = null;

/**
 * Checks whether ALL elevators are empty AND no pending requests exist.
 * When idle, we should NOT move any elevator.
 */
function isSystemIdle() {
  if (env.pendingRequests.length > 0) return false;
  for (const passengers of env.elevatorPassengers) {
    if (passengers.length > 0) return false;
  }
  for (const t of env.committedTarget) {
    if (t !== null) return false;
  }
  if (env.assignedStops) {
    for (const stops of env.assignedStops) {
      if (stops.length > 0) return false;
    }
  }
  return true;
}

/**
 * Runs one RL dispatch step.
 * ONLY moves elevators when there is actual work to do
 * (pending requests or passengers already on board).
 * Returns null when idle so callers can detect no-op.
 */
function runRlStep() {
  // ── IDLE GUARD: do nothing if no passengers and no pending requests ──
  if (isSystemIdle()) {
    return null;
  }

  const s  = env.getState();
  env.congestionPrediction = computeCongestion(s);

  // 2. Centralized dispatch for any unassigned requests
  const unassigned = env.getUnassignedRequests();
  let dispatchAction = null;
  for (const req of unassigned) {
    const { validActions, heuristicScores } = env.getDispatchCandidates(req);
    if (validActions.length > 0) {
      const act = agent.getDispatchAction(s, validActions, heuristicScores);
      if (act) {
        env.assignRequest(req, act.elevator_id);
        if (!dispatchAction) dispatchAction = act;
      }
    }
  }

  // 3. Step environment
  const { newState, reward, info } = env.step(dispatchAction);
  if (dispatchAction) {
    agent.update(s, dispatchAction, reward, newState);
    totalDispatched++;
  }
  return { reward, info };
}

function getSystemState() {
  const s = env.getState();
  const t = agent.getStats();
  return {
    ...s,
    congestion_level: s.congestion_prediction,
    q_table_size:     t.q_table_size,
    epsilon:          t.epsilon,
    is_running:       isRunning,
    total_dispatched: totalDispatched,
  };
}

function startLoop() {
  if (simulationInterval) clearInterval(simulationInterval);
  simulationInterval = setInterval(() => {
    if (!isRunning) return;
    // ── Only run an RL step when there is work to do.
    //    Do NOT auto-spawn phantom passengers here;
    //    all requests must come from the frontend dispatcher.
    runRlStep();
  }, 1200);
}

function stopLoop() {
  if (simulationInterval) { clearInterval(simulationInterval); simulationInterval = null; }
}

// ── 5. HTTP SERVER ────────────────────────────────────────────────────────────
const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const pathname = url.parse(req.url, true).pathname;

  const json = (code, data) => {
    res.writeHead(code, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  };

  const body = (cb) => {
    let s = '';
    req.on('data', (c) => s += c);
    req.on('end', () => { try { cb(null, s ? JSON.parse(s) : {}); } catch(e) { cb(e, {}); } });
  };

  if (pathname === '/api/state' && req.method === 'GET') {
    return json(200, getSystemState());
  }

  if (pathname === '/api/start' && req.method === 'POST') {
    isRunning = true;
    env.reset();
    startLoop();
    return json(200, { status: 'success', message: 'System started', is_running: true, state: getSystemState() });
  }

  if (pathname === '/api/reset' && req.method === 'POST') {
    isRunning = false;
    stopLoop();
    env.reset();
    agent.reinitialize(env.numElevators, env.numFloors);
    return json(200, { status: 'success', message: 'System reset', is_running: false, state: getSystemState() });
  }

  if (pathname === '/api/configure' && req.method === 'POST') {
    body((err, data) => {
      if (err) return json(400, { error: 'Invalid JSON' });
      const nf = data.num_floors != null ? parseInt(data.num_floors, 10) : null;
      const ne = data.num_elevators != null ? parseInt(data.num_elevators, 10) : null;
      const nc = data.capacity    != null ? parseInt(data.capacity, 10) : null;

      isRunning = false;
      stopLoop();
      env.reconfigure({ numFloors: nf, numElevators: ne, capacity: nc });
      agent.reinitialize(env.numElevators, env.numFloors);

      return json(200, {
        status: 'success',
        message: `Reconfigured: ${env.numFloors} floors, ${env.numElevators} elevators, capacity ${env.capacity}`,
        state: getSystemState(),
      });
    });
    return;
  }

  if (pathname === '/api/request' && req.method === 'POST') {
    body((err, data) => {
      if (err) return json(400, { error: 'Invalid JSON' });
      const src = parseInt(data.src_floor, 10);
      const dst = parseInt(data.dst_floor, 10);
      if (!src || !dst || src === dst || src < 1 || src > env.numFloors || dst < 1 || dst > env.numFloors) {
        return json(400, { status: 'error', message: `Invalid floors (1–${env.numFloors}), source ≠ destination` });
      }
      env.addRequest(src, dst);
      totalDispatched++;
      // After adding the request the system is no longer idle – dispatch immediately
      const result = runRlStep();
      const { reward = 0, info = {} } = result || {};
      return json(200, { status: 'success', message: `Dispatched F${src}→F${dst}`, reward, info, state: getSystemState() });
    });
    return;
  }

  if (pathname === '/api/step' && req.method === 'POST') {
    // Respect idle guard – only step if there is actual work
    const result = runRlStep();
    if (!result) {
      // System idle: return current state without touching elevator positions
      return json(200, { status: 'idle', reward: 0, info: { picked_up: 0, dropped_off: 0, movement: 0 }, state: getSystemState() });
    }
    return json(200, { status: 'success', reward: result.reward, info: result.info, state: getSystemState() });
  }

  if (pathname === '/api/stats' && req.method === 'GET') {
    const t = agent.getStats();
    const s = env.getState();
    return json(200, { q_table_size: t.q_table_size, epsilon: t.epsilon, pending_requests: s.pending_requests, avg_wait_time: s.avg_wait_time, total_dispatched: totalDispatched, current_congestion: s.congestion_prediction });
  }

  if (pathname === '/api/predict' && req.method === 'POST') {
    body(() => json(200, { congestion_level: computeCongestion(env.getState()) }));
    return;
  }

  json(404, { error: 'Not found', path: pathname });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`====================================================`);
  console.log(` Elevator Backend: http://127.0.0.1:${PORT}`);
  console.log(` Config: ${env.numFloors} floors | ${env.numElevators} elevators | cap ${env.capacity}`);
  console.log(` Limits: ${MIN_FLOORS}–${MAX_FLOORS} floors, ${MIN_ELEVATORS}–${MAX_ELEVATORS} elevators`);
  console.log(`====================================================`);
});

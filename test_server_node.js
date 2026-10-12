/**
 * Integration test for Node.js backend (backend/server.js)
 * Tests:
 * 1. 20 idle steps: zero movement, all elevators stationary.
 * 2. Single request: exactly 1 elevator assigned and moves, others stay stationary.
 * 3. Pickup and dropoff without unnecessary reversals.
 * 4. Concurrent request: idle elevator assigned, busy elevator not interrupted.
 */

const http = require('http');

function post(path, body = {}) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = http.request({
      hostname: '127.0.0.1',
      port: 5001,
      path: path,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data),
      },
    }, (res) => {
      let resData = '';
      res.on('data', chunk => resData += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(resData) });
        } catch (e) {
          resolve({ status: res.statusCode, raw: resData });
        }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

function get(path) {
  return new Promise((resolve, reject) => {
    http.get(`http://127.0.0.1:5001${path}`, (res) => {
      let resData = '';
      res.on('data', chunk => resData += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(resData) });
        } catch (e) {
          resolve({ status: res.statusCode, raw: resData });
        }
      });
    }).on('error', reject);
  });
}

async function runTests() {
  console.log('======================================================');
  console.log('NODE.JS BACKEND INTEGRATION TEST (PORT 5001)');
  console.log('======================================================');

  // Reset system
  let res = await post('/api/reset');
  if (res.status !== 200) throw new Error('Reset failed: ' + JSON.stringify(res));
  let state = res.body.state;
  console.log('Reset completed. Positions:', state.elevator_positions);
  if (JSON.stringify(state.elevator_positions) !== JSON.stringify([1, 1, 1])) {
    throw new Error('Positions should be [1, 1, 1]');
  }

  // TEST 1: 20 Idle steps
  console.log('\n[TEST 1] Running 20 simulation steps on idle building...');
  for (let i = 1; i <= 20; i++) {
    const stepRes = await post('/api/step');
    if (stepRes.body.status !== 'idle') {
      throw new Error(`Step ${i} status was ${stepRes.body.status}, expected 'idle'`);
    }
    if (stepRes.body.info.movement !== 0) {
      throw new Error(`Step ${i} movement was ${stepRes.body.info.movement}, expected 0`);
    }
    if (JSON.stringify(stepRes.body.state.elevator_positions) !== JSON.stringify([1, 1, 1])) {
      throw new Error(`Step ${i} positions changed! ${JSON.stringify(stepRes.body.state.elevator_positions)}`);
    }
  }
  console.log('  [PASS] 20 idle steps: 0 movement, all positions remained [1, 1, 1].');

  // TEST 2: Single request Floor 3 -> Floor 8
  console.log('\n[TEST 2] Submitting single request: Floor 3 -> Floor 8...');
  const reqRes = await post('/api/request', { src_floor: 3, dst_floor: 8 });
  state = reqRes.body.state;
  console.log('  Request response: Positions =', state.elevator_positions, 'Targets =', state.elevator_targets);

  const activeTargets = state.elevator_targets.map((t, idx) => ({ t, idx })).filter(item => item.t !== null);
  if (activeTargets.length !== 1) {
    throw new Error(`Expected exactly 1 elevator with active target, got: ${JSON.stringify(state.elevator_targets)}`);
  }
  const assignedIdx = activeTargets[0].idx;
  const otherIdxs = [0, 1, 2].filter(i => i !== assignedIdx);
  console.log(`  Assigned elevator: Lift ${assignedIdx + 1}`);

  // Step until passenger drops off at Floor 8
  let reachedDropoff = false;
  let prevFloor = state.elevator_positions[assignedIdx];
  for (let step = 1; step <= 15; step++) {
    const stepRes = await post('/api/step');
    state = stepRes.body.state;
    const currFloor = state.elevator_positions[assignedIdx];
    console.log(`  Step ${step}: Lift ${assignedIdx + 1} at F${currFloor}, Positions:`, state.elevator_positions, 'Targets:', state.elevator_targets);

    // Verify other elevators NEVER moved from floor 1
    for (const other of otherIdxs) {
      if (state.elevator_positions[other] !== 1) {
        throw new Error(`Other elevator ${other + 1} moved to floor ${state.elevator_positions[other]}!`);
      }
    }

    if (currFloor < prevFloor) {
      throw new Error(`Lift ${assignedIdx + 1} reversed direction! Prev: ${prevFloor}, Curr: ${currFloor}`);
    }
    prevFloor = currFloor;

    if (currFloor === 8 && state.elevator_occupancy[assignedIdx] === 0 && state.elevator_targets[assignedIdx] === null) {
      reachedDropoff = true;
      console.log(`  Lift ${assignedIdx + 1} successfully dropped off passenger at Floor 8 and returned to idle!`);
      break;
    }
  }

  if (!reachedDropoff) {
    throw new Error('Passenger did not complete journey to Floor 8!');
  }
  console.log('  [PASS] Single request handled exclusively by 1 elevator with no other elevator movement.');

  // TEST 3: Concurrent request handling
  console.log('\n[TEST 3] Testing concurrent requests without conflict...');
  // Lift assignedIdx is at Floor 8, others are at Floor 1
  // Send Lift assignedIdx down to Floor 2 with passenger
  // First reset to make clean setup:
  await post('/api/reset');
  // Add req 1: F1 -> F9
  const firstReq = await post('/api/request', { src_floor: 1, dst_floor: 9 });
  const firstElev = firstReq.body.state.elevator_targets.findIndex(t => t !== null);
  console.log(`  Lift ${firstElev + 1} assigned to first request F1->F9.`);
  // Step 3 times
  await post('/api/step');
  await post('/api/step');
  let midState = (await post('/api/step')).body.state;
  console.log(`  Lift ${firstElev + 1} is at Floor ${midState.elevator_positions[firstElev]} (heading to 9 with passenger). Submitting concurrent request F2->F7...`);
  let concRes = (await post('/api/request', { src_floor: 2, dst_floor: 7 })).body.state;
  console.log('  After concurrent request: Targets:', concRes.elevator_targets, 'Positions:', concRes.elevator_positions);
  
  if (concRes.elevator_targets[firstElev] !== 9) {
    throw new Error(`Lift ${firstElev + 1} target was overwritten to ${concRes.elevator_targets[firstElev]}`);
  }
  // Check that one of the other idle elevators was assigned
  const otherElevs = [0, 1, 2].filter(i => i !== firstElev);
  const secondElev = otherElevs.find(i => concRes.elevator_targets[i] !== null);
  if (secondElev === undefined) {
    throw new Error('Neither idle elevator was assigned to concurrent request!');
  }
  const thirdElev = otherElevs.find(i => i !== secondElev);
  if (concRes.elevator_targets[thirdElev] !== null) {
    throw new Error(`Third elevator ${thirdElev + 1} should remain idle!`);
  }
  console.log(`  Idle Lift ${secondElev + 1} successfully assigned without disturbing Lift ${firstElev + 1}. Third Lift ${thirdElev + 1} remains idle.`);
  console.log('  [PASS] Concurrent request handled without conflict or target overwriting.');

  // TEST 4: Dynamic batching along existing route (F1->F9 + F4->F8)
  console.log('\n[TEST 4] Testing dynamic route batching: F1->F9 en route, F4->F8 batched onto same lift...');
  await post('/api/reset');
  // Add req 1: F1 -> F9 (boards at F1 and moves to F2 because /api/request calls runRlStep)
  const req1Res = await post('/api/request', { src_floor: 1, dst_floor: 9 });
  const liftIdx = req1Res.body.state.elevator_positions.findIndex(p => p === 2);
  console.log(`  Lift ${liftIdx + 1} at Floor 2 heading to Floor 9. Occupancy: ${req1Res.body.state.elevator_occupancy[liftIdx]}`);

  // Step 1: moves from F2 to F3
  const stepRes = await post('/api/step');
  console.log(`  Lift ${liftIdx + 1} at Floor ${stepRes.body.state.elevator_positions[liftIdx]} heading to Floor 9.`);

  // Now submit F4 -> F8 along the upward trajectory
  // Because /api/request runs 1 step, Lift moves from F3 to F4, boarding passenger 2!
  console.log('  Submitting request F4 -> F8 along the same upward trajectory...');
  const batchRes = (await post('/api/request', { src_floor: 4, dst_floor: 8 })).body.state;
  console.log('  Assigned stops:', batchRes.assigned_stops);
  console.log('  Elevator targets:', batchRes.elevator_targets);
  console.log('  Elevator positions:', batchRes.elevator_positions);
  console.log('  Elevator occupancies:', batchRes.elevator_occupancy);

  // Verify that only liftIdx was assigned and others remain idle
  const otherLifts = [0, 1, 2].filter(i => i !== liftIdx);
  for (const o of otherLifts) {
    if (batchRes.elevator_targets[o] !== null) {
      throw new Error(`Elevator ${o + 1} was assigned when Lift ${liftIdx + 1} should have batched the request!`);
    }
  }

  // Lift liftIdx is now at F4 with both passengers (occupancy 2) and next stops [8, 9] (target 8)
  if (batchRes.elevator_positions[liftIdx] !== 4 || batchRes.elevator_occupancy[liftIdx] !== 2) {
    throw new Error(`Expected Lift ${liftIdx + 1} at F4 with 2 passengers; got pos=${batchRes.elevator_positions[liftIdx]}, occ=${batchRes.elevator_occupancy[liftIdx]}`);
  }
  if (batchRes.elevator_targets[liftIdx] !== 8) {
    throw new Error(`Expected Lift ${liftIdx + 1} target F8; got ${batchRes.elevator_targets[liftIdx]}`);
  }
  console.log(`  Lift ${liftIdx + 1} correctly batched F4->F8 at F4! Onboard=${batchRes.elevator_occupancy[liftIdx]}, Target=${batchRes.elevator_targets[liftIdx]}`);

  // Step to F5, F6, F7, F8
  await post('/api/step'); // F5
  await post('/api/step'); // F6
  await post('/api/step'); // F7
  let s8 = (await post('/api/step')).body.state; // F8
  if (s8.elevator_positions[liftIdx] !== 8 || s8.elevator_occupancy[liftIdx] !== 1 || s8.elevator_targets[liftIdx] !== 9) {
    throw new Error(`Expected Lift ${liftIdx + 1} at F8 with 1 passenger and target 9; got pos=${s8.elevator_positions[liftIdx]}, occ=${s8.elevator_occupancy[liftIdx]}, target=${s8.elevator_targets[liftIdx]}`);
  }
  console.log(`  Step arrived at F8: dropped off Passenger 2! Occupancy = ${s8.elevator_occupancy[liftIdx]}, Next Target = ${s8.elevator_targets[liftIdx]}`);

  // Step to F9
  let s9 = (await post('/api/step')).body.state; // F9
  if (s9.elevator_positions[liftIdx] !== 9 || s9.elevator_occupancy[liftIdx] !== 0 || s9.elevator_targets[liftIdx] !== null) {
    throw new Error(`Expected Lift ${liftIdx + 1} at F9 idle with 0 passengers; got pos=${s9.elevator_positions[liftIdx]}, occ=${s9.elevator_occupancy[liftIdx]}, target=${s9.elevator_targets[liftIdx]}`);
  }
  console.log(`  Step arrived at F9: dropped off Passenger 1! Occupancy = ${s9.elevator_occupancy[liftIdx]}, returned to idle.`);

  // Verify other lifts remained at Floor 1 the entire time
  for (const o of otherLifts) {
    if (s9.elevator_positions[o] !== 1) {
      throw new Error(`Other lift ${o + 1} moved to ${s9.elevator_positions[o]} during batching journey!`);
    }
  }
  console.log('  Other lifts remained stationary at Floor 1 throughout the entire batched trip.');
  console.log('  [PASS] Dynamic batching along existing route succeeded with 100% precision.');

  console.log('\n======================================================');
  console.log('ALL NODE.JS BACKEND TESTS PASSED SUCCESSFULLY!');
  console.log('======================================================\n');
}

runTests().catch(err => {
  console.error('[TEST ERROR]', err);
  process.exit(1);
});

// Node.js runner to instantly generate training_data.csv conforming to generate_training_data.py
const fs = require('fs');
const path = require('path');

const NUM_ROWS = 50000;
const NUM_FLOORS = 10;
const NUM_ELEVATORS = 3;
const CAPACITY = 8;
const BASE_WAIT = 15.0;

function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

function getScenarioAndCongestion(hour, minute) {
  if (Math.random() < 0.02) {
    return { scenario: 'sudden_surge', congestion: 0.8 + Math.random() * 0.2 };
  }
  if (hour >= 22 || hour < 6) {
    return { scenario: 'night_idle', congestion: Math.random() * 0.08 };
  }
  if (hour >= 8 && hour < 10) {
    return { scenario: 'morning_peak', congestion: 0.7 + Math.random() * 0.25 };
  }
  if (hour >= 12 && hour < 14) {
    return { scenario: 'lunch_rush', congestion: 0.4 + Math.random() * 0.25 };
  }
  if (hour >= 17 && hour < 20) {
    return { scenario: 'evening_peak', congestion: 0.6 + Math.random() * 0.3 };
  }
  return { scenario: 'normal', congestion: 0.1 + Math.random() * 0.2 };
}

function generateFloorRequests(scenario, congestion) {
  let totalRequests = 0;
  if (scenario === 'night_idle') {
    totalRequests = Math.floor(Math.random() * 3);
  } else if (scenario === 'morning_peak') {
    totalRequests = Math.round((25 + Math.random() * 20) * (congestion / 0.8));
  } else if (scenario === 'evening_peak') {
    totalRequests = Math.round((20 + Math.random() * 20) * (congestion / 0.75));
  } else if (scenario === 'lunch_rush') {
    totalRequests = Math.round((15 + Math.random() * 15) * (congestion / 0.5));
  } else if (scenario === 'sudden_surge') {
    totalRequests = Math.round((35 + Math.random() * 20) * (congestion / 0.9));
  } else {
    totalRequests = Math.round((5 + Math.random() * 10) * (congestion / 0.2));
  }

  totalRequests = Math.max(0, totalRequests);
  const floorReqs = new Array(NUM_FLOORS).fill(0);
  if (totalRequests === 0) return { floorReqs, total: 0 };

  if (scenario === 'morning_peak') {
    const lobbyRatio = 0.6 + Math.random() * 0.2;
    const lobbyCount = Math.floor(totalRequests * lobbyRatio);
    floorReqs[0] = lobbyCount;
    for (let i = 0; i < totalRequests - lobbyCount; i++) {
      const idx = 1 + Math.floor(Math.random() * (NUM_FLOORS - 1));
      floorReqs[idx]++;
    }
  } else if (scenario === 'evening_peak') {
    const upperRatio = 0.6 + Math.random() * 0.2;
    const upperCount = Math.floor(totalRequests * upperRatio);
    for (let i = 0; i < upperCount; i++) {
      const idx = [7, 8, 9][Math.floor(Math.random() * 3)];
      floorReqs[idx]++;
    }
    for (let i = 0; i < totalRequests - upperCount; i++) {
      const idx = Math.floor(Math.random() * 7);
      floorReqs[idx]++;
    }
  } else if (scenario === 'lunch_rush') {
    const peakRatio = 0.5 + Math.random() * 0.2;
    const peakCount = Math.floor(totalRequests * peakRatio);
    for (let i = 0; i < peakCount; i++) {
      const idx = Math.random() < 0.5 ? 0 : 4;
      floorReqs[idx]++;
    }
    for (let i = 0; i < totalRequests - peakCount; i++) {
      const idx = Math.floor(Math.random() * NUM_FLOORS);
      floorReqs[idx]++;
    }
  } else if (scenario === 'sudden_surge') {
    const start = Math.floor(Math.random() * (NUM_FLOORS - 1));
    const clusterRatio = 0.7 + Math.random() * 0.2;
    const clusterCount = Math.floor(totalRequests * clusterRatio);
    for (let i = 0; i < clusterCount; i++) {
      const idx = Math.random() < 0.5 ? start : start + 1;
      floorReqs[idx]++;
    }
    for (let i = 0; i < totalRequests - clusterCount; i++) {
      const idx = Math.floor(Math.random() * NUM_FLOORS);
      floorReqs[idx]++;
    }
  } else {
    for (let i = 0; i < totalRequests; i++) {
      const idx = Math.floor(Math.random() * NUM_FLOORS);
      floorReqs[idx]++;
    }
  }

  const actualTotal = floorReqs.reduce((a, b) => a + b, 0);
  return { floorReqs, total: actualTotal };
}

function run() {
  const headers = [
    'timestamp', 'hour', 'minute', 'day_of_week',
    'floor_req_f1', 'floor_req_f2', 'floor_req_f3', 'floor_req_f4', 'floor_req_f5',
    'floor_req_f6', 'floor_req_f7', 'floor_req_f8', 'floor_req_f9', 'floor_req_f10',
    'total_requests',
    'elev1_floor', 'elev2_floor', 'elev3_floor',
    'elev1_occ', 'elev2_occ', 'elev3_occ',
    'avg_wait_time', 'traffic_scenario', 'congestion_level'
  ];

  let currentTime = new Date(Date.UTC(2024, 0, 1, 0, 0, 0));
  const elevFloors = [1, 5, 10];
  const elevDirs = [1, 1, -1];
  const elevOccs = [0, 0, 0];

  const targetDirs = [
    path.join(__dirname),
    path.join(__dirname, 'backend')
  ];

  targetDirs.forEach(dir => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  });

  const streams = targetDirs.map(dir => {
    const p = path.join(dir, 'training_data.csv');
    const ws = fs.createWriteStream(p, { encoding: 'utf8' });
    ws.write(headers.join(',') + '\n');
    return ws;
  });

  for (let i = 0; i < NUM_ROWS; i++) {
    const hour = currentTime.getUTCHours();
    const minute = currentTime.getUTCMinutes();
    const dayOfWeek = (currentTime.getUTCDay() + 6) % 7; // Monday = 0

    const { scenario, congestion: baseCongestion } = getScenarioAndCongestion(hour, minute);
    const noisy = baseCongestion + gaussianRandom(0, 0.05);
    const congestionLevel = Math.max(0.0, Math.min(1.0, parseFloat(noisy.toFixed(4))));

    const { floorReqs, total: totalRequests } = generateFloorRequests(scenario, congestionLevel);

    for (let e = 0; e < NUM_ELEVATORS; e++) {
      const steps = [0, 1, 1, 2];
      const step = steps[Math.floor(Math.random() * steps.length)];
      elevFloors[e] += step * elevDirs[e];
      if (elevFloors[e] >= NUM_FLOORS) {
        elevFloors[e] = NUM_FLOORS;
        elevDirs[e] = -1;
      } else if (elevFloors[e] <= 1) {
        elevFloors[e] = 1;
        elevDirs[e] = 1;
      }
      let occ = Math.round(congestionLevel * CAPACITY + (Math.random() * 3 - 1.5));
      elevOccs[e] = Math.max(0, Math.min(CAPACITY, occ));
    }

    const waitNoise = gaussianRandom(0, 2.0);
    const avgWaitTime = Math.max(2.0, parseFloat((BASE_WAIT * (1.0 + congestionLevel * 4.0) + waitNoise).toFixed(2)));
    const isoTimestamp = currentTime.toISOString().replace('.000Z', 'Z');

    const line = [
      isoTimestamp, hour, minute, dayOfWeek,
      ...floorReqs,
      totalRequests,
      elevFloors[0], elevFloors[1], elevFloors[2],
      elevOccs[0], elevOccs[1], elevOccs[2],
      avgWaitTime, scenario, congestionLevel
    ].join(',') + '\n';

    streams.forEach(s => s.write(line));

    currentTime = new Date(currentTime.getTime() + 60000);
  }

  streams.forEach(s => s.end());
  console.log(`Dataset generated: ${NUM_ROWS} rows`);
}

run();

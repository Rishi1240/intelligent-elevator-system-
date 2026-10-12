/**
 * API client for Elevator System backend (Flask).
 * Configured for http://localhost:5000 with resilient error handling.
 */

const API_BASE_URL = 'http://localhost:5001/api';

export const elevatorApi = {
  async getState() {
    const res = await fetch(`${API_BASE_URL}/state`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch state`);
    return await res.json();
  },

  async startSystem() {
    const res = await fetch(`${API_BASE_URL}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to start system`);
    return await res.json();
  },

  async resetSystem() {
    const res = await fetch(`${API_BASE_URL}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to reset system`);
    return await res.json();
  },

  async dispatchRequest(srcFloor, dstFloor) {
    const res = await fetch(`${API_BASE_URL}/request`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ src_floor: srcFloor, dst_floor: dstFloor }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to submit request`);
    return await res.json();
  },

  async stepSimulation() {
    const res = await fetch(`${API_BASE_URL}/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to step simulation`);
    return await res.json();
  },

  async getStats() {
    const res = await fetch(`${API_BASE_URL}/stats`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to get stats`);
    return await res.json();
  },

  async predictCongestion(sequence) {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to predict congestion`);
    return await res.json();
  },

  async configureSystem({ numFloors, numElevators, capacity } = {}) {
    const res = await fetch(`${API_BASE_URL}/configure`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_floors: numFloors, num_elevators: numElevators, capacity }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to configure system`);
    return await res.json();
  },
};

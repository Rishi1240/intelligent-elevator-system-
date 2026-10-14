import React, { useState, useEffect, useCallback, useRef } from 'react';
import ElevatorShaft from './components/ElevatorShaft';
import CongestionMeter from './components/CongestionMeter';
import RequestPanel from './components/RequestPanel';
import StatsPanel from './components/StatsPanel';
import SimulationTimeControl from './components/SimulationTimeControl';
import { elevatorApi } from './api/elevatorApi';

// ── hard limits mirrored from server ─────────────────────────────────────────
const MAX_FLOORS    = 20;
const MAX_ELEVATORS = 5;
const MIN_FLOORS    = 2;
const MIN_ELEVATORS = 1;

/**
 * Main Application Component for AI Elevator System Dashboard.
 * Supports dynamic floor and elevator configuration (2–20 floors, 1–5 lifts).
 */
export default function App() {
  // ── backend-synced state ──────────────────────────────────────────────────
  const [elevatorPositions, setElevatorPositions] = useState([1, 1, 1]);
  const [occupancy,         setOccupancy]         = useState([0, 0, 0]);
  const [assignedStops,     setAssignedStops]     = useState([[], [], []]);
  const [elevatorTargets,   setElevatorTargets]   = useState([null, null, null]);
  const [congestionLevel,   setCongestionLevel]   = useState(0.2);
  const [pendingRequests,   setPendingRequests]   = useState(0);
  const [avgWaitTime,       setAvgWaitTime]       = useState(0.0);
  const [qTableSize,        setQTableSize]        = useState(0);
  const [epsilon,           setEpsilon]           = useState(0.9);
  const [lastAction,        setLastAction]        = useState(null);
  const [isRunning,         setIsRunning]         = useState(false);
  const [numFloors,         setNumFloors]         = useState(10);
  const [numElevators,      setNumElevators]      = useState(3);
  const [capacity,          setCapacity]          = useState(8);
  // ── batch window state ──────────────────────────────────────────
  const [batchActive,       setBatchActive]       = useState(false);
  const [batchCountdownMs,  setBatchCountdownMs]  = useState(0);
  const [batchQueued,       setBatchQueued]       = useState(0);
  // ── emergency state ──────────────────────────────────────────────
  const [activeEmergencies, setActiveEmergencies] = useState([]);
  const [currentEmergency,  setCurrentEmergency]  = useState(null);

  // ── simulation clock & predictive telemetry state ─────────────────
  const [clockState,          setClockState]          = useState(null);
  const [lstmMode,            setLstmMode]            = useState('loaded');
  const [lastPredictionTime,  setLastPredictionTime]  = useState('');
  const [predictionHorizon,   setPredictionHorizon]   = useState(5);
  const [predictiveEnabled,   setPredictiveEnabled]   = useState(true);
  const [positioningStatus,   setPositioningStatus]   = useState('inactive');
  const [targetFloor,         setTargetFloor]         = useState(null);
  const [targetDemand,        setTargetDemand]        = useState(0.0);
  const [assignedElevators,   setAssignedElevators]   = useState([]);
  const [elevatorOpsStatus,   setElevatorOpsStatus]   = useState([]);
  const [floorDemandT5,       setFloorDemandT5]       = useState([]);
  const [assignedAllocations, setAssignedAllocations] = useState([]);
  const [eligibleElevatorsCount, setEligibleElevatorsCount] = useState(0);
  const [lastDecisionLog,     setLastDecisionLog]     = useState('');
  const [highThreshold,       setHighThreshold]       = useState(0.55);

  // ── UI state ──────────────────────────────────────────────────────────────
  const [logs,             setLogs]             = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('Checking backend...');
  const [showConfig,       setShowConfig]       = useState(false);
  const [configuring,      setConfiguring]      = useState(false);

  // ── config form inputs (draft – only applied on submit) ───────────────────
  const [draftFloors,    setDraftFloors]    = useState(10);
  const [draftElevators, setDraftElevators] = useState(3);
  const [draftCapacity,  setDraftCapacity]  = useState(8);

  const isRunningRef = useRef(isRunning);
  isRunningRef.current = isRunning;

  const addLog = (message) => {
    if (!message) return;
    const t = new Date().toLocaleTimeString();
    setLogs((prev) => {
      if (prev.length > 0 && prev[0].replace(/^\[.*?\]\s*/, '') === message) {
        return prev;
      }
      return [`[${t}] ${message}`, ...prev.slice(0, 49)];
    });
  };

  const applyState = (data) => {
    if (!data) return;
    if (Array.isArray(data.elevator_positions)) setElevatorPositions(data.elevator_positions);
    if (Array.isArray(data.elevator_occupancy)) setOccupancy(data.elevator_occupancy);
    if (Array.isArray(data.assigned_stops))     setAssignedStops(data.assigned_stops);
    if (Array.isArray(data.elevator_targets))   setElevatorTargets(data.elevator_targets);
    if (typeof data.congestion_level === 'number') setCongestionLevel(data.congestion_level);
    if (typeof data.congestion_prediction === 'number') setCongestionLevel(data.congestion_prediction);
    if (typeof data.pending_requests === 'number') setPendingRequests(data.pending_requests);
    if (typeof data.avg_wait_time === 'number') setAvgWaitTime(data.avg_wait_time);
    if (typeof data.q_table_size === 'number') setQTableSize(data.q_table_size);
    if (typeof data.epsilon === 'number') setEpsilon(data.epsilon);
    if (data.last_action !== undefined) setLastAction(data.last_action);
    if (typeof data.is_running === 'boolean') setIsRunning(data.is_running);
    // ── config fields ──
    if (typeof data.num_floors === 'number')    setNumFloors(data.num_floors);
    if (typeof data.num_elevators === 'number') setNumElevators(data.num_elevators);
    if (typeof data.capacity === 'number')      setCapacity(data.capacity);
    // ── batch window fields ──
    if (typeof data.batch_active === 'boolean')       setBatchActive(data.batch_active);
    if (typeof data.batch_countdown_ms === 'number')  setBatchCountdownMs(data.batch_countdown_ms);
    if (typeof data.batch_queued === 'number')        setBatchQueued(data.batch_queued);
    // ── emergency priority fields ──
    if (Array.isArray(data.active_emergencies))       setActiveEmergencies(data.active_emergencies);
    if (data.current_emergency !== undefined)         setCurrentEmergency(data.current_emergency);
    if (Array.isArray(data.emergency_events) && data.emergency_events.length > 0) {
      data.emergency_events.forEach((ev) => addLog(ev));
    }
    if (Array.isArray(data.dispatch_events) && data.dispatch_events.length > 0) {
      data.dispatch_events.forEach((ev) => addLog(ev));
    }
    // ── clock & predictive fields ──
    if (data.clock) setClockState(data.clock);
    if (data.lstm_mode) setLstmMode(data.lstm_mode);
    if (data.last_prediction_time) setLastPredictionTime(data.last_prediction_time);
    if (typeof data.prediction_horizon_min === 'number') setPredictionHorizon(data.prediction_horizon_min);
    if (typeof data.predictive_positioning_enabled === 'boolean') setPredictiveEnabled(data.predictive_positioning_enabled);
    if (data.predictive_positioning_status) setPositioningStatus(data.predictive_positioning_status);
    if (data.predicted_target_floor !== undefined) setTargetFloor(data.predicted_target_floor);
    if (typeof data.predicted_target_demand === 'number') setTargetDemand(data.predicted_target_demand);
    if (Array.isArray(data.assigned_predictive_elevators)) setAssignedElevators(data.assigned_predictive_elevators);
    if (Array.isArray(data.elevator_operational_status)) setElevatorOpsStatus(data.elevator_operational_status);
    if (Array.isArray(data.floor_demand_t5)) setFloorDemandT5(data.floor_demand_t5);
    if (Array.isArray(data.assigned_allocations)) setAssignedAllocations(data.assigned_allocations);
    if (typeof data.eligible_elevators_count === 'number') setEligibleElevatorsCount(data.eligible_elevators_count);
    if (typeof data.last_decision_log === 'string') setLastDecisionLog(data.last_decision_log);
    if (typeof data.high_congestion_threshold === 'number') setHighThreshold(data.high_congestion_threshold);
  };

  const fetchState = useCallback(async () => {
    try {
      const data = await elevatorApi.getState();
      applyState(data);
      setConnectionStatus('Connected to Backend (Port 5000)');
    } catch {
      setConnectionStatus('Backend Disconnected – start: python backend/app.py');
    }
  }, []);

  // Poll state every 2 s
  useEffect(() => {
    fetchState();
    const id = setInterval(fetchState, 2000);
    return () => clearInterval(id);
  }, [fetchState]);

  // Trigger sim steps when running (steps faster at 1.5s during emergency for responsiveness)
  useEffect(() => {
    if (!isRunning) return;
    const intervalMs = (activeEmergencies.length > 0 || currentEmergency) ? 1500 : 3000;
    const id = setInterval(async () => {
      try {
        const res = await elevatorApi.stepSimulation();
        if (res?.state) applyState(res.state);
      } catch { /* backend offline */ }
    }, intervalMs);
    return () => clearInterval(id);
  }, [isRunning, activeEmergencies.length, currentEmergency]);

  // ── handlers ──────────────────────────────────────────────────────────────
  const handleToggleRunning = async () => {
    try {
      if (!isRunning) {
        const res = await elevatorApi.startSystem();
        applyState(res.state);
        setIsRunning(true);
        addLog('Simulation started.');
      } else {
        const res = await elevatorApi.resetSystem();
        applyState(res.state);
        setIsRunning(false);
        addLog('Simulation stopped and reset.');
      }
    } catch (err) {
      addLog(`Error: ${err.message}`);
      setIsRunning(!isRunning);
    }
  };

  const handleRequestSubmit = async (src, dst) => {
    try {
      const res = await elevatorApi.dispatchRequest(src, dst);
      if (res?.state) {
        applyState(res.state);
      }
      setIsRunning(true);
    } catch (err) {
      addLog(`Dispatch error: ${err.message}`);
      setPendingRequests((p) => p + 1);
    }
  };

  const handleEmergencySubmit = async (floor) => {
    addLog(`🚨 [EMERGENCY] Request triggered at Floor ${floor}`);
    try {
      const res = await elevatorApi.dispatchEmergency(floor);
      if (res?.state) applyState(res.state);
      setIsRunning(true);
      if (res?.emergency?.elevator_id !== null && res?.emergency?.elevator_id !== undefined) {
        const liftNum = res.emergency.elevator_id + 1;
        const curFloor = res.emergency.current_floor ?? '—';
        const dist = res.emergency.distance ?? '—';
        addLog(`🚨 [EMERGENCY] Assigned to Lift ${liftNum} (At Floor ${curFloor}, distance ${dist} floors)`);
        if (res.emergency.rationale) {
          addLog(`ℹ️ [SELECTION] ${res.emergency.rationale}`);
        }
      } else if (res?.message) {
        addLog(`🚨 [EMERGENCY] ${res.message}`);
      }
    } catch (err) {
      addLog(`🚨 Emergency Error: ${err.message}`);
    }
  };

  const handleResolveEmergency = async (emergencyId, floor) => {
    addLog(`🚨 Resolving emergency...`);
    try {
      const res = await elevatorApi.resolveEmergency(emergencyId, floor);
      if (res?.state) applyState(res.state);
      addLog(`✅ [RESOLVED] Emergency on Floor ${res?.emergency?.floor || floor || ''} resolved. Normal dispatch restored.`);
    } catch (err) {
      addLog(`Resolve Emergency Error: ${err.message}`);
    }
  };

  const handleConfigure = async (e) => {
    e.preventDefault();
    setConfiguring(true);
    try {
      const res = await elevatorApi.configureSystem({
        numFloors:    draftFloors,
        numElevators: draftElevators,
        capacity:     draftCapacity,
      });
      applyState(res.state);
      addLog(`Reconfigured → ${draftFloors} floors, ${draftElevators} lifts, cap ${draftCapacity}`);
      setShowConfig(false);
    } catch (err) {
      addLog(`Configure error: ${err.message}`);
    } finally {
      setConfiguring(false);
    }
  };

  // ── styles ─────────────────────────────────────────────────────────────────
  const inputStyle = {
    backgroundColor: '#0f172a',
    color: '#e2e8f0',
    border: '1px solid #475569',
    borderRadius: '6px',
    padding: '7px 10px',
    fontSize: '14px',
    width: '70px',
    textAlign: 'center',
    outline: 'none',
  };

  const labelStyle = {
    fontSize: '11px',
    color: '#94a3b8',
    textTransform: 'uppercase',
    fontWeight: '600',
    marginBottom: '4px',
    display: 'block',
  };

  return (
    <div
      style={{
        backgroundColor: '#1a1a2e',
        minHeight: '100vh',
        color: '#f8fafc',
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        padding: '24px',
        boxSizing: 'border-box',
      }}
    >
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>

        {/* ── Header ── */}
        <header
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '20px',
            borderBottom: '1px solid #334155',
            paddingBottom: '16px',
            flexWrap: 'wrap',
            gap: '12px',
          }}
        >
          <div>
            <h1
              style={{
                margin: '0 0 4px 0',
                fontSize: '22px',
                fontWeight: '800',
                background: 'linear-gradient(90deg, #38bdf8, #818cf8)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              AI Elevator System — Q-Learning + LSTM
            </h1>
            <div style={{ fontSize: '12px', color: '#94a3b8' }}>
              {numElevators} lifts · {numFloors} floors · cap {capacity} · RL dispatch + neural congestion
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <span
              style={{
                fontSize: '11px',
                padding: '4px 10px',
                borderRadius: '12px',
                backgroundColor: connectionStatus.includes('Connected') ? '#064e3b' : '#7f1d1d',
                color: connectionStatus.includes('Connected') ? '#6ee7b7' : '#fca5a5',
                fontWeight: '600',
              }}
            >
              {connectionStatus}
            </span>

            {/* Configure button */}
            <button
              onClick={() => {
                setDraftFloors(numFloors);
                setDraftElevators(numElevators);
                setDraftCapacity(capacity);
                setShowConfig((v) => !v);
              }}
              style={{
                backgroundColor: '#334155',
                color: '#e2e8f0',
                border: '1px solid #475569',
                padding: '8px 14px',
                borderRadius: '8px',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'background 0.2s',
              }}
            >
              ⚙ Configure
            </button>

            {/* Start / Stop */}
            <button
              onClick={handleToggleRunning}
              style={{
                backgroundColor: isRunning ? '#ef4444' : '#10b981',
                color: '#ffffff',
                border: 'none',
                padding: '10px 20px',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: 'bold',
                cursor: 'pointer',
                boxShadow: isRunning
                  ? '0 0 12px rgba(239, 68, 68, 0.5)'
                  : '0 0 12px rgba(16, 185, 129, 0.5)',
                transition: 'all 0.2s',
              }}
            >
              {isRunning ? '⏹ Stop System' : '▶ Start System'}
            </button>
          </div>
        </header>

        {/* ── Configuration Panel ── */}
        {showConfig && (
          <form
            onSubmit={handleConfigure}
            style={{
              backgroundColor: '#16213e',
              border: '1px solid #475569',
              borderRadius: '12px',
              padding: '20px 24px',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'flex-end',
              gap: '24px',
              flexWrap: 'wrap',
            }}
          >
            <div style={{ color: '#818cf8', fontWeight: '700', fontSize: '13px', width: '100%', marginBottom: '-8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              🏢 Building Configuration
              <span style={{ fontSize: '11px', color: '#64748b', marginLeft: '12px', textTransform: 'none', fontWeight: '400' }}>
                Lifts 1–{MAX_ELEVATORS} · Floors 2–{MAX_FLOORS} · Resets simulation
              </span>
            </div>

            <div>
              <label style={labelStyle}>Num Floors</label>
              <input
                type="number"
                min={MIN_FLOORS}
                max={MAX_FLOORS}
                value={draftFloors}
                onChange={(e) => setDraftFloors(Math.max(MIN_FLOORS, Math.min(MAX_FLOORS, parseInt(e.target.value, 10) || MIN_FLOORS)))}
                style={inputStyle}
              />
            </div>

            <div>
              <label style={labelStyle}>Num Lifts</label>
              <input
                type="number"
                min={MIN_ELEVATORS}
                max={MAX_ELEVATORS}
                value={draftElevators}
                onChange={(e) => setDraftElevators(Math.max(MIN_ELEVATORS, Math.min(MAX_ELEVATORS, parseInt(e.target.value, 10) || MIN_ELEVATORS)))}
                style={inputStyle}
              />
            </div>

            <div>
              <label style={labelStyle}>Capacity / Lift</label>
              <input
                type="number"
                min={1}
                max={20}
                value={draftCapacity}
                onChange={(e) => setDraftCapacity(Math.max(1, Math.min(20, parseInt(e.target.value, 10) || 1)))}
                style={inputStyle}
              />
            </div>

            {/* Slider for floors */}
            <div style={{ flexGrow: 1 }}>
              <label style={labelStyle}>Floors (slider)</label>
              <input
                type="range"
                min={MIN_FLOORS}
                max={MAX_FLOORS}
                value={draftFloors}
                onChange={(e) => setDraftFloors(parseInt(e.target.value, 10))}
                style={{ width: '100%', accentColor: '#818cf8' }}
              />
            </div>

            {/* Slider for lifts */}
            <div style={{ flexGrow: 1 }}>
              <label style={labelStyle}>Lifts (slider)</label>
              <input
                type="range"
                min={MIN_ELEVATORS}
                max={MAX_ELEVATORS}
                value={draftElevators}
                onChange={(e) => setDraftElevators(parseInt(e.target.value, 10))}
                style={{ width: '100%', accentColor: '#38bdf8' }}
              />
            </div>

            <button
              type="submit"
              disabled={configuring}
              style={{
                backgroundColor: '#7c3aed',
                color: '#fff',
                border: 'none',
                padding: '10px 22px',
                borderRadius: '8px',
                fontWeight: '700',
                fontSize: '14px',
                cursor: configuring ? 'not-allowed' : 'pointer',
                opacity: configuring ? 0.65 : 1,
                transition: 'all 0.2s',
              }}
            >
              {configuring ? 'Applying…' : 'Apply & Reset'}
            </button>
          </form>
        )}

        {/* ── Simulation Time and Predictive Control Panel ── */}
        <SimulationTimeControl
          clockState={clockState}
          isRunning={isRunning}
          lstmMode={lstmMode}
          lastPredictionTime={lastPredictionTime}
          predictionHorizon={predictionHorizon}
          predictiveEnabled={predictiveEnabled}
          onStateUpdate={applyState}
          addLog={addLog}
        />

        {/* ── Congestion Meter ── */}
        <CongestionMeter
          congestionLevel={congestionLevel}
          horizonMin={predictionHorizon}
          targetFloor={targetFloor}
          targetDemand={targetDemand}
          positioningStatus={positioningStatus}
          assignedElevators={assignedElevators}
          floorDemandT5={floorDemandT5}
          assignedAllocations={assignedAllocations}
          eligibleElevatorsCount={eligibleElevatorsCount}
          lastDecisionLog={lastDecisionLog}
          highThreshold={highThreshold}
          numFloors={numFloors}
        />

        {/* ── Stats Telemetry Panel ── */}
        <StatsPanel
          avgWaitTime={avgWaitTime}
          pendingRequests={pendingRequests}
          qTableSize={qTableSize}
          epsilon={epsilon}
          lastAction={lastAction}
        />

        {/* ── Shafts + Right Panel ── */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: `1fr 340px`,
            gap: '24px',
            alignItems: 'start',
          }}
        >
          {/* Elevator Shafts Visualizer */}
          <div
            style={{
              backgroundColor: '#16213e',
              padding: '20px',
              borderRadius: '12px',
              border: '1px solid #334155',
              overflowX: 'auto',
            }}
          >
            <div
              style={{
                textAlign: 'center',
                color: '#94a3b8',
                fontSize: '12px',
                fontWeight: '700',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                marginBottom: '16px',
              }}
            >
              Building Shaft Live Visualizer ({numFloors} Floors · {numElevators} Lifts)
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'flex-start',
                gap: '12px',
                flexWrap: 'wrap',
              }}
            >
              {elevatorPositions.map((pos, idx) => {
                const emgForThisLift = activeEmergencies.find(
                  (em) => em.elevator_id === idx && ['en_route', 'dispatched', 'arrived', 'assigned'].includes(em.status)
                );
                const isPredAssigned = assignedElevators.includes(idx);
                const opStatus = elevatorOpsStatus[idx] || (isPredAssigned ? 'repositioning' : 'idle');
                return (
                  <ElevatorShaft
                    key={idx}
                    elevatorId={idx + 1}
                    currentFloor={pos || 1}
                    occupancy={occupancy[idx] || 0}
                    capacity={capacity}
                    numFloors={numFloors}
                    assignedStops={assignedStops[idx] || []}
                    targetFloor={elevatorTargets[idx]}
                    emergencyInfo={emgForThisLift}
                    operationalStatus={opStatus}
                    predictiveTarget={isPredAssigned ? targetFloor : null}
                  />
                );
              })}
            </div>
          </div>

          {/* Right Column */}
          <div>
            <RequestPanel
              onSubmitRequest={handleRequestSubmit}
              onSubmitEmergency={handleEmergencySubmit}
              onResolveEmergency={handleResolveEmergency}
              isRunning={isRunning}
              numFloors={numFloors}
              elevatorPositions={elevatorPositions}
              elevatorTargets={elevatorTargets}
              batchActive={batchActive}
              batchCountdownMs={batchCountdownMs}
              batchQueued={batchQueued}
              activeEmergencies={activeEmergencies}
              currentEmergency={currentEmergency}
            />

            {/* Event Console */}
            <div
              style={{
                backgroundColor: '#1e293b',
                padding: '16px',
                borderRadius: '12px',
                border: '1px solid #334155',
              }}
            >
              <div
                style={{
                  color: '#38bdf8',
                  fontSize: '12px',
                  fontWeight: '700',
                  textTransform: 'uppercase',
                  marginBottom: '10px',
                }}
              >
                Event &amp; Dispatch Log
              </div>
              <div
                style={{
                  backgroundColor: '#0a0f1d',
                  borderRadius: '6px',
                  padding: '10px',
                  height: '190px',
                  overflowY: 'auto',
                  fontFamily: 'monospace',
                  fontSize: '11px',
                  color: '#cbd5e1',
                  border: '1px solid #1e293b',
                }}
              >
                {logs.length === 0 ? (
                  <div style={{ color: '#64748b' }}>Waiting for system events…</div>
                ) : (
                  logs.map((log, i) => {
                    const isEmergency = log.includes('🚨') || log.includes('EMERGENCY');
                    const isResolved = log.includes('✅') || log.includes('RESOLVED');
                    return (
                      <div
                        key={i}
                        style={{
                          marginBottom: '4px',
                          lineHeight: '1.4',
                          color: isEmergency ? '#f87171' : isResolved ? '#4ade80' : '#cbd5e1',
                          fontWeight: isEmergency || isResolved ? '700' : 'normal',
                          backgroundColor: isEmergency ? 'rgba(239, 68, 68, 0.1)' : 'transparent',
                          borderRadius: '4px',
                          padding: isEmergency ? '2px 4px' : '0',
                        }}
                      >
                        {log}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

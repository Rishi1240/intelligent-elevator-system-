import React, { useState, useEffect } from 'react';
import { elevatorApi } from '../api/elevatorApi';

/**
 * Simulation Time and Predictive Control Panel
 *
 * Provides authoritative simulation clock controls, timezone Asia/Kolkata display,
 * custom date/time selection, speed factors (1x, 2x, 5x), quick presets,
 * and live predictive telemetry badges.
 */
export default function SimulationTimeControl({
  clockState,
  isRunning,
  lstmMode = 'loaded',
  lastPredictionTime = '',
  predictionHorizon = 5,
  predictiveEnabled = true,
  onStateUpdate,
  addLog,
}) {
  // Current draft inputs (local state until "Set Simulation Time" is pressed)
  const [selectedDate, setSelectedDate] = useState(() => {
    return clockState?.calendar_date_str || new Date().toISOString().split('T')[0];
  });
  const [selectedTime, setSelectedTime] = useState(() => {
    return clockState?.calendar_time_str || '09:00:00 AM';
  });
  const [selectedMode, setSelectedMode] = useState(() => {
    return clockState?.mode || 'custom';
  });
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Sync draft inputs when backend clock date changes externally
  useEffect(() => {
    if (clockState?.calendar_date_str) {
      setSelectedDate(clockState.calendar_date_str);
    }
    if (clockState?.calendar_time_str) {
      setSelectedTime(clockState.calendar_time_str);
    }
    if (clockState?.mode) {
      setSelectedMode(clockState.mode);
    }
  }, [clockState?.calendar_date_str, clockState?.calendar_time_str, clockState?.mode]);

  // Derive human-readable weekday from selected date
  const derivedWeekday = (() => {
    if (!selectedDate) return '—';
    try {
      const parts = selectedDate.split('-');
      if (parts.length === 3) {
        const d = new Date(parseInt(parts[0], 10), parseInt(parts[1], 10) - 1, parseInt(parts[2], 10));
        return d.toLocaleDateString('en-US', { weekday: 'long' });
      }
    } catch { /* fallback */ }
    return '—';
  })();

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleApplyTime = async (customDate, customTime, customMode) => {
    const d = customDate || selectedDate;
    const t = customTime || selectedTime;
    const m = customMode || selectedMode;

    setIsSubmitting(true);
    try {
      const res = await elevatorApi.setClock({ date: d, time: t, mode: m });
      if (res?.state && onStateUpdate) {
        onStateUpdate(res.state);
      }
      if (addLog) {
        addLog(`⏰ [CLOCK] Set to ${d} ${t} (${m.toUpperCase()} Mode · Asia/Kolkata)`);
      }
    } catch (err) {
      if (addLog) addLog(`Clock Error: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleToggleMode = async (newMode) => {
    setSelectedMode(newMode);
    try {
      const res = await elevatorApi.setClockMode(newMode);
      if (res?.state && onStateUpdate) onStateUpdate(res.state);
      if (addLog) addLog(`⏰ [MODE] Switched to ${newMode === 'custom' ? 'Custom Simulation Time' : 'Real-Time Wall Clock'}`);
    } catch (err) {
      if (addLog) addLog(`Mode Error: ${err.message}`);
    }
  };

  const handleSpeedChange = async (speed) => {
    try {
      const res = await elevatorApi.setClockSpeed(speed);
      if (res?.state && onStateUpdate) onStateUpdate(res.state);
      if (addLog) addLog(`⚡ [SPEED] Simulation speed changed to ${speed}×`);
    } catch (err) {
      if (addLog) addLog(`Speed Error: ${err.message}`);
    }
  };

  const handleStart = async () => {
    try {
      const res = await elevatorApi.startClock();
      if (res?.state && onStateUpdate) onStateUpdate(res.state);
      if (addLog) addLog('▶ [CLOCK] Simulation clock started.');
    } catch (err) {
      if (addLog) addLog(`Start Error: ${err.message}`);
    }
  };

  const handlePause = async () => {
    try {
      const res = await elevatorApi.pauseClock();
      if (res?.state && onStateUpdate) onStateUpdate(res.state);
      if (addLog) addLog('⏸ [CLOCK] Simulation paused. Time and movement frozen.');
    } catch (err) {
      if (addLog) addLog(`Pause Error: ${err.message}`);
    }
  };

  const handleResume = async () => {
    try {
      const res = await elevatorApi.resumeClock();
      if (res?.state && onStateUpdate) onStateUpdate(res.state);
      if (addLog) addLog('▶ [CLOCK] Simulation resumed from exact paused moment.');
    } catch (err) {
      if (addLog) addLog(`Resume Error: ${err.message}`);
    }
  };

  const handleReset = async () => {
    setShowResetConfirm(false);
    try {
      const res = await elevatorApi.resetSystem();
      if (res?.state && onStateUpdate) onStateUpdate(res.state);
      if (addLog) addLog('🔄 [RESET] Elevator simulation reset to clean baseline.');
    } catch (err) {
      if (addLog) addLog(`Reset Error: ${err.message}`);
    }
  };

  // Quick Presets: Select predefined calendar scenario
  const handleQuickPreset = (presetName, presetDate, presetTime) => {
    setSelectedDate(presetDate);
    setSelectedTime(presetTime);
    setSelectedMode('custom');
    handleApplyTime(presetDate, presetTime, 'custom');
  };

  const currentSpeed = clockState?.speed || 1.0;
  const isClockRunning = clockState?.is_running ?? isRunning;
  const activeDateDisplay = clockState?.calendar_date_str || selectedDate;
  const activeTimeDisplay = clockState?.calendar_time_str || selectedTime;
  const activeWeekday = clockState?.weekday || derivedWeekday;

  // ── Styles ────────────────────────────────────────────────────────────────
  const cardStyle = {
    backgroundColor: '#16213e',
    border: '1px solid #334155',
    borderRadius: '12px',
    padding: '20px 24px',
    marginBottom: '20px',
    boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
  };

  const badgeStyle = (bgColor, textColor) => ({
    fontSize: '11px',
    fontWeight: '700',
    padding: '4px 9px',
    borderRadius: '12px',
    backgroundColor: bgColor,
    color: textColor,
    letterSpacing: '0.3px',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '5px',
  });

  const buttonStyle = (bg, hoverBg, border = 'none', color = '#ffffff') => ({
    backgroundColor: bg,
    color,
    border,
    padding: '8px 16px',
    borderRadius: '8px',
    fontSize: '13px',
    fontWeight: '700',
    cursor: 'pointer',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    transition: 'all 0.15s ease',
  });

  const inputStyle = {
    backgroundColor: '#0f172a',
    color: '#f8fafc',
    border: '1px solid #475569',
    borderRadius: '6px',
    padding: '7px 12px',
    fontSize: '13px',
    outline: 'none',
  };

  return (
    <div style={cardStyle}>
      {/* ── Header & Live Telemetry ── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', borderBottom: '1px solid #2d3748', paddingBottom: '14px', marginBottom: '16px' }}>
        <div>
          <h2 style={{ margin: '0 0 4px 0', fontSize: '17px', fontWeight: '800', background: 'linear-gradient(90deg, #38bdf8, #a78bfa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>⏱</span> Simulation Time and Predictive Control
          </h2>
          <div style={{ fontSize: '12px', color: '#94a3b8' }}>
            Authoritative Calendar Clock · Timezone: <strong style={{ color: '#cbd5e1' }}>Asia/Kolkata (IST)</strong> · Neural Forecast Horizon: <strong style={{ color: '#38bdf8' }}>+{predictionHorizon} min</strong>
          </div>
        </div>

        {/* Live Status Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span style={badgeStyle(isClockRunning ? '#064e3b' : '#451a03', isClockRunning ? '#6ee7b7' : '#fdba74')}>
            <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: isClockRunning ? '#10b981' : '#f59e0b', display: 'inline-block' }} />
            {isClockRunning ? 'RUNNING' : 'PAUSED'}
          </span>

          <span style={badgeStyle('#1e1b4b', '#c7d2fe')}>
            MODE: {clockState?.mode === 'real_time' ? 'REAL-TIME' : 'CUSTOM TIME'}
          </span>

          <span style={badgeStyle('#1f2937', '#e2e8f0')}>
            SPEED: {currentSpeed}×
          </span>

          <span style={badgeStyle(lstmMode === 'loaded' ? '#042f2e' : '#451a03', lstmMode === 'loaded' ? '#5eead4' : '#fed7aa')}>
            LSTM: {lstmMode.toUpperCase()}
          </span>

          <span style={badgeStyle(predictiveEnabled ? '#0f291e' : '#292524', predictiveEnabled ? '#86efac' : '#a8a29e')}>
            PREPOSITION: {predictiveEnabled ? 'BENEFIT-AWARE' : 'OFF'}
          </span>
        </div>
      </div>

      {/* ── Active Clock Banner ── */}
      <div style={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '12px 18px', marginBottom: '18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '700', letterSpacing: '0.5px' }}>
              Active Simulation Clock
            </div>
            <div style={{ fontSize: '20px', fontWeight: '800', color: '#38bdf8', fontFamily: 'monospace', letterSpacing: '0.5px' }}>
              {activeTimeDisplay}
            </div>
          </div>

          <div style={{ borderLeft: '1px solid #334155', paddingLeft: '14px' }}>
            <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '700', letterSpacing: '0.5px' }}>
              Simulated Date & Day
            </div>
            <div style={{ fontSize: '14px', fontWeight: '700', color: '#e2e8f0' }}>
              {activeWeekday}, {activeDateDisplay}
            </div>
          </div>

          <div style={{ borderLeft: '1px solid #334155', paddingLeft: '14px' }}>
            <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', fontWeight: '700', letterSpacing: '0.5px' }}>
              Elapsed Sim Time
            </div>
            <div style={{ fontSize: '14px', fontWeight: '700', color: '#a5b4fc', fontFamily: 'monospace' }}>
              {clockState?.sim_time_seconds ? `${clockState.sim_time_seconds}s` : '0.0s'}
            </div>
          </div>
        </div>

        {lastPredictionTime && (
          <div style={{ fontSize: '11px', color: '#64748b', textAlign: 'right' }}>
            Last LSTM Prediction:{' '}
            <span style={{ color: '#94a3b8', fontWeight: '600' }}>
              {lastPredictionTime.split('T')[1]?.split('.')[0] || lastPredictionTime}
            </span>
          </div>
        )}
      </div>

      {/* ── Time Controls & Form ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '18px' }}>
        {/* Column 1: Time Mode & Date/Time Inputs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {/* Mode Switcher */}
          <div>
            <label style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '700', display: 'block', marginBottom: '6px' }}>
              Time Mode
            </label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                type="button"
                onClick={() => handleToggleMode('custom')}
                style={{
                  ...buttonStyle(
                    selectedMode === 'custom' ? '#2563eb' : '#1e293b',
                    '#3b82f6',
                    selectedMode === 'custom' ? '1px solid #60a5fa' : '1px solid #475569'
                  ),
                  fontSize: '12px',
                  padding: '6px 12px',
                }}
              >
                📅 Custom Simulation Time
              </button>
              <button
                type="button"
                onClick={() => handleToggleMode('real_time')}
                style={{
                  ...buttonStyle(
                    selectedMode === 'real_time' ? '#2563eb' : '#1e293b',
                    '#3b82f6',
                    selectedMode === 'real_time' ? '1px solid #60a5fa' : '1px solid #475569'
                  ),
                  fontSize: '12px',
                  padding: '6px 12px',
                }}
              >
                🕒 Real-Time Clock
              </button>
            </div>
          </div>

          {/* Date Picker & Derived Weekday */}
          <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: '140px' }}>
              <label style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '700', display: 'block', marginBottom: '4px' }}>
                Simulation Date
              </label>
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                disabled={selectedMode === 'real_time'}
                style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' }}
              />
            </div>

            <div style={{ flex: 1, minWidth: '120px' }}>
              <label style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '700', display: 'block', marginBottom: '4px' }}>
                Derived Weekday
              </label>
              <div style={{ ...inputStyle, width: '100%', boxSizing: 'border-box', backgroundColor: '#1e293b', color: '#93c5fd', fontWeight: '600' }}>
                {derivedWeekday}
              </div>
            </div>
          </div>

          {/* Time Input */}
          <div>
            <label style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '700', display: 'block', marginBottom: '4px' }}>
              Simulation Starting Time (e.g., 09:00:00 AM)
            </label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="text"
                value={selectedTime}
                onChange={(e) => setSelectedTime(e.target.value)}
                placeholder="09:00:00 AM"
                disabled={selectedMode === 'real_time'}
                style={{ ...inputStyle, flex: 1, fontFamily: 'monospace', fontWeight: '700' }}
              />
              <button
                type="button"
                onClick={() => handleApplyTime()}
                disabled={isSubmitting || selectedMode === 'real_time'}
                style={{
                  ...buttonStyle('#3b82f6', '#2563eb'),
                  opacity: selectedMode === 'real_time' ? 0.5 : 1,
                  whiteSpace: 'nowrap',
                }}
              >
                Set Time
              </button>
            </div>
          </div>
        </div>

        {/* Column 2: Clock Lifecycle & Speed Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {/* Simulation Speed Buttons */}
          <div>
            <label style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '700', display: 'block', marginBottom: '6px' }}>
              Simulation Speed
            </label>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {[1.0, 2.0, 5.0].map((spd) => {
                const isActive = Math.abs(currentSpeed - spd) < 0.05;
                return (
                  <button
                    key={spd}
                    type="button"
                    onClick={() => handleSpeedChange(spd)}
                    style={{
                      ...buttonStyle(
                        isActive ? '#8b5cf6' : '#1e293b',
                        '#7c3aed',
                        isActive ? '1px solid #c4b5fd' : '1px solid #475569'
                      ),
                      padding: '7px 14px',
                      fontSize: '12px',
                    }}
                  >
                    {spd}× Speed {spd === 1.0 ? '(Real-time)' : spd === 2.0 ? '(2s/s)' : '(5s/s)'}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Clock Control Buttons */}
          <div>
            <label style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '700', display: 'block', marginBottom: '6px' }}>
              Simulation Clock Lifecycle
            </label>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {!isClockRunning ? (
                <button
                  type="button"
                  onClick={clockState?.sim_time_seconds > 0 ? handleResume : handleStart}
                  style={buttonStyle('#10b981', '#059669')}
                  title="Start or resume simulation time progression"
                >
                  ▶ {clockState?.sim_time_seconds > 0 ? 'Resume' : 'Start'}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handlePause}
                  style={buttonStyle('#f59e0b', '#d97706')}
                  title="Freeze simulation clock progression"
                >
                  ⏸ Pause
                </button>
              )}

              <button
                type="button"
                onClick={() => setShowResetConfirm(true)}
                style={buttonStyle('#ef4444', '#dc2626')}
                title="Reset simulation environment and clock"
              >
                🔄 Reset Simulation
              </button>
            </div>
          </div>

          {/* Quick Presets */}
          <div>
            <label style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', fontWeight: '700', display: 'block', marginBottom: '6px' }}>
              Quick Presets (Predefined Scenarios)
            </label>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                type="button"
                onClick={() => handleQuickPreset('Morning Rush', '2026-10-14', '09:00:00 AM')}
                style={{ ...buttonStyle('#1e293b', '#334155', '1px solid #475569', '#f1f5f9'), fontSize: '11px', padding: '5px 10px' }}
                title="Weekday 09:00:00 AM (High Lobby Arrival)"
              >
                🌅 Morning Rush (09:00 AM)
              </button>

              <button
                type="button"
                onClick={() => handleQuickPreset('Afternoon', '2026-10-14', '02:30:00 PM')}
                style={{ ...buttonStyle('#1e293b', '#334155', '1px solid #475569', '#f1f5f9'), fontSize: '11px', padding: '5px 10px' }}
                title="Weekday 02:30:00 PM (Moderate Steady Traffic)"
              >
                ☀️ Afternoon (02:30 PM)
              </button>

              <button
                type="button"
                onClick={() => handleQuickPreset('Evening', '2026-10-14', '05:30:00 PM')}
                style={{ ...buttonStyle('#1e293b', '#334155', '1px solid #475569', '#f1f5f9'), fontSize: '11px', padding: '5px 10px' }}
                title="Weekday 05:30:00 PM (Exit Rush to Lobby)"
              >
                🌆 Evening (05:30 PM)
              </button>

              <button
                type="button"
                onClick={() => handleQuickPreset('Night', '2026-10-14', '11:00:00 PM')}
                style={{ ...buttonStyle('#1e293b', '#334155', '1px solid #475569', '#f1f5f9'), fontSize: '11px', padding: '5px 10px' }}
                title="Weekday 11:00:00 PM (Idle Building)"
              >
                🌙 Night (11:00 PM)
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Confirmation Modal for Reset ── */}
      {showResetConfirm && (
        <div style={{ marginTop: '12px', padding: '12px 16px', borderRadius: '8px', backgroundColor: '#450a0a', border: '1px solid #b91c1c', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <span style={{ fontSize: '13px', color: '#fecaca', fontWeight: '600' }}>
            ⚠️ Are you sure you want to reset? This will clear all active passenger journeys and reset elevator positions to Floor 1.
          </span>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              type="button"
              onClick={handleReset}
              style={{ ...buttonStyle('#dc2626', '#b91c1c'), fontSize: '12px', padding: '5px 12px' }}
            >
              Confirm Reset
            </button>
            <button
              type="button"
              onClick={() => setShowResetConfirm(false)}
              style={{ ...buttonStyle('#334155', '#475569'), fontSize: '12px', padding: '5px 12px' }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

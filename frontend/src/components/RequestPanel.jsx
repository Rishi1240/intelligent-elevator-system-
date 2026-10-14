import React, { useState, useRef } from 'react';

/**
 * RequestPanel Component
 * Handles passenger summon requests: source floor and destination floor selection,
 * 2-second batching indicator, and Emergency Priority Request trigger/resolver.
 */
export default function RequestPanel({
  onSubmitRequest,
  onSubmitEmergency,
  onResolveEmergency,
  isRunning,
  numFloors = 10,
  elevatorPositions = [],
  elevatorTargets = [],
  batchActive = false,
  batchCountdownMs = 0,
  batchQueued = 0,
  activeEmergencies = [],
  currentEmergency = null,
}) {
  const [srcFloor, setSrcFloor] = useState(1);
  const [dstFloor, setDstFloor] = useState(Math.min(8, numFloors));
  const [submitting, setSubmitting] = useState(false);

  // Emergency local state
  const [emergencyFloor, setEmergencyFloor] = useState(1);
  const [submittingEmergency, setSubmittingEmergency] = useState(false);
  const [resolvingEmergency, setResolvingEmergency] = useState(false);

  // Debounce refs to prevent accidental duplicate rapid clicks
  const lastSubmitTimeRef = useRef(0);
  const lastEmgSubmitTimeRef = useRef(0);

  // Derive a display-friendly seconds value from the ms countdown
  const batchSecondsLeft = Math.ceil(batchCountdownMs / 1000);
  const batchColor = batchSecondsLeft > 1 ? '#f59e0b' : '#ef4444';

  // Build floor list from backend-reported numFloors
  const floorOptions = Array.from({ length: numFloors }, (_, i) => i + 1);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (srcFloor === dstFloor) {
      alert('Source floor and Destination floor must be different.');
      return;
    }

    const now = Date.now();
    if (now - lastSubmitTimeRef.current < 600) {
      return; // Debounce rapid clicks
    }
    lastSubmitTimeRef.current = now;

    setSubmitting(true);
    try {
      await onSubmitRequest(Number(srcFloor), Number(dstFloor));
    } catch (error) {
      console.error('Failed to dispatch passenger request:', error);
      alert('Failed to dispatch request. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleEmergencySubmit = async (e) => {
    e.preventDefault();
    if (!onSubmitEmergency) return;

    const now = Date.now();
    if (now - lastEmgSubmitTimeRef.current < 600) {
      return; // Debounce rapid emergency clicks
    }
    lastEmgSubmitTimeRef.current = now;

    setSubmittingEmergency(true);
    try {
      await onSubmitEmergency(Number(emergencyFloor));
    } catch (error) {
      console.error('Failed to trigger emergency:', error);
      alert('Failed to trigger emergency. Please try again.');
    } finally {
      setSubmittingEmergency(false);
    }
  };

  const handleResolveEmergency = async (emg) => {
    if (!onResolveEmergency) return;
    setResolvingEmergency(true);
    try {
      await onResolveEmergency(emg?.id, emg?.floor);
    } catch (error) {
      console.error('Failed to resolve emergency:', error);
      alert('Failed to resolve emergency. Please try again.');
    } finally {
      setResolvingEmergency(false);
    }
  };

  // Presets only LOAD the dropdowns — they do NOT auto-dispatch.
  const handleQuickPreset = (src, dst) => {
    const clampedSrc = Math.min(src, numFloors);
    const clampedDst = Math.min(dst, numFloors);
    if (clampedSrc === clampedDst) return;
    setSrcFloor(clampedSrc);
    setDstFloor(clampedDst);
  };

  const selectStyle = {
    backgroundColor: '#0f172a',
    color: '#e2e8f0',
    border: '1px solid #334155',
    padding: '8px 12px',
    borderRadius: '6px',
    fontSize: '14px',
    outline: 'none',
    width: '100%',
    cursor: 'pointer',
  };

  const quickBtnStyle = {
    backgroundColor: '#334155',
    color: '#e2e8f0',
    border: 'none',
    padding: '6px 10px',
    borderRadius: '6px',
    fontSize: '11px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  };

  // Active emergency record (if any)
  const activeEmg = currentEmergency || (activeEmergencies && activeEmergencies[0]);

  return (
    <div
      style={{
        backgroundColor: '#1e293b',
        padding: '18px 20px',
        borderRadius: '12px',
        border: '1px solid #334155',
        marginBottom: '20px',
      }}
    >
      <div
        style={{
          color: '#38bdf8',
          fontSize: '14px',
          fontWeight: '700',
          textTransform: 'uppercase',
          marginBottom: '14px',
          letterSpacing: '0.5px',
        }}
      >
        Passenger Request Dispatcher
      </div>

      <form
        onSubmit={handleSubmit}
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr auto',
          gap: '12px',
          alignItems: 'end',
        }}
      >
        <div>
          <label
            style={{
              display: 'block',
              fontSize: '12px',
              color: '#94a3b8',
              marginBottom: '6px',
            }}
          >
            Source Floor (Pickup)
          </label>
          <select
            value={srcFloor}
            onChange={(e) => setSrcFloor(Number(e.target.value))}
            style={selectStyle}
          >
            {floorOptions.map((f) => (
              <option key={`src-${f}`} value={f}>
                Floor {f}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label
            style={{
              display: 'block',
              fontSize: '12px',
              color: '#94a3b8',
              marginBottom: '6px',
            }}
          >
            Destination Floor (Dropoff)
          </label>
          <select
            value={dstFloor}
            onChange={(e) => setDstFloor(Number(e.target.value))}
            style={selectStyle}
          >
            {floorOptions.map((f) => (
              <option key={`dst-${f}`} value={f}>
                Floor {f}
              </option>
            ))}
          </select>
        </div>

        <button
          type="submit"
          disabled={submitting || isRunning === false}
          style={{
            backgroundColor: '#0284c7',
            color: '#ffffff',
            border: 'none',
            padding: '10px 18px',
            borderRadius: '6px',
            fontWeight: '600',
            fontSize: '14px',
            cursor: submitting || isRunning === false ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s',
            height: '38px',
            opacity: submitting || isRunning === false ? 0.6 : 1,
          }}
        >
          {submitting ? 'Dispatching...' : 'Dispatch'}
        </button>
      </form>

      {/* Quick traffic scenario presets */}
      <div
        style={{
          marginTop: '14px',
          display: 'flex',
          gap: '8px',
          alignItems: 'center',
          flexWrap: 'wrap',
        }}
      >
        <span style={{ fontSize: '11px', color: '#64748b', fontWeight: '600' }}>
          Quick Presets:
        </span>

        <button
          type="button"
          onClick={() => handleQuickPreset(1, numFloors)}
          style={quickBtnStyle}
          title="Load Morning Rush scenario into the form"
        >
          🌅 Morning Rush (F1→F{numFloors})
        </button>

        <button
          type="button"
          onClick={() => handleQuickPreset(Math.ceil(numFloors / 2), 1)}
          style={quickBtnStyle}
          title="Load Afternoon scenario into the form"
        >
          ☀️ Afternoon (F{Math.ceil(numFloors / 2)}→F1)
        </button>

        <button
          type="button"
          onClick={() => handleQuickPreset(Math.max(numFloors - 1, 2), 1)}
          style={quickBtnStyle}
          title="Load Evening scenario into the form"
        >
          🌆 Evening (F{Math.max(numFloors - 1, 2)}→F1)
        </button>

        <button
          type="button"
          onClick={() => handleQuickPreset(numFloors, 1)}
          style={quickBtnStyle}
          title="Load Night scenario into the form"
        >
          🌙 Night (F{numFloors}→F1)
        </button>
      </div>

      {/* 2-second batch window countdown badge */}
      {batchActive && (
        <div
          style={{
            marginTop: '12px',
            padding: '10px 14px',
            borderRadius: '8px',
            backgroundColor: '#1e293b',
            border: `1px solid ${batchColor}`,
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontSize: '12px',
            color: batchColor,
            fontWeight: '600',
          }}
        >
          <span style={{ fontSize: '16px' }}>⏳</span>
          <span>
            Batching window open — {batchQueued} request{batchQueued !== 1 ? 's' : ''} queued
            &nbsp;·&nbsp;
            dispatching in&nbsp;
            <span
              style={{
                fontVariantNumeric: 'tabular-nums',
                color: '#f8fafc',
                background: batchColor,
                borderRadius: '4px',
                padding: '1px 6px',
                fontSize: '11px',
              }}
            >
              {batchSecondsLeft}s
            </span>
          </span>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* EMERGENCY PRIORITY REQUEST SECTION                                  */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      <div
        style={{
          marginTop: '18px',
          paddingTop: '16px',
          borderTop: '1px solid #334155',
        }}
      >
        <div
          style={{
            color: '#f87171',
            fontSize: '13px',
            fontWeight: '700',
            textTransform: 'uppercase',
            marginBottom: '10px',
            letterSpacing: '0.5px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span>🚨</span> Emergency Priority Summon
        </div>

        {/* Active Emergency Alert Card */}
        {activeEmg ? (() => {
          const assignedId = activeEmg.elevator_id;
          const assignedLiftNum = (assignedId !== null && assignedId !== undefined) ? assignedId + 1 : null;
          const currentLiftFloor = (assignedId !== null && assignedId !== undefined && elevatorPositions[assignedId] !== undefined)
            ? elevatorPositions[assignedId]
            : (activeEmg.current_floor ?? '—');
          const emgFloor = activeEmg.floor;
          const distance = (typeof currentLiftFloor === 'number')
            ? Math.abs(currentLiftFloor - emgFloor)
            : (typeof activeEmg.distance === 'number' ? activeEmg.distance : null);
          const isArrived = activeEmg.status === 'arrived' || (typeof distance === 'number' && distance === 0);

          return (
            <div
              style={{
                backgroundColor: '#450a0a',
                border: '2px solid #ef4444',
                borderRadius: '8px',
                padding: '14px 16px',
                boxShadow: '0 0 15px rgba(239, 68, 68, 0.35)',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '8px',
                  marginBottom: '10px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '18px' }}>🚨</span>
                  <span style={{ fontWeight: '800', color: '#fca5a5', fontSize: '14px' }}>
                    ACTIVE EMERGENCY AT FLOOR {emgFloor}
                  </span>
                </div>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: '700',
                    padding: '3px 8px',
                    borderRadius: '12px',
                    backgroundColor: isArrived ? '#065f46' : '#991b1b',
                    color: isArrived ? '#6ee7b7' : '#fecaca',
                    textTransform: 'uppercase',
                    border: `1px solid ${isArrived ? '#10b981' : '#f87171'}`,
                  }}
                >
                  {isArrived ? '✓ ARRIVED AT FLOOR' : '⏳ EN ROUTE (HIGH PRIORITY)'}
                </span>
              </div>

              <div
                style={{
                  fontSize: '12px',
                  color: '#fed7aa',
                  lineHeight: '1.6',
                  marginBottom: '12px',
                  backgroundColor: 'rgba(0, 0, 0, 0.25)',
                  padding: '8px 12px',
                  borderRadius: '6px',
                }}
              >
                <div>
                  <strong>Assigned Elevator:</strong>{' '}
                  {assignedLiftNum ? `Lift ${assignedLiftNum}` : 'Searching for suitable elevator...'}
                  {assignedLiftNum && (
                    <span style={{ color: '#93c5fd', marginLeft: '8px' }}>
                      (Currently at Floor {currentLiftFloor})
                    </span>
                  )}
                </div>
                <div>
                  <strong>Live Distance:</strong>{' '}
                  <span style={{ color: isArrived ? '#4ade80' : '#fde047', fontWeight: '700' }}>
                    {distance !== null ? (distance === 0 ? '0 floors (Arrived)' : `${distance} floor${distance !== 1 ? 's' : ''} away`) : '—'}
                  </span>
                </div>
                <div>
                  <strong>Status:</strong>{' '}
                  <span style={{ color: isArrived ? '#4ade80' : '#f87171', fontWeight: '700' }}>
                    {isArrived ? 'Reserved & Stationary at Emergency Floor' : (currentLiftFloor < emgFloor ? '▲ Moving Upward to Floor ' + emgFloor : '▼ Moving Downward to Floor ' + emgFloor)}
                  </span>
                </div>
                {activeEmg.rationale && (
                  <div style={{ fontSize: '11px', color: '#fca5a5', marginTop: '4px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '4px' }}>
                    <strong>Dispatch Rationale:</strong> {activeEmg.rationale}
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  onClick={() => handleResolveEmergency(activeEmg)}
                  disabled={resolvingEmergency}
                  style={{
                    backgroundColor: '#10b981',
                    color: '#ffffff',
                    border: 'none',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    fontWeight: '700',
                    fontSize: '12px',
                    cursor: resolvingEmergency ? 'not-allowed' : 'pointer',
                    boxShadow: '0 0 10px rgba(16, 185, 129, 0.4)',
                    transition: 'background-color 0.2s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}
                >
                  <span>✓</span>
                  {resolvingEmergency ? 'Resolving...' : 'Resolve Emergency'}
                </button>
              </div>
            </div>
          );
        })() : (
          /* Normal trigger form */
          <form
            onSubmit={handleEmergencySubmit}
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr auto',
              gap: '12px',
              alignItems: 'end',
            }}
          >
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: '11px',
                  color: '#fca5a5',
                  marginBottom: '6px',
                  fontWeight: '600',
                }}
              >
                Select Emergency Location (Floor)
              </label>
              <select
                value={emergencyFloor}
                onChange={(e) => setEmergencyFloor(Number(e.target.value))}
                style={{
                  ...selectStyle,
                  borderColor: '#ef4444',
                  backgroundColor: '#18181b',
                }}
              >
                {floorOptions.map((f) => (
                  <option key={`emg-${f}`} value={f}>
                    Floor {f} (Emergency Call)
                  </option>
                ))}
              </select>
            </div>

            <button
              type="submit"
              disabled={submittingEmergency}
              style={{
                backgroundColor: '#dc2626',
                color: '#ffffff',
                border: '1px solid #ef4444',
                padding: '10px 18px',
                borderRadius: '6px',
                fontWeight: '700',
                fontSize: '13px',
                cursor: submittingEmergency ? 'not-allowed' : 'pointer',
                boxShadow: '0 0 12px rgba(220, 38, 38, 0.45)',
                transition: 'all 0.2s',
                height: '38px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <span>🚨</span>
              {submittingEmergency ? 'Summoning...' : 'Emergency Request'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
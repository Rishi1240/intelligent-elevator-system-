import React, { useState } from 'react';

/**
 * RequestPanel Component
 * Handles passenger summon requests: source floor and destination floor selection.
 * numFloors is passed dynamically from the backend state.
 */
export default function RequestPanel({ onSubmitRequest, isRunning, numFloors = 10 }) {
  const [srcFloor, setSrcFloor] = useState(1);
  const [dstFloor, setDstFloor] = useState(Math.min(8, numFloors));
  const [submitting, setSubmitting] = useState(false);

  // Build floor list from backend-reported numFloors
  const floorOptions = Array.from({ length: numFloors }, (_, i) => i + 1);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (srcFloor === dstFloor) {
      alert('Source floor and Destination floor must be different.');
      return;
    }

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

  // Presets only LOAD the dropdowns — they do NOT auto-dispatch.
  // The user still clicks the Dispatch button to actually send the request.
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
          onClick={() => handleQuickPreset(Math.max(numFloors - 1, 2), 1)}
          style={quickBtnStyle}
          title="Load Evening Exit scenario into the form"
        >
          🌆 Evening Exit (F{Math.max(numFloors - 1, 2)}→F1)
        </button>

        <button
          type="button"
          onClick={() => handleQuickPreset(Math.ceil(numFloors / 2), 1)}
          style={quickBtnStyle}
          title="Load Lunch Rush scenario into the form"
        >
          ☀️ Lunch Rush (F{Math.ceil(numFloors / 2)}→F1)
        </button>
      </div>
    </div>
  );
}
import React from 'react';

/**
 * StatsPanel Component
 * Displays system telemetry: average wait time, pending queue count,
 * Q-table exploration states, epsilon parameter, and last dispatch action.
 */
export default function StatsPanel({ avgWaitTime = 0, pendingRequests = 0, qTableSize = 0, epsilon = 0.9, lastAction = null }) {
  const cardStyle = {
    backgroundColor: '#0f172a',
    padding: '12px 16px',
    borderRadius: '8px',
    border: '1px solid #334155',
    flex: '1',
    minWidth: '130px',
  };

  const labelStyle = {
    fontSize: '11px',
    color: '#94a3b8',
    textTransform: 'uppercase',
    fontWeight: '600',
    marginBottom: '4px',
  };

  const valueStyle = {
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#f8fafc',
  };

  return (
    <div
      style={{
        backgroundColor: '#1e293b',
        padding: '16px 20px',
        borderRadius: '12px',
        border: '1px solid #334155',
        marginBottom: '20px',
      }}
    >
      <div
        style={{
          color: '#38bdf8',
          fontSize: '13px',
          fontWeight: '700',
          textTransform: 'uppercase',
          marginBottom: '12px',
          letterSpacing: '0.5px',
        }}
      >
        Real-Time RL & Building Telemetry
      </div>

      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        <div style={cardStyle}>
          <div style={labelStyle}>Avg Wait Time</div>
          <div style={{ ...valueStyle, color: avgWaitTime > 30 ? '#ef4444' : '#38bdf8' }}>
            {avgWaitTime}s
          </div>
        </div>

        <div style={cardStyle}>
          <div style={labelStyle}>Pending Queue</div>
          <div style={{ ...valueStyle, color: pendingRequests > 5 ? '#f59e0b' : '#10b981' }}>
            {pendingRequests} passengers
          </div>
        </div>

        <div style={cardStyle}>
          <div style={labelStyle}>Q-Table States</div>
          <div style={valueStyle}>{qTableSize} keys</div>
        </div>

        <div style={cardStyle}>
          <div style={labelStyle}>RL Epsilon</div>
          <div style={valueStyle}>{typeof epsilon === 'number' ? epsilon.toFixed(3) : epsilon}</div>
        </div>

        <div style={cardStyle}>
          <div style={labelStyle}>Last RL Action</div>
          <div style={{ ...valueStyle, fontSize: '13px', color: '#a855f7' }}>
            {lastAction
              ? `Elev ${lastAction.elevator_id + 1} -> F${lastAction.target_floor}`
              : 'Idle'}
          </div>
        </div>
      </div>
    </div>
  );
}

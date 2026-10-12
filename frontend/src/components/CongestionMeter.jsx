import React from 'react';

/**
 * CongestionMeter Component
 * Visualizes predicted congestion level from LSTM on a 0-100% scale.
 * Colors: green (<30%), yellow (30-69%), red (>=70%).
 */
export default function CongestionMeter({ congestionLevel = 0.0 }) {
  const percentage = Math.min(100, Math.max(0, Math.round(congestionLevel * 100)));

  let barColor = '#10b981'; // Green
  let label = 'Low Traffic (Optimal)';
  if (percentage >= 70) {
    barColor = '#ef4444'; // Red
    label = 'Severe Congestion (Peak Rush)';
  } else if (percentage >= 30) {
    barColor = '#f59e0b'; // Yellow
    label = 'Moderate Traffic';
  }

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
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '8px',
        }}
      >
        <span style={{ color: '#94a3b8', fontSize: '13px', fontWeight: '600', textTransform: 'uppercase' }}>
          LSTM Traffic Congestion Forecast
        </span>
        <span
          style={{
            fontSize: '14px',
            fontWeight: 'bold',
            color: barColor,
          }}
        >
          {percentage}% — {label}
        </span>
      </div>

      {/* Progress Bar Container */}
      <div
        style={{
          height: '14px',
          width: '100%',
          backgroundColor: '#0f172a',
          borderRadius: '7px',
          overflow: 'hidden',
          border: '1px solid #334155',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${percentage}%`,
            backgroundColor: barColor,
            borderRadius: '7px',
            transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.5s ease',
            boxShadow: `0 0 12px ${barColor}88`,
          }}
        />
      </div>
    </div>
  );
}
